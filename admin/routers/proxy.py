"""代理网关：带 API Key 的 /v1/chat/completions 与 /v1/models。

流程：校验 Key → 配额拦截（超额提示『积分已耗尽』）→ 从可用账号中挑选 →
用该账号凭据转发到后端 → 流式返回 → 按用量回扣 Key 额度。
"""
import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta

import httpx
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from admin import backend
from admin.config import settings
from admin.db import SessionLocal, get_db
from admin.models import Account, ApiKey, ModelConfig, UsageLog, UsageLogDetail
from admin.routers.models import _is_model_allowed
from admin.security import check_quota, get_key_row

HTTP_LIMITS = backend.HTTP_LIMITS

_logger = logging.getLogger("proxy")

# Responses API 适配器（converter 同款）；缺失时 /v1/responses 优雅降级为 501
try:
    from responses_adapter import responses_request_to_chat, ResponsesStreamConverter
    from responses_projection import project_responses_chat_body
    _RESPONSES_AVAILABLE = True
except Exception:  # pragma: no cover - 降级分支
    _RESPONSES_AVAILABLE = False
    responses_request_to_chat = None
    ResponsesStreamConverter = None
    project_responses_chat_body = None

router = APIRouter(tags=["proxy"])


def _client_ip(request: Request) -> str:
    """提取发起请求的真实客户端 IP。

    经反向代理部署时，上游往往会带上 X-Forwarded-For / X-Real-IP；
    取 XFF 首个（最原始客户端），否则 X-Real-IP，最后退回直连 socket 地址。
    这是「记录原客户端用户真实 IP」的关键，便于风控对账与上游用途日志对齐。
    """
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    real = request.headers.get("X-Real-IP")
    if real:
        return real.strip()
    return request.client.host if request.client else ""


def _upstream_extra_headers(request: Request, purpose: str = "conversation") -> dict:
    """构造需要透传给上游的风控/审计头。

    - X-Forwarded-For / X-Real-IP / X-Client-IP: 把原始客户端真实 IP 带上去，
      让上游请求用量里的「客户端」列能显示真实来源，而不是反代服务器 IP。
      若环境变量 ADMIN_UPSTREAM_CLIENT_HEADER 指定了自定义 header 名，则额外发送该头。
    - X-Agent-Purpose: WorkBuddy 请求用途头，用于上游用量分类；
      缺失时上游请求用量「用途」列为空，易被风控识别为异常调用。
      真实 WorkBuddy 桌面端普通对话使用 "conversation"。
    - X-IDE-Name / X-IDE-Type / X-Product: 上游记录到请求用量「client」列的产品名，
      缺失时该列为空；真实桌面端发 WorkBuddy，因此默认带 WorkBuddy。
    """
    ip = _client_ip(request)
    client_name = settings.UPSTREAM_CLIENT_NAME.strip() or "WorkBuddy"
    h: dict[str, str] = {
        "X-Agent-Purpose": purpose or "conversation",
        "X-IDE-Name": client_name,
        "X-IDE-Type": client_name,
        "X-Product": client_name,
    }
    if ip:
        h.update({
            "X-Forwarded-For": ip,
            "X-Real-IP": ip,
            "X-Client-IP": ip,
        })
        custom = settings.UPSTREAM_CLIENT_HEADER.strip()
        if custom:
            h[custom] = ip
    return h


def _record_usage(key_id: int, account_id: int, model: str, credits: float | None,
                  updated_auth_json: str | None, *,
                  client_ip: str = "", use_case: str = "",
                  prompt_tokens: int | None = None,
                  completion_tokens: int | None = None, total_tokens: int | None = None,
                  cached_tokens: int | None = None,
                  seq: int = 0, ttfb_ms: int | None = None,
                  latency_ms: int | None = None, error_kind: str = "",
                  http_status: int | None = None,
                  req_preview: str = "", resp_preview: str = "",
                  full_payload=None, full_response: str = "") -> int | None:
    """流式响应结束后独立开一个 DB 会话写入用量/额度。

    关键点：请求作用域的 db 会话在端点返回 StreamingResponse 时已被依赖 teardown 关闭，
    不能在流式生成器里复用它做 commit（会抛 ResourceClosedError 且被流式 except 静默吞掉）。
    这里用全新的 SessionLocal 落库，并把错误显式记录到日志，绝不再静默丢失。

    credits 语义：
    - None 表示上游未返回真实积分，此时按模型倍率估算；
    - 0.0 表示上游明确返回 0 或请求失败，不再估算。

    若本次使用了估算值，会启动后台线程在 60 秒后调用上游用量接口回写真实积分。
    """
    log_id = None
    try:
        db = SessionLocal()
        try:
            # 上游 usage 没给 credits 时，用本地模型配置的 credit_multiplier 估算。
            # credit_multiplier 在 model_configs 里保存的是「每千 token 积分」，
            # 因此估算公式为 total_tokens * multiplier / 1000；
            # 当 total_tokens 缺失时用 completion_tokens 兜底。
            estimated = False
            original_credits = credits
            if credits is None and (total_tokens or completion_tokens):
                mc = db.query(ModelConfig).filter(ModelConfig.model_id == (model or ""),
                                                   ModelConfig.enabled == 1).first()
                mult = mc.credit_multiplier if mc else 0
                if mult:
                    toks = total_tokens if total_tokens else completion_tokens
                    credits = float(toks) * mult / 1000.0
                    estimated = True
            if credits is None:
                credits = 0.0
            _logger.info("记录用量 model=%s raw_credits=%s est=%s pt=%s ct=%s tt=%s cached=%s client_ip=%s use_case=%s",
                         model, original_credits, credits, prompt_tokens, completion_tokens, total_tokens,
                         cached_tokens, client_ip, use_case)
            key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
            if key is not None:
                key.credit_used = float(key.credit_used or 0) + credits
            acc = db.query(Account).filter(Account.id == account_id).first()
            if acc is not None:
                acc.balance_remain = max(0, int(acc.balance_remain or 0) - int(credits))
                acc.last_used_at = datetime.utcnow()
                if updated_auth_json:
                    acc.auth_json = updated_auth_json
            log = UsageLog(
                api_key_id=key_id, account_id=account_id, model=model, credits=credits,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                total_tokens=total_tokens, cached_tokens=cached_tokens,
                client_ip=client_ip or "", use_case=use_case or "",
                seq=seq, ttfb_ms=ttfb_ms, latency_ms=latency_ms, error_kind=error_kind or "",
                http_status=http_status,
                request_preview=req_preview or None, response_preview=resp_preview or None,
            )
            db.add(log)
            db.commit()
            log_id = log.id
            # 完整上下文（系统提示 / 多轮历史 / 工具调用 / 原始报文）单独落详情表
            if full_payload is not None or full_response:
                _save_detail(log_id, full_payload, full_response)
            if estimated and account_id and updated_auth_json:
                threading.Thread(
                    target=_fetch_real_credits,
                    args=(log_id, account_id, updated_auth_json, model, credits, log.created_at),
                    daemon=True,
                ).start()
        finally:
            db.close()
    except Exception as e:  # 记账失败不应影响已返回的响应，但必须留痕便于排查
        _logger.exception("记录用量失败 key=%s acc=%s model=%s credits=%s: %s",
                          key_id, account_id, model, credits, e)
    return log_id


def _fetch_real_credits(log_id: int, account_id: int, auth_json: str, model: str,
                        estimated_credits: float, created_at: datetime) -> None:
    """延迟查询上游真实用量接口，回写 UsageLog.credits 并校正额度。

    上游 /billing/meter/get-user-request-usage 有分钟级延迟，通常在请求完成后 30~90s
    才能查到。这里等待 60s 后按 [created_at-5min, created_at+5min] + model 匹配最近一条。
    """
    try:
        time.sleep(60)
        sess = AccountSession(auth_json)
        try:
            start = (created_at - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
            end = (created_at + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
            result = sess.fetch_request_usage(start, end, page_num=1, page_size=50)
            data = (result.get("data") or {}).get("data") or []
            client_name = (settings.UPSTREAM_CLIENT_NAME or "WorkBuddy").strip() or "WorkBuddy"
            candidates = [
                r for r in data
                if r.get("model") == model and client_name in (r.get("client") or "")
            ]
            if not candidates:
                _logger.info("真实积分回写未找到匹配 log=%s model=%s", log_id, model)
                return
            # 取 requestTime 最接近 created_at 的一条
            def _ts(item):
                try:
                    return datetime.strptime(item.get("requestTime", ""), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    return datetime.min
            best = min(candidates, key=lambda r: abs((_ts(r) - created_at).total_seconds()))
            real = float(best.get("credit") or 0)
            _logger.info("真实积分匹配 log=%s model=%s real=%s est=%s requestTime=%s",
                         log_id, model, real, estimated_credits, best.get("requestTime"))
            db = SessionLocal()
            try:
                log = db.query(UsageLog).filter(UsageLog.id == log_id).first()
                if log is None:
                    return
                delta = real - log.credits
                if abs(delta) < 0.0001:
                    return
                log.credits = real
                key = db.query(ApiKey).filter(ApiKey.id == log.api_key_id).first()
                if key is not None:
                    key.credit_used = max(0.0, float(key.credit_used or 0) + delta)
                acc = db.query(Account).filter(Account.id == log.account_id).first()
                if acc is not None:
                    acc.balance_remain = max(0, int(acc.balance_remain or 0) - int(delta))
                db.commit()
                _logger.info("真实积分回写完成 log=%s real=%s delta=%s", log_id, real, delta)
            finally:
                db.close()
        finally:
            sess.close()
    except Exception as e:
        _logger.exception("真实积分回写失败 log=%s: %s", log_id, e)

# ---------------------------------------------------------------------------
# 请求级表格日志：每个 /v1/* chat 请求出口打印一行。
# seq 进程级递增；模型截断 11 字符；uid 只显示前 8 位。
# ---------------------------------------------------------------------------
import itertools

_CHAT_SEQ = itertools.count(1)


def _content_to_text(content) -> str:
    """把 message.content（字符串 / 多模态数组）压成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text") or p.get("content") or ""
                if isinstance(t, str):
                    parts.append(t)
        return " ".join(parts)
    return ""


# agent 类客户端（Codex / Claude Code / WorkBuddy）会把运行环境说明塞进 user 消息，
# 形如 <additional_data>...</additional_data>，直接预览会淹没真正的提问，这里先剥掉。
_AGENT_BLOCK_TAGS = (
    "additional_data", "environment", "system-reminder", "system_reminder",
    "context", "instructions", "background", "functions", "tool_calls",
)
_ENV_BLOCK_RE = re.compile(
    r"<(" + "|".join(_AGENT_BLOCK_TAGS) + r")\b[^>]*>.*?</\1\s*>",
    re.S | re.I,
)
# 兜底：无论标签名，剥掉开头那一段 <tag>...</tag>
_LEADING_BLOCK_RE = re.compile(r"^\s*<([A-Za-z_][\w:-]*)\b[^>]*>.*?</\1\s*>\s*", re.S)
# Claude Code / Codex 等客户端注入的方括号式提示，形如 [System reminder: ...]
_BRACKET_REMINDER_RE = re.compile(r"\[\s*System\s*[Rr]eminder:[^\]]*\]", re.S)


def _strip_env_blocks(text: str) -> str:
    """去掉 agent 运行时注入的环境块，只留下人类真正写的内容。"""
    s = _BRACKET_REMINDER_RE.sub("", text or "")
    for _ in range(5):
        new = _LEADING_BLOCK_RE.sub("", _ENV_BLOCK_RE.sub("", s))
        if new == s:
            break
        s = new
    return s


def _last_user_text(items) -> str:
    """从消息数组里取最后一条 user 消息的纯文本。

    同时兼容两种形态：
      - Chat Completions：messages=[{role, content}]
      - Responses：input=[{role, content:[{type:input_text,text}]}] 或纯字符串数组
    """
    if not isinstance(items, list):
        return ""
    for m in reversed(items):
        if isinstance(m, dict) and m.get("role") == "user":
            t = _content_to_text(m.get("content"))
            if t:
                return t
    # 没有 role 标记（部分 Responses 客户端）时退化为取最后一条有内容的
    for m in reversed(items):
        if isinstance(m, dict):
            t = _content_to_text(m.get("content") or m.get("text"))
            if t:
                return t
        elif isinstance(m, str) and m.strip():
            return m
    return ""


def _extract_input(payload) -> str:
    """从请求体里取「用户输入」：优先最后一条 user 消息，兼容 Responses 的 input。"""
    if not isinstance(payload, dict):
        return ""
    raw = _last_user_text(payload.get("messages"))
    if not raw:
        inp = payload.get("input")
        # Responses 的 input 可能是字符串，也可能是消息数组
        raw = _last_user_text(inp) if isinstance(inp, list) else _content_to_text(inp)
    stripped = _strip_env_blocks(raw).strip()
    return stripped or raw


def _extract_output(sse_text: str) -> str:
    """从上游 SSE 原文里拼出「模型输出」。

    同时兼容两种协议：
      - Chat Completions：choices[0].delta.content / choices[0].message.content
      - Responses：type=response.output_text.delta 的 delta 字段
    """
    buf: list[str] = []
    reason: list[str] = []
    errs: list[str] = []
    for line in (sse_text or "").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        # 上游限流/业务错误常在 HTTP 200 的 SSE 流里以 error 字段返回，
        # 必须一并捕获，否则这类失败在日志里表现为「空回复」。
        err = obj.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("msg") or ""
            if msg:
                errs.append(str(msg))
            continue
        elif isinstance(err, str) and err.strip():
            errs.append(err.strip())
            continue
        # Responses 协议
        if str(obj.get("type") or "").startswith("response.output_text.delta"):
            d = obj.get("delta")
            if isinstance(d, str):
                buf.append(d)
            continue
        # Chat Completions 协议
        try:
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            t = delta.get("content")
            if t:
                buf.append(t)
            # 思考型模型可能只回 reasoning_content，正文为空时用它兜底
            rc = delta.get("reasoning_content")
            if rc:
                reason.append(rc)
            msg = choice.get("message") or {}
            if not t and msg.get("content"):
                buf.append(msg["content"])
        except Exception:
            continue
    out = "".join(buf)
    if out:
        return out
    # 正常正文为空时，优先展示上游错误（限流/配额提示等），其次才是思考内容
    if errs:
        return "[上游错误] " + " ".join(dict.fromkeys(errs))
    if reason:
        return "[思考] " + "".join(reason)
    return ""


def _preview(text: str, limit: int) -> str:
    """压平换行并按 limit 截断，超出加省略号。"""
    s = " ".join((text or "").split())
    if limit > 0 and len(s) > limit:
        s = s[:limit] + "..."
    return s


def _store_preview(text: str) -> str:
    """按 LOG_STORE 截断对话内容用于入库；配置为 0 时不记录正文。"""
    limit = getattr(settings, "LOG_STORE", 1000)
    if not limit:
        return ""
    return _preview(text, limit)


# ---------------------------------------------------------------------------
# 完整上下文：系统提示 / 多轮历史 / 工具调用 / 模型输出 / 原始报文
# ---------------------------------------------------------------------------

def _clamp(s: str, max_bytes: int) -> str:
    """按字节上限截断字符串，避免把 UTF-8 字符切一半导致入库乱码。"""
    if not s:
        return ""
    b = s.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="ignore") + "\n... [已截断，超出 ADMIN_LOG_FULL_MAX]"


def _tool_calls_text(calls) -> str:
    """把 tool_calls 渲染成可读文本（chat 协议 assistant 消息里的字段）。"""
    if not isinstance(calls, list):
        return ""
    parts = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") or {}
        name = fn.get("name") or c.get("name") or "?"
        args = fn.get("arguments") or c.get("arguments") or ""
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = str(args)
        parts.append(f"{name}({args})")
    return "\n".join(parts)


def _norm_message(m) -> dict | None:
    """把 chat / responses 两种协议的消息项归一成统一结构。

    统一输出：{role, content, tool_calls(文本), name, tool_call_id, kind}
    """
    if isinstance(m, str):
        return {"role": "user", "content": m, "kind": "text"}
    if not isinstance(m, dict):
        return None
    # Responses 协议：带 type 的事件项
    mtype = str(m.get("type") or "")
    role = m.get("role") or ""
    if mtype == "function_call" or mtype == "tool_call":
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": _tool_calls_text([m]),
            "kind": "tool_call",
            "tool_call_id": m.get("call_id") or m.get("id") or "",
        }
    if mtype == "function_call_output" or mtype == "tool_result":
        return {
            "role": "tool",
            "content": _content_to_text(m.get("output") if m.get("output") is not None else m.get("content")),
            "kind": "tool_result",
            "tool_call_id": m.get("call_id") or m.get("tool_call_id") or "",
        }
    content = _content_to_text(m.get("content"))
    tc = _tool_calls_text(m.get("tool_calls")) if m.get("tool_calls") else ""
    if not content and not tc and not role:
        return None
    return {
        "role": role or ("assistant" if tc else "user"),
        "content": content,
        "tool_calls": tc,
        "name": m.get("name") or "",
        "tool_call_id": m.get("tool_call_id") or "",
        "kind": "tool_call" if tc and not content else ("tool_result" if role == "tool" else "text"),
    }


def _dump_messages(payload) -> list[dict]:
    """抽取完整提示词上下文：系统提示 + 多轮历史 + 工具调用与结果。"""
    out: list[dict] = []
    if not isinstance(payload, dict):
        return out
    # Responses 协议的 instructions / system 等价于 system prompt
    for key in ("instructions", "system"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            out.append({"role": "system", "content": v, "kind": "text"})
    items = None
    if isinstance(payload.get("messages"), list):
        items = payload["messages"]
    elif isinstance(payload.get("input"), list):
        items = payload["input"]
    elif isinstance(payload.get("input"), str):
        out.append({"role": "user", "content": payload["input"], "kind": "text"})
    for m in (items or []):
        n = _norm_message(m)
        if n:
            out.append(n)
    return out


def _scan_response(sse_text: str) -> dict:
    """扫描上游完整响应，拆出 正文 / 思考 / 工具调用 / usage。

    同时兼容 Chat Completions 与 Responses 两种 SSE 协议。
    """
    out: list[str] = []
    reason: list[str] = []
    calls: dict[int, dict] = {}
    usage: dict | None = None
    for line in (sse_text or "").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:"):].strip()
        if not body or body == "[DONE]":
            continue
        try:
            obj = json.loads(body)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        try:
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            # ---- Responses 协议 ----
            t = str(obj.get("type") or "")
            if t == "response.output_text.delta" and isinstance(obj.get("delta"), str):
                out.append(obj["delta"])
            elif t in ("response.reasoning_text.delta", "response.reasoning.delta") and isinstance(obj.get("delta"), str):
                reason.append(obj["delta"])
            elif t == "response.output_item.added":
                item = obj.get("item") or {}
                if item.get("type") in ("function_call", "tool_call"):
                    idx = len(calls)
                    calls[idx] = {
                        "name": item.get("name") or "?",
                        "arguments": item.get("arguments") or "",
                        "id": item.get("call_id") or item.get("id") or "",
                    }
            elif t == "response.function_call_arguments.delta" and isinstance(obj.get("delta"), str) and calls:
                calls[max(calls.keys())]["arguments"] += obj["delta"]
            elif t == "response.completed":
                resp = obj.get("response") or {}
                if isinstance(resp.get("usage"), dict):
                    usage = resp["usage"]
            # ---- Chat Completions 协议 ----
            ch = (obj.get("choices") or [])
            if ch:
                c0 = ch[0] if isinstance(ch[0], dict) else {}
                delta = c0.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    out.append(delta["content"])
                if isinstance(delta.get("reasoning_content"), str):
                    reason.append(delta["reasoning_content"])
                for tc in (delta.get("tool_calls") or []):
                    if not isinstance(tc, dict):
                        continue
                    idx = tc.get("index") or 0
                    cur = calls.setdefault(idx, {"name": "", "arguments": "", "id": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        cur["name"] = fn["name"]
                    if tc.get("id"):
                        cur["id"] = tc["id"]
                    if fn.get("arguments"):
                        cur["arguments"] += fn["arguments"]
                msg = c0.get("message") or {}
                if isinstance(msg.get("content"), str) and msg["content"] and not out:
                    out.append(msg["content"])
        except Exception:
            continue
    return {
        "output": "".join(out),
        "reasoning": "".join(reason),
        "tool_calls": [calls[k] for k in sorted(calls)],
        "usage": usage,
    }


def _build_detail(payload, resp_text: str) -> dict:
    """组装完整上下文（供后台弹窗展示）。"""
    scanned = _scan_response(resp_text)
    return {
        "messages": _dump_messages(payload),
        "output": scanned["output"],
        "reasoning": scanned["reasoning"],
        "tool_calls": scanned["tool_calls"],
        "usage": scanned["usage"],
    }


def _save_detail(log_id: int | None, payload, resp_text: str) -> None:
    """把完整上下文写入 usage_log_details（失败不影响主流程）。"""
    if not log_id or not getattr(settings, "LOG_FULL", True):
        return
    max_bytes = getattr(settings, "LOG_FULL_MAX", 2 * 1024 * 1024)
    try:
        detail = _build_detail(payload, resp_text)
        payload_json = json.dumps(detail, ensure_ascii=False)
        try:
            raw_req = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
        except Exception:
            raw_req = str(payload)
        db = SessionLocal()
        try:
            db.add(UsageLogDetail(
                log_id=log_id,
                payload_json=_clamp(payload_json, max_bytes),
                raw_request=_clamp(raw_req, max_bytes),
                raw_response=_clamp(resp_text or "", max_bytes),
                size_bytes=len(payload_json.encode("utf-8", errors="ignore")),
            ))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        _logger.warning("写入完整上下文失败 log=%s: %s", log_id, e)


def _log_chat_row(ttfb_ms, latency_ms, model, mode, uid, status, toks, error_kind="",
                  prompt_text: str = "", output_text: str = ""):
    """向 stdout 打印一行表格日志，便于排查慢请求与风控。

    ADMIN_LOG_PREVIEW > 0 时在末尾追加 输入 / 输出 两列（截断预览）。
    """
    seq = next(_CHAT_SEQ)
    now = datetime.now().strftime("%H:%M:%S")
    model = (model or "-")[:11]
    tok_field = "-" if toks is None else str(toks)
    tps = "-"
    if toks is not None and latency_ms and latency_ms > 0:
        tps = f"{toks * 1000 / latency_ms:.1f}"
    ttfb = "-" if ttfb_ms is None or ttfb_ms <= 0 else f"{ttfb_ms}ms"
    uid_prefix = (uid or "-")[:8]
    latency = f"{latency_ms}ms" if latency_ms is not None else "-"
    row = (f"| #{seq:03d} | {now} | {model:11s} | {mode:6s} | {status:3d} | uid={uid_prefix} "
           f"| TTFB={ttfb:>5} | tok={tok_field:>5} | {tps:>6}t/s | total={latency:>7} | {error_kind}")
    limit = getattr(settings, "LOG_PREVIEW", 48)
    if limit > 0:
        row += f' | in="{_preview(prompt_text, limit)}" | out="{_preview(output_text, limit)}"'
    print(row, flush=True)
    return seq


# ---------------------------------------------------------------------------
# 上游错误分类 + 账号状态机（ pool/upstream）：
#   - 网络层错误不累计 errCount
#   - 404 短冷却不累计 errCount（防雪崩）
#   - HTTP 5xx 累计 errCount，阈值 5 触发 10m 冷却
#   - 余额不足 / session 死亡 / 429 分别处理
# ---------------------------------------------------------------------------

_HARD_CREDIT_MARKERS = [
    "insufficient credit", "no credit", "credit exhausted", "out of credit",
    "quota exceeded", "quota exhaust", "payment required", "credit not enough",
    "not enough credit",
    "积分不足", "额度不足", "余额不足", "积分用完", "额度用尽", "没有积分",
]
_SESSION_DEAD_MARKERS = ["Offline user session not found", "12153", "session not found", "invalid session"]


def _classify_error(status: int, body: str) -> str:
    """按 HTTP 状态码 + body 关键词返回错误分类（字符串形式，与 UsageLog.error_kind 对齐）。"""
    if status == 402 or status == 412:
        return "hard_credit"
    body_l = (body or "").lower()
    for m in _HARD_CREDIT_MARKERS:
        if m.lower() in body_l or m in body:
            return "hard_credit"
    for m in _SESSION_DEAD_MARKERS:
        if m in body:
            return "session_dead"
    if status == 429:
        return "soft_rate"
    if status == 404:
        return "not_found"
    if status >= 500:
        return "server"
    if status >= 400:
        return "client"
    return "transport"  # 网络层/无响应状态


def _next_day_4am(now: datetime) -> datetime:
    """返回 now 所属日期的次日 04:00（本地时区）。"""
    return now.replace(hour=4, minute=0, second=0, microsecond=0) + timedelta(days=1)


def _apply_account_policy(db: Session, acc: Account, kind: str, status: int, msg: str) -> None:
    """根据错误分类更新账号状态：冷却/禁用/错误计数。"""
    now = datetime.utcnow()
    if kind == "hard_credit":
        acc.cool_until = _next_day_4am(now)
        acc.cool_kind = "hard_credit"
        acc.err_count = 0
        acc.last_err_at = now
        acc.last_err_msg = (msg or "余额不足")[:255]
    elif kind == "soft_rate":
        acc.cool_until = now + timedelta(seconds=60)
        acc.cool_kind = "soft_rate"
        acc.err_count = 0
        acc.last_err_at = now
        acc.last_err_msg = (msg or "429 rate limit")[:255]
    elif kind == "session_dead":
        acc.status = "disabled"
        acc.cool_kind = "session_dead"
        acc.last_err_at = now
        acc.last_err_msg = (msg or "session dead")[:255]
    elif kind == "not_found":
        # 404 短冷却不累计 errCount（防雪崩）
        acc.cool_until = now + timedelta(seconds=60)
        acc.cool_kind = "not_found"
        acc.last_err_at = now
        acc.last_err_msg = (msg or "upstream 404")[:255]
    elif kind == "server" or status >= 500:
        # HTTP 5xx 累计 errCount；阈值 5 触发 10m 冷却
        acc.err_count = (acc.err_count or 0) + 1
        acc.last_err_at = now
        acc.last_err_msg = (msg or f"upstream {status}")[:255]
        if acc.err_count >= 5:
            acc.cool_until = now + timedelta(minutes=10)
            acc.cool_kind = "error_threshold"
            acc.err_count = 0
    elif kind == "transport":
        # 网络层抖动不累计 errCount，只记录时间
        acc.last_err_at = now
        acc.last_err_msg = (msg or "transport error")[:255]
    else:
        # 其他 4xx 只换号，不累计 errCount
        acc.last_err_at = now
        acc.last_err_msg = (msg or f"upstream {status}")[:255]
    try:
        db.commit()
    except Exception:
        db.rollback()


def _account_session_safe(db: Session, acc: Account) -> backend.AccountSession | None:
    """创建 AccountSession 并调用 get_headers()（可能触发 token 刷新）。

    若刷新失败（session 死亡等），按策略禁用/冷却该账号并返回 None。
    """
    sess = backend.AccountSession(acc.auth_json)
    try:
        sess.get_headers()  # 内部会触发 token 刷新并写临时文件
        return sess
    except Exception as e:
        msg = str(e)
        kind = _classify_error(0, msg)
        if kind == "transport":
            kind = "session_dead"  # token 刷新失败通常等于 session 失效
        _apply_account_policy(db, acc, kind, 0, msg)
        try:
            sess.close()
        except Exception:
            pass
        return None


_CREDIT_RE = re.compile(r"x\s*([0-9]+(?:\.[0-9]+)?)")


def _select_account(db: Session, exclude_ids: set | None = None,
                    min_balance: int = 1, mark_picked: bool = True) -> Account | None:
    """从健康账号池中挑选一个账号。

    健康条件：active、有余额、不在冷却期、不在 exclude_ids 中。
    防撞号：优先跳过 last_picked_at 距今 < 100ms 的账号；若全部刚被用过则兜底。
    挑选策略默认按 balance_remain 降序（也可切 LRU）。
    """
    now = datetime.utcnow()
    q = db.query(Account).filter(Account.status == "active")
    if min_balance > 0:
        q = q.filter(Account.balance_remain > 0)
    # 排除已尝试或已冷却账号
    if exclude_ids:
        q = q.filter(~Account.id.in_(exclude_ids))
    q = q.filter(or_(Account.cool_until.is_(None), Account.cool_until <= now))

    # 防撞号窗口：100ms 内不重复选中同一账号
    anti = now - timedelta(milliseconds=100)
    q_anti = q.filter(or_(Account.last_picked_at.is_(None), Account.last_picked_at <= anti))

    if settings.ACCOUNT_SELECT == "lru":
        q_anti = q_anti.order_by(Account.last_used_at.asc())
        q = q.order_by(Account.last_used_at.asc())
    else:
        q_anti = q_anti.order_by(Account.balance_remain.desc())
        q = q.order_by(Account.balance_remain.desc())

    acc = q_anti.first()
    if not acc:
        acc = q.first()
    if acc and mark_picked:
        acc.last_picked_at = now
        db.commit()
    return acc


def _no_account_reason(db: Session) -> str:
    """当 _select_account 返回 None 时，给出人能看懂的真实原因。

    原文案固定写「全部禁用或额度耗尽」，但最常见的其实是「冷却中」——
    账号被上游限流（soft_rate）后 60 秒内不可选，此时其它模型也会被一起挡掉，
    却被告知「额度耗尽」，排查时极易误判。
    """
    now = datetime.utcnow()
    total = db.query(Account).count()
    if total == 0:
        return "账号池为空，请先在「账号管理」添加账号"
    if db.query(Account).filter(Account.status == "active").count() == 0:
        return "全部账号已禁用"
    if db.query(Account).filter(Account.status == "active",
                                Account.balance_remain > 0).count() == 0:
        return "全部账号额度耗尽"
    cooling = db.query(Account).filter(
        Account.status == "active", Account.balance_remain > 0,
        Account.cool_until.isnot(None), Account.cool_until > now,
    ).all()
    if cooling:
        parts = []
        for a in cooling[:3]:
            left = int((a.cool_until - now).total_seconds())
            parts.append(f"{a.name}:{a.cool_kind or 'cool'} 剩余{left}s")
        return "账号冷却中（" + "，".join(parts) + "），冷却结束后自动恢复"
    return "账号暂不可用（防撞号窗口内刚被占用）"

def _readable_upstream(text: str) -> str:
    """把上游错误原文转成可直接展示的中文；解析失败则原样返回（截断）。

    上游业务错误多为 JSON（含 code/msg 字段），若直接把整段 JSON 甩给客户端，
    会被拆成多块、且看不清真实原因。这里优先提取 msg/message 字段。
    """
    if not text:
        return ""
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            for k in ("msg", "message", "error_msg", "errmsg", "errorMessage"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            if isinstance(obj.get("error"), dict):
                ev = obj["error"].get("message")
                if isinstance(ev, str) and ev.strip():
                    return ev.strip()
    except Exception:
        pass
    return s[:300]

    return s[:300]


async def _try_open_upstream(db, order, body, request, url, key, payload, use_case, requested_model):
    """在流出 SSE 头之前，先尝试打开上游连接并判定本次请求能否成功。

    必要性：Starlette 的 StreamingResponse 一旦开始迭代生成器，HTTP 响应头
    （status=200）便已发出，之后无论生成器内发生什么都无法再把状态码改成 503。
    因此把「账号级重试 + 判定上游是否可用」整体前移到 StreamingResponse 构造之前。

    - 若所有账号/候选模型均失败（含上游 6004 限流、账号冷却等），在此写完失败日志，
      并返回 ("fail", err_kind, err_msg, status)。
    - 只有当某个候选拿到 status<400 的连接时，才返回 ("ok", client, r, model, acc, sess)，
      交由调用方以 HTTP 200 流式透传（连接复用，成功路径不额外发请求）。
    """
    client = httpx.AsyncClient(timeout=300, limits=backend.HTTP_LIMITS)
    tried_ids: set = set()
    last_err_kind = ""
    last_err_msg = ""
    last_status = 0
    request_start = time.perf_counter()
    try:
        for m in order:
            body["model"] = m
            for _attempt in range(3):
                acc = _select_account(db, exclude_ids=tried_ids, min_balance=1)
                if not acc:
                    break
                tried_ids.add(acc.id)
                sess = _account_session_safe(db, acc)
                if sess is None:
                    continue
                headers = sess.get_headers(extra=_upstream_extra_headers(request))
                try:
                    r = await client.send(
                        client.build_request("POST", url, headers=headers, json=body),
                        stream=True)
                    if r.status_code >= 400:
                        detail = await r.aread()
                        text = (detail[:500].decode(errors="ignore")
                                if isinstance(detail, bytes) else str(detail)[:500])
                        kind = _classify_error(r.status_code, text)
                        _apply_account_policy(db, acc, kind, r.status_code, text)
                        if kind in ("hard_credit", "session_dead", "soft_rate",
                                    "not_found", "server"):
                            last_err_kind, last_err_msg, last_status = kind, text, r.status_code
                            await r.aclose()
                            sess.close()
                            continue
                        # 不可重试的客户端错误：原样返回上游真实状态码
                        await r.aclose()
                        sess.close()
                        await client.aclose()
                        return ("fail", kind, text, r.status_code)
                    # 成功建立连接：保持打开，交由调用方透传
                    acc.last_used_at = datetime.utcnow()
                    db.commit()
                    return ("ok", client, r, m, acc, sess)
                except Exception as e:
                    kind = _classify_error(0, str(e))
                    _apply_account_policy(db, acc, kind, 0, str(e))
                    if kind == "transport":
                        last_err_kind, last_err_msg = kind, str(e)
                        sess.close()
                        continue
                    last_err_kind, last_err_msg = kind, str(e)
                    sess.close()
                    continue
        # 全部账号/模型均失败：把上游真实报错（如「使用量已超出频率限制…」）落库，
        # 否则失败请求在日志里只有提问、没有回复，排查限流时看不到任何线索。
        latency_ms = int((time.perf_counter() - request_start) * 1000)
        err_kind = last_err_kind or "no_account"
        seq = _log_chat_row(None, latency_ms, requested_model, "stream", "-",
                            (last_status or 503), None, error_kind=err_kind,
                            prompt_text=_extract_input(payload))
        _record_usage(key.id, 0, requested_model, 0.0, None,
                      client_ip=_client_ip(request), use_case=use_case, seq=seq,
                      latency_ms=latency_ms, error_kind=err_kind,
                      req_preview=_store_preview(_extract_input(payload)),
                      resp_preview=_store_preview(last_err_msg),
                      full_payload=payload, full_response=last_err_msg or "")
        await client.aclose()
        return ("fail", err_kind, last_err_msg, last_status)
    except Exception:
        await client.aclose()
        raise

def _pick_best_model(db: Session, requested_model: str) -> str | None:
    """根据请求模型和可用配置，选出最优实际使用的模型 ID。

    策略：
      - 用户指定了具体模型 → 校验白名单后直接用（或返回 None 表示被拒）
      - 用户传 "auto" 或空 → 优先选免费模型（credit_multiplier=0），没有免费的才选付费的
      - 未配置任何模型规则时放行全部（向后兼容），返回原始 model
    """
    from admin.routers.models import _is_model_allowed, _get_free_models, _get_enabled_models
    from admin.models import ModelConfig

    # 检查是否有任何配置记录（无配置=向后兼容，放行全部）
    has_any_config = db.query(ModelConfig).first() is not None

    # 具体模型：有配置时校验白名单，无配置直接放行
    if requested_model and requested_model != "auto":
        if not has_any_config:
            return requested_model  # 无配置，放行
        if _is_model_allowed(db, requested_model):
            return requested_model
        return None  # 被白名单拒绝

    # auto 模式：有配置时免费优先，无配置也从后端取模型列表自选（绝不透传 auto）
    if not has_any_config:
        # 无本地配置时：尝试从后端拉一次模型列表来选免费模型
        try:
            acc_tmp = _select_account(db)
            if acc_tmp:
                with backend.AccountSession(acc_tmp.auth_json) as sess:
                    raw_models = sess.fetch_models()
                # 选第一个 credits 为 0 或含 "free"/"x0.00" 的模型
                for rm in raw_models:
                    mid = rm.get("id", "")
                    if mid and mid.lower() != "auto":
                        cred = str(rm.get("credits") or "")
                        if not cred or "x0.00" in cred or "free" in cred.lower():
                            return mid
                # 没有免费模型就返回第一个非 auto
                for rm in raw_models:
                    mid = rm.get("id", "")
                    if mid and mid.lower() != "auto":
                        return mid
        except Exception:
            pass
        return "deepseek-v4-flash"  # 兜底：无配置且后端不可达时用默认模型

    free_models = _get_free_models(db)
    if free_models:
        return list(free_models)[0]  # 取第一个免费模型

    # 无免费模型：取任意一个启用的
    enabled = _get_enabled_models(db)
    if enabled:
        return list(enabled)[0]

    return None  # 有配置但全禁用


def _candidate_models(db: Session, tried: set) -> list:
    """按 免费→付费 顺序返回可用模型候选（排除已尝试的），用于 429/5xx 自动切换。"""
    from admin.routers.models import _get_enabled_models, _get_free_models

    free = _get_free_models(db) - tried
    paid = (_get_enabled_models(db) - free) - tried
    return list(free) + list(paid)


def _parse_usage(sse_text: str) -> dict:
    """从 chat SSE 文本里找最后一个带 usage 的事件，解析 credits 与 token 明细。

    返回 {"credits", "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"}。
    上游通常只在最后一个事件回传 usage（配合 stream_options.include_usage=True）。

    积分字段优先级：usage.credits > usage.credit > usage.cost，均缺失时返回 credits=None，
    由调用方按模型倍率估算（倍率单位为「每千 token」）。
    """
    credits = None
    prompt_tokens = completion_tokens = total_tokens = cached_tokens = None
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        usage = obj.get("usage")
        if not isinstance(usage, dict):
            continue
        # 桌面端源码读取 usage.credit / usage.credits；旧协议有 usage.cost，一并兼容。
        cred = usage.get("credits")
        if cred is None:
            cred = usage.get("credit")
        if cred is None and isinstance(usage.get("cost"), (int, float)):
            cred = usage.get("cost")
        if cred is not None:
            cred_str = str(cred).strip()
            # 兼容 "x 100" / "x100" 以及纯数字 "100" / "100.5"
            m = _CREDIT_RE.search(cred_str)
            if m:
                credits = float(m.group(1))
            else:
                try:
                    credits = float(cred_str)
                except ValueError:
                    pass
            _logger.debug("parse_usage credit raw=%r parsed=%s", cred, credits)
        if usage.get("prompt_tokens") is not None:
            prompt_tokens = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            completion_tokens = usage["completion_tokens"]
        if usage.get("total_tokens") is not None:
            total_tokens = usage["total_tokens"]
        # 缓存命中 token：OpenAI 标准在 prompt_tokens_details.cached_tokens
        cached = None
        ptd = usage.get("prompt_tokens_details")
        if isinstance(ptd, dict):
            cached = ptd.get("cached_tokens")
        if cached is None and usage.get("cached_tokens") is not None:
            cached = usage["cached_tokens"]
        if cached is not None:
            cached_tokens = cached
    return {
        "credits": credits,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }


def _estimate_credits(sse_text: str, model: str) -> float:
    """兼容旧调用：仅返回 credits 估算（token 明细用 _parse_usage）。"""
    u = _parse_usage(sse_text)
    if u["credits"] is not None:
        return u["credits"]
    toks = u["total_tokens"] or u["completion_tokens"]
    if toks:
        return float(toks) * settings.COST_PER_TOKEN / 1000.0
    return 0.0


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    db: Session = Depends(get_db),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
):
    api_key = x_api_key
    if not api_key and authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:].strip()
    if not api_key:
        return JSONResponse(status_code=401, content={"error": {"message": "缺少 API Key", "type": "auth_error"}})

    key = get_key_row(db, api_key)
    if not key:
        return JSONResponse(status_code=401, content={"error": {"message": "无效 API Key", "type": "auth_error"}})
    try:
        check_quota(key)
    except Exception as e:
        return JSONResponse(status_code=e.status_code, content=e.detail)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "bad json", "type": "invalid_request"}})

    acc = _select_account(db)
    if not acc:
        # 无可用账号此前不写日志，后台完全看不到这类失败；这里补记录并写明真实原因
        reason = _no_account_reason(db)
        _record_usage(key.id, 0, str(payload.get("model") or "auto"), 0.0, None,
                      client_ip=_client_ip(request), use_case="chat-completion",
                      error_kind="no_account", http_status=503,
                      req_preview=_store_preview(_extract_input(payload)),
                      resp_preview=_store_preview(reason), full_payload=payload, full_response=reason)
        return JSONResponse(status_code=503,
                            content={"error": {"message": f"无可用账号（{reason}）", "type": "no_account"}})

    model = payload.get("model", "auto")

    # 模型白名单检查 + 免费优先选择
    resolved_model = _pick_best_model(db, model)
    if resolved_model is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"模型 '{model}' 不存在或已被禁用", "type": "model_not_found"}},
        )

    # 候选模型顺序：auto 模式按 免费→付费 排列，支持上游 429/5xx 自动切换下一个
    if model in ("auto", ""):
        order = [resolved_model] + _candidate_models(db, {resolved_model})
        order = order[:8]  # 最多尝试 8 个，避免全局限流时反复重试
    else:
        order = [resolved_model]  # 具体模型：不静默切换，失败即报错

    body = dict(payload)
    body["stream"] = True
    # 始终要求上游回传 usage（token 与缓存命中），保证调用方一定能拿到用量自行记录
    opts = dict(body.get("stream_options") or {})
    opts["include_usage"] = True
    body["stream_options"] = opts

    url = f"{settings.BACKEND}/v2/chat/completions"

    # 在流出 SSE 头之前先尝试打开上游连接：若所有账号/候选模型均不可用
    # （含上游 6004 限流、账号冷却等），直接返回真实 HTTP 503 + 上游中文原话；
    # 只有拿到 status<400 的连接才进入流式透传（否则一旦发了 HTTP 200 就无法改 503）。
    opened = await _try_open_upstream(
        db, order, body, request, url, key, payload, "chat-completion",
        str(payload.get("model") or "auto"))
    if opened[0] == "fail":
        kind, msg, status = opened[1], opened[2], opened[3]
        readable = _readable_upstream(msg)
        fail_msg = (f"{readable}（错误类型：{kind}）" if msg else "无可用账号或模型")
        # 这里此前直接返回 503 而不写日志，导致限流类报错在后台完全看不到；现已补记
        _record_usage(key.id, 0, str(payload.get("model") or "auto"), 0.0, None,
                      client_ip=_client_ip(request), use_case="chat-completion",
                      error_kind=kind, http_status=503,
                      req_preview=_store_preview(_extract_input(payload)),
                      resp_preview=_store_preview(readable), full_payload=payload, full_response=msg or "")
        return JSONResponse(status_code=503,
                            content={"error": {"message": fail_msg, "type": "no_model_available"}})

    client, upstream_r, final_model, acc_i, sess_i = opened[1], opened[2], opened[3], opened[4], opened[5]

    async def _stream():
        db2 = SessionLocal()
        try:
            request_start = time.perf_counter()
            ttfb_at = None
            collected = []
            async for chunk in upstream_r.aiter_text():
                if ttfb_at is None:
                    ttfb_at = time.perf_counter()
                collected.append(chunk)
                yield chunk
            text = "".join(collected)
            usage = _parse_usage(text)
            status_out = 200
            latency_ms = int((time.perf_counter() - request_start) * 1000)
            ttfb_ms = int((ttfb_at - request_start) * 1000) if ttfb_at else None
            seq = _log_chat_row(ttfb_ms, latency_ms, final_model, "stream", acc_i.uid or "-",
                                status_out, usage["total_tokens"] or usage["completion_tokens"],
                                error_kind="success", prompt_text=_extract_input(payload),
                                output_text=_extract_output(text))
            updated = sess_i.updated_json()
            sess_i.close()
            _record_usage(key.id, acc_i.id, final_model, usage["credits"], updated,
                          client_ip=_client_ip(request), use_case="chat-completion",
                          prompt_tokens=usage["prompt_tokens"],
                          completion_tokens=usage["completion_tokens"],
                          total_tokens=usage["total_tokens"],
                          cached_tokens=usage["cached_tokens"],
                          seq=seq, ttfb_ms=ttfb_ms, latency_ms=latency_ms,
                          error_kind="success", http_status=200,
                          req_preview=_store_preview(_extract_input(payload)),
                          resp_preview=_store_preview(_extract_output(text)),
                          full_payload=payload, full_response=text)
        except Exception as e:
            latency_ms = int((time.perf_counter() - request_start) * 1000)
            ttfb_ms = int((ttfb_at - request_start) * 1000) if ttfb_at else None
            err_text = str(e)
            seq = _log_chat_row(ttfb_ms, latency_ms, final_model, "stream", acc_i.uid or "-",
                                200, None, error_kind="transport", prompt_text=_extract_input(payload))
            _record_usage(key.id, acc_i.id, final_model, 0.0, None,
                          client_ip=_client_ip(request), use_case="chat-completion", seq=seq,
                          latency_ms=latency_ms, error_kind="transport", http_status=200,
                          req_preview=_store_preview(_extract_input(payload)),
                          resp_preview=_store_preview(err_text),
                          full_payload=payload, full_response=err_text)
        finally:
            try:
                await upstream_r.aclose()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass
            db2.close()

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/v1/responses")
async def responses_proxy(
    request: Request,
    db: Session = Depends(get_db),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
):
    """OpenAI Responses API 兼容端点（带 API Key 配额 / 用量记账 / 账号级熔断重试）。

    与 /v1/chat/completions 同一套托管逻辑：校验 Key → 配额 → 候选模型 →
    自动挑选健康账号；遇到余额不足 / session 死亡 / 限流 / 5xx 时自动换号，
    绝不把上游中断感传递给客户端。
    """
    if not _RESPONSES_AVAILABLE:
        return JSONResponse(status_code=501, content={"error": {"message": "Responses 适配器未加载", "type": "not_supported"}})

    api_key = x_api_key
    if not api_key and authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:].strip()
    if not api_key:
        return JSONResponse(status_code=401, content={"error": {"message": "缺少 API Key", "type": "auth_error"}})
    key = get_key_row(db, api_key)
    if not key:
        return JSONResponse(status_code=401, content={"error": {"message": "无效 API Key", "type": "auth_error"}})
    try:
        check_quota(key)
    except Exception as e:
        return JSONResponse(status_code=e.status_code, content=e.detail)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "bad json", "type": "invalid_request"}})

    try:
        chat_body = responses_request_to_chat(payload)
    except Exception as e:
        return JSONResponse(status_code=400,
                            content={"error": {"message": f"请求转换失败：{e}", "type": "invalid_request"}})

    chat_body, _stats = project_responses_chat_body(chat_body)
    chat_body.setdefault("model", "auto")
    chat_body["stream"] = True
    opts = dict(chat_body.get("stream_options") or {})
    opts["include_usage"] = True
    chat_body["stream_options"] = opts

    requested = payload.get("model", "auto")
    resolved = _pick_best_model(db, requested)
    if resolved is None:
        return JSONResponse(status_code=400,
                            content={"error": {"message": f"模型 '{requested}' 不存在或已被禁用", "type": "model_not_found"}})

    order = [resolved]
    if requested in ("auto", ""):
        order = [resolved] + _candidate_models(db, {resolved})
        order = order[:8]

    client_wants_stream = bool(payload.get("stream", True))
    model_name = payload.get("model", "auto")
    url = f"{settings.BACKEND}/v2/chat/completions"

    if not client_wants_stream:
        # 非流式：内部重试，成功后聚合为单一 Response 对象
        db2 = SessionLocal()
        try:
            for m in order:
                body = dict(chat_body)
                body["model"] = m
                tried_ids: set = set()
                for _ in range(3):
                    acc_i = _select_account(db2, exclude_ids=tried_ids, min_balance=1)
                    if not acc_i:
                        break
                    tried_ids.add(acc_i.id)
                    sess_i = _account_session_safe(db2, acc_i)
                    if sess_i is None:
                        continue
                    headers_i = sess_i.get_headers(extra=_upstream_extra_headers(request))
                    try:
                        async with httpx.AsyncClient(timeout=300, limits=backend.HTTP_LIMITS) as client:
                            r = await client.post(url, headers=headers_i, json=body)
                            if r.status_code >= 400:
                                text = r.text[:500]
                                kind = _classify_error(r.status_code, text)
                                _apply_account_policy(db2, acc_i, kind, r.status_code, text)
                                if kind in ("hard_credit", "session_dead", "soft_rate", "not_found", "server"):
                                    sess_i.close()
                                    continue
                                sess_i.close()
                                return JSONResponse(status_code=r.status_code,
                                                    content={"error": {"message": text, "code": r.status_code}})
                            converter = ResponsesStreamConverter(model=model_name)
                            for line in r.text.splitlines():
                                if not line.strip():
                                    continue
                                converter.feed_line(line)
                            converter.finish()
                            obj = converter.get_nonstream_response()
                            cost_info = _parse_usage(r.text)
                            acc_i.last_used_at = datetime.utcnow()
                            db2.commit()
                            updated = sess_i.updated_json()
                            total_toks = cost_info["total_tokens"] or cost_info["completion_tokens"]
                            seq = _log_chat_row(None, None, m, "resp", acc_i.uid or "-", 200, total_toks,
                                                error_kind="success", prompt_text=_extract_input(payload),
                                                output_text=_extract_output(r.text))
                            sess_i.close()
                            _record_usage(key.id, acc_i.id, m, cost_info["credits"], updated,
                                          client_ip=_client_ip(request), use_case="responses",
                                          prompt_tokens=cost_info["prompt_tokens"],
                                          completion_tokens=cost_info["completion_tokens"],
                                          total_tokens=cost_info["total_tokens"],
                                          cached_tokens=cost_info["cached_tokens"],
                                          seq=seq, error_kind="success", http_status=200,
                                          req_preview=_store_preview(_extract_input(payload)),
                                          resp_preview=_store_preview(_extract_output(r.text)),
                                          full_payload=payload, full_response=r.text)
                            return JSONResponse(content=obj)
                    except Exception as e:
                        kind = _classify_error(0, str(e))
                        _apply_account_policy(db2, acc_i, kind, 0, str(e))
                        sess_i.close()
                        continue
            reason = _no_account_reason(db2)
            seq = _log_chat_row(None, None, resolved, "resp", "-", 503, None,
                                error_kind="no_account", prompt_text=_extract_input(payload))
            _record_usage(key.id, 0, resolved, 0.0, None,
                          client_ip=_client_ip(request), use_case="responses", seq=seq,
                          error_kind="no_account", http_status=503,
                          req_preview=_store_preview(_extract_input(payload)),
                          full_payload=payload)
            return JSONResponse(status_code=503,
                                content={"error": {"message": f"无可用账号（{reason}）", "type": "no_account"}})

        finally:
            db2.close()

    opened = await _try_open_upstream(
        db, order, chat_body, request, url, key, payload, "responses",
        str(payload.get("model") or "auto"))
    if opened[0] == "fail":
        kind, msg, status = opened[1], opened[2], opened[3]
        readable = _readable_upstream(msg)
        fail_msg = (f"{readable}（错误类型：{kind}）" if msg else "无可用账号或模型")
        # 这里此前直接返回 503 而不写日志，导致限流类报错在后台完全看不到；现已补记
        _record_usage(key.id, 0, str(payload.get("model") or "auto"), 0.0, None,
                      client_ip=_client_ip(request), use_case="responses",
                      error_kind=kind, http_status=503,
                      req_preview=_store_preview(_extract_input(payload)),
                      resp_preview=_store_preview(readable), full_payload=payload, full_response=msg or "")
        return JSONResponse(status_code=503,
                            content={"error": {"message": fail_msg, "type": "no_model_available"}})

    client, upstream_r, final_model, acc_i, sess_i = opened[1], opened[2], opened[3], opened[4], opened[5]

    async def _stream():
        db2 = SessionLocal()
        try:
            request_start = time.perf_counter()
            ttfb_at = None
            raw_lines = []
            converter = ResponsesStreamConverter(model=model_name)
            async for line in upstream_r.aiter_lines():
                if not line.strip():
                    continue
                if ttfb_at is None:
                    ttfb_at = time.perf_counter()
                events = converter.feed_line(line)
                if events:
                    yield events
                raw_lines.append(line)
            finish = converter.finish()
            if finish:
                yield finish
            text = "\n".join(raw_lines)
            usage = _parse_usage(text)
            latency_ms = int((time.perf_counter() - request_start) * 1000)
            ttfb_ms = int((ttfb_at - request_start) * 1000) if ttfb_at else None
            seq = _log_chat_row(ttfb_ms, latency_ms, final_model, "resp", acc_i.uid or "-", 200,
                                usage["total_tokens"] or usage["completion_tokens"],
                                error_kind="success", prompt_text=_extract_input(payload),
                                output_text=_extract_output(text))
            updated = sess_i.updated_json()
            sess_i.close()
            _record_usage(key.id, acc_i.id, final_model, usage["credits"], updated,
                          client_ip=_client_ip(request), use_case="responses",
                          prompt_tokens=usage["prompt_tokens"],
                          completion_tokens=usage["completion_tokens"],
                          total_tokens=usage["total_tokens"],
                          cached_tokens=usage["cached_tokens"],
                          seq=seq, ttfb_ms=ttfb_ms, latency_ms=latency_ms,
                          error_kind="success", http_status=200,
                          req_preview=_store_preview(_extract_input(payload)),
                          resp_preview=_store_preview(_extract_output(text)),
                          full_payload=payload, full_response=text)
        except Exception as e:
            latency_ms = int((time.perf_counter() - request_start) * 1000)
            ttfb_ms = int((ttfb_at - request_start) * 1000) if ttfb_at else None
            err_text = str(e)
            seq = _log_chat_row(ttfb_ms, latency_ms, final_model, "resp", acc_i.uid or "-", 200,
                                None, error_kind="transport", prompt_text=_extract_input(payload))
            _record_usage(key.id, acc_i.id, final_model, 0.0, None,
                          client_ip=_client_ip(request), use_case="responses", seq=seq,
                          latency_ms=latency_ms, error_kind="transport", http_status=200,
                          req_preview=_store_preview(_extract_input(payload)),
                          resp_preview=_store_preview(err_text),
                          full_payload=payload, full_response=err_text)
        finally:
            try:
                await upstream_r.aclose()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass
            db2.close()

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/v1/models")
async def models(
    db: Session = Depends(get_db),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
):
    api_key = x_api_key
    if not api_key and authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:].strip()
    if not api_key:
        return JSONResponse(status_code=401, content={"error": {"message": "缺少 API Key", "type": "auth_error"}})
    key = get_key_row(db, api_key)
    if not key:
        return JSONResponse(status_code=401, content={"error": {"message": "无效 API Key", "type": "auth_error"}})
    try:
        check_quota(key)
    except Exception as e:
        return JSONResponse(status_code=e.status_code, content=e.detail)

    acc = _select_account(db)
    if not acc:
        return JSONResponse(status_code=503,
                            content={"error": {"message": f"无可用账号（{_no_account_reason(db)}）",
                                               "type": "no_account"}})
    try:
        with backend.AccountSession(acc.auth_json) as sess:
            models_raw = sess.fetch_models()
            acc.auth_json = sess.updated_json()
        acc.last_used_at = datetime.utcnow()
        db.commit()
        data = [{
            "id": m.get("id"),
            "object": "model",
            "owned_by": "codebuddy",
            "name": m.get("name") or m.get("id"),
            "credit_multiplier": backend.CredentialManager._parse_credit_multiplier(m.get("credits"))
            if hasattr(backend.CredentialManager, "_parse_credit_multiplier") else None,
        } for m in models_raw if m.get("id") and _is_model_allowed(db, m.get("id")) and m.get("id","").lower() != "auto"]
        return {"object": "list", "data": data, "source": "backend"}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": {"message": f"获取模型失败：{e}", "type": "upstream"}})
