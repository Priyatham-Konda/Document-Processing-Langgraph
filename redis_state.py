import os
import json
from datetime import timedelta
from typing import Any, Dict, Optional

import redis


def _redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.from_url(url, decode_responses=True)


DEFAULT_TTL_SECONDS = int(os.getenv("JOB_STATE_TTL_SECONDS", "86400"))  # 24h


def _job_key(document_id: str) -> str:
    return f"job:{document_id}"


def set_job_state(document_id: str, state: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
    r = _redis_client()
    def _json_default(obj):
        try:
            from dataclasses import is_dataclass, asdict
            if is_dataclass(obj):
                return asdict(obj)
        except Exception:
            pass
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except Exception:
                pass
        try:
            return str(obj)
        except Exception:
            return None
    payload = json.dumps(state, ensure_ascii=False, default=_json_default)
    key = _job_key(document_id)
    r.set(key, payload, ex=ttl_seconds or DEFAULT_TTL_SECONDS)


def get_job_state(document_id: str) -> Optional[Dict[str, Any]]:
    r = _redis_client()
    raw = r.get(_job_key(document_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def update_job_state(document_id: str, patch: Dict[str, Any], ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
    current = get_job_state(document_id) or {}
    current.update(patch)
    set_job_state(document_id, current, ttl_seconds)
    return current


def set_job_status(document_id: str, status: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {"status": status}
    if extra:
        payload.update(extra)
    return update_job_state(document_id, payload)


def delete_job_state(document_id: str) -> None:
    r = _redis_client()
    r.delete(_job_key(document_id))


def list_jobs(prefix: str = "job:") -> Dict[str, Dict[str, Any]]:
    r = _redis_client()
    jobs: Dict[str, Dict[str, Any]] = {}
    for key in r.scan_iter(f"{prefix}*"):
        try:
            raw = r.get(key)
            data = json.loads(raw) if raw else None
            if not data:
                continue
            document_id = key.split(":", 1)[1]
            jobs[document_id] = data
        except Exception:
            continue
    return jobs


