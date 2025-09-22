import os
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta

from celery.schedules import crontab

from celery_app import celery_app
from redis_state import set_job_status, set_job_state, update_job_state, list_jobs
from doc_process_gemini_v2 import build_graph
from google import genai
from google.genai import types
from dataclasses import is_dataclass, asdict


@celery_app.task(name="tasks.process_document")
def process_document_task(document_id: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    set_job_status(document_id, "processing", {"filename": filename, "started_at": datetime.utcnow().isoformat()})

    try:
        workflow_app = build_graph()
        initial_state = {
            "file_bytes": file_bytes,
            "metadata": {"filename": filename},
            "document_id": document_id,
        }

        final_state = None
        for event in workflow_app.stream(initial_state):
            final_state = event

        if not final_state:
            set_job_status(document_id, "failed", {"error": "No final state returned"})
            return {"status": "failed"}

        last_node = list(final_state.keys())[0]
        final_results = final_state[last_node] or {}

        def _to_jsonable(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if is_dataclass(obj):
                return _to_jsonable(asdict(obj))
            if isinstance(obj, dict):
                return {str(k): _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_to_jsonable(v) for v in obj]
            if hasattr(obj, "isoformat"):
                try:
                    return obj.isoformat()
                except Exception:
                    pass
            try:
                return json.loads(json.dumps(obj))
            except Exception:
                return str(obj)

        final_results_serialized = _to_jsonable(final_results)

        # Persist results
        set_job_state(document_id, {
            "status": "completed",
            "results": final_results_serialized,
            "completed_at": datetime.utcnow().isoformat(),
        })
        return {"status": "completed", "document_id": document_id}

    except Exception as e:
        set_job_status(document_id, "failed", {"error": str(e)})
        return {"status": "failed", "error": str(e)}


@celery_app.task(name="tasks.cleanup_old_artifacts")
def cleanup_old_artifacts() -> Dict[str, Any]:
    now = datetime.utcnow()
    # Defaults: keep for 3 days
    keep_hours = int(os.getenv("ARTIFACT_KEEP_HOURS", "72"))
    cutoff = now - timedelta(hours=keep_hours)

    deleted: Dict[str, Any] = {"document_images": 0, "extraction_logs": 0, "review_data": 0, "learning_data": 0, "raw documents": 0}

    def _maybe_delete_dir(path: Path) -> int:
        # Hard safeguard: never delete prompts folder or anything within it
        try:
            if path.name == "prompts" or any(p.name == "prompts" for p in path.parents):
                return 0
        except Exception:
            pass
        if not path.exists():
            return 0
        count = 0
        for item in path.iterdir():
            try:
                # Skip if this item is the prompts dir
                if item.name == "prompts":
                    continue
                # Infer timestamp by folder mtime
                mtime = datetime.utcfromtimestamp(item.stat().st_mtime)
                if mtime < cutoff:
                    if item.is_dir():
                        for sub in item.rglob("*"):
                            try:
                                if sub.name == "prompts" or any(p.name == "prompts" for p in sub.parents):
                                    continue
                                if sub.is_file():
                                    sub.unlink(missing_ok=True) if hasattr(sub, "unlink") else sub.unlink()
                            except Exception:
                                pass
                        try:
                            item.rmdir()
                        except Exception:
                            pass
                    else:
                        item.unlink(missing_ok=True) if hasattr(item, "unlink") else item.unlink()
                    count += 1
            except Exception:
                continue
        return count

    for folder in ["document_images", "extraction_logs", "review_data", "learning_data", "raw documents"]:
        deleted[folder] = _maybe_delete_dir(Path(folder))

    return {"status": "ok", "deleted": deleted, "cutoff": cutoff.isoformat()}


@celery_app.task(
    name="tasks.ask_gemini_text_only",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=int(os.getenv("GEMINI_TEXT_MAX_RETRIES", "3")),
    soft_time_limit=int(os.getenv("GEMINI_TEXT_SOFT_TIME_LIMIT_SEC", "300")),
)
def ask_gemini_text_only_task(self, prompt: str, model: str = "gemini-2.5-pro") -> str:
    """Background task to call Gemini text-only API.

    Returns the text response; callers should use Celery AsyncResult to retrieve.
    """
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        parts = [types.Part.from_text(text=prompt)]
        contents = [types.Content(role="user", parts=parts)]
        config = types.GenerateContentConfig(response_modalities=["TEXT"]) 
        response = client.models.generate_content(model=model, contents=contents, config=config)
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        raise e


