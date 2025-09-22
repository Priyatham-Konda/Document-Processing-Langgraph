import os
from celery import Celery
from celery.schedules import crontab


def make_celery() -> Celery:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    app = Celery(
        "document_processor",
        broker=redis_url,
        backend=redis_url,
        include=["tasks"],
    )

    # Sensible defaults
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone=os.getenv("TZ", "UTC"),
        enable_utc=True,
        worker_concurrency=int(os.getenv("CELERY_CONCURRENCY", "4")),
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        broker_heartbeat=0,
    )

    # Beat schedule (cleanup). Default: every 2 hours at minute 0
    # If you need a different cadence, set CLEANUP_CRON_MINUTE and CLEANUP_CRON_HOUR (supports crontab wildcards)
    minute = os.getenv("CLEANUP_CRON_MINUTE", "0")
    hour = os.getenv("CLEANUP_CRON_HOUR", "*/2")
    app.conf.beat_schedule = {
        "cleanup-old-artifacts": {
            "task": "tasks.cleanup_old_artifacts",
            "schedule": crontab(minute=minute, hour=hour),
        }
    }
    return app


celery_app = make_celery()


