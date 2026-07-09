from apscheduler.schedulers.background import BackgroundScheduler
from state.sync import sync

_scheduler = BackgroundScheduler()

_scheduler.add_job(
    sync,
    trigger="interval", # kol fatra
    seconds=30,
    max_instances=1,
    coalesce=True,
)


def start_scheduler() -> None:
    """Start the background sync scheduler."""
    _scheduler.start()
    print("🕐 Scheduler started — sync() will run every 30 seconds.")


def stop_scheduler() -> None:
    """Shut down the background sync scheduler gracefully."""
    _scheduler.shutdown()
    print("🛑 Scheduler stopped.")