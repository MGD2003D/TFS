import asyncio
import time
import app_state


class ChatCleanupWorker:
    def __init__(self, ttl_seconds: int = 3600, interval_seconds: int = 60):
        self.ttl_seconds = ttl_seconds
        self.interval_seconds = interval_seconds
        self._stop_event = asyncio.Event()
        self._task = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._cleanup()
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

    def _cleanup(self) -> None:
        now = time.monotonic()
        expired = [
            user_id
            for user_id, last_seen in list(app_state.chat_last_activity.items())
            if now - last_seen >= self.ttl_seconds
        ]

        for user_id in expired:
            app_state.delete_user_history(user_id)
