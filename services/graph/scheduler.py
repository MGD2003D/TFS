"""
CommunityScheduler — фоновая периодическая пересборка сообществ.

Паттерн аналогичен ChatCleanupWorker (services/chat_cleanup.py):
  asyncio.create_task + stop_event.

Rebuild запускается при первом наступившем условии:
  - накопилось >= min_new_triplets новых триплетов с прошлой пересборки
  - прошло >= max_interval_hours часов с прошлой пересборки
"""

import asyncio
import time
from typing import TYPE_CHECKING, Optional
from .communities import CommunityManager


class CommunityScheduler:

    def __init__(
        self,
        manager: CommunityManager,
        check_interval_seconds: int = 600,
        min_new_triplets: int = 50,
        max_interval_hours: float = 6.0,
    ):
        self.manager = manager
        self.check_interval = check_interval_seconds
        self.min_new_triplets = min_new_triplets
        self.max_interval_seconds = max_interval_hours * 3600
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop())
        print(
            f"[COMMUNITY SCHEDULER] Started "
            f"(check every {self.check_interval}s, "
            f"rebuild on {self.min_new_triplets}+ new triplets "
            f"or {self.max_interval_seconds/3600:.1f}h elapsed)"
        )

    async def stop(self):
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("[COMMUNITY SCHEDULER] Stopped")

    async def _loop(self):
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.check_interval,
                )
                break  # stop_event был выставлен
            except asyncio.TimeoutError:
                pass  # timeout истёк — проверяем условия

            if await self._should_rebuild():
                await self._run_rebuild()

    async def _should_rebuild(self) -> bool:
        # Условие 1: давно не пересобирали
        elapsed = time.time() - self.manager._last_rebuild_at
        if self.manager._last_rebuild_at > 0 and elapsed >= self.max_interval_seconds:
            print(f"[COMMUNITY SCHEDULER] Time trigger ({elapsed/3600:.1f}h elapsed)")
            return True

        # Условие 2: накопилось достаточно новых триплетов
        new_triplets = await self.manager.triplets_since_last_rebuild()
        if new_triplets >= self.min_new_triplets:
            print(f"[COMMUNITY SCHEDULER] Triplet threshold ({new_triplets} new triplets)")
            return True

        return False

    async def _run_rebuild(self):
        print("[COMMUNITY SCHEDULER] Starting rebuild...")
        start = time.perf_counter()
        try:
            saved = await self.manager.rebuild()
            elapsed = time.perf_counter() - start
            print(f"[COMMUNITY SCHEDULER] Rebuild done in {elapsed:.1f}s ({saved} communities)")
        except Exception as e:
            print(f"[COMMUNITY SCHEDULER] Rebuild failed: {e}")
