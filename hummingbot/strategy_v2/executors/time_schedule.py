"""
Shared release clock for executors that spread a total across a duration.

TWAPExecutor.create_order_plan uses release_timestamps(start, number_of_orders, order_interval),
which is start + i * interval. An exit iceberg uses those offsets as its release clock
and closes the last window at the duration.

Range iceberg entry keeps the same offsets so each gap can include timing jitter.
It does not open the next entry slice when that offset is reached. The next entry
slice waits until the previous one is fully filled and one gap has passed since
that fill. The duration clock is separate and still starts at the first fill, or
at executor start when schedule_start is SUBMIT.
"""

import re
from decimal import Decimal
from typing import List, Optional, Sequence, Union

Number = Union[int, float, Decimal]
_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])?\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: Union[Number, str]) -> float:
    """
    Seconds from a number or a short string such as 24h, 30m, 45s, or 1d.
    A bare number is seconds.
    """
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    match = _DURATION.match(str(value).strip())
    if match is None:
        raise ValueError(f"duration ({value}) must be seconds or a value like 24h, 30m, 45s, 1d")
    seconds = float(match.group(1)) * _UNIT_SECONDS[(match.group(2) or "s").lower()]
    return seconds


def release_timestamps(start: float, count: int, interval: float) -> List[float]:
    """
    Release times used by the TWAP order plan: start + i * interval for i in 0..count-1.
    """
    return [start + i * interval for i in range(count)]


class SliceSchedule:
    """
    How much of a total is released at each offset, and how much of a missed window is dropped.

    CARRY keeps an unfilled window in the amount later windows may still place.
    SKIP drops the part of a window that was never placed because the market was outside the
    cap when that window closed. Size that is still working on the book is not dropped, so a
    resting order can still fill.
    """

    def __init__(
        self,
        amounts: Sequence[Decimal],
        offsets: Sequence[float],
        duration: Optional[float],
        skip_unplaced: bool,
    ):
        if len(amounts) != len(offsets) or not amounts:
            raise ValueError("A schedule needs one amount for every release time")
        self.amounts = [Decimal(str(amount)) for amount in amounts]
        self.offsets = [float(offset) for offset in offsets]
        self.duration = None if duration is None else float(duration)
        self.skip_unplaced = skip_unplaced
        self.skipped = Decimal("0")
        self._closed = 0
        self._assigned = Decimal("0")

    @property
    def count(self) -> int:
        return len(self.amounts)

    def opened(self, elapsed: float) -> Decimal:
        total = Decimal("0")
        for amount, offset in zip(self.amounts, self.offsets):
            if offset <= elapsed + 1e-9:
                total += amount
        return total

    def advance(self, elapsed: float, filled: Decimal, open_amount: Decimal, outside_cap: bool) -> None:
        """
        Close every window whose end is at or before elapsed.

        A resting order is placed size, so it can cover only as much budget as it actually
        occupies. It is applied to the earliest closed windows and does not shield later ones.
        SKIP drops the unplaced part of a window only when the market is still outside the
        cap as that window closes. An earlier outside tick does not block the rest of the window.
        """
        filled = Decimal(str(filled))
        open_amount = Decimal(str(open_amount))
        while self._closed < len(self.amounts):
            if self._closed + 1 < len(self.offsets):
                end = self.offsets[self._closed + 1]
            else:
                end = self.duration
            if end is None or elapsed + 1e-9 < end:
                break
            budget = self.amounts[self._closed]
            available = filled + open_amount - self._assigned
            if available < 0:
                available = Decimal("0")
            assigned = budget if available >= budget else available
            unplaced = budget - assigned
            if self.skip_unplaced and outside_cap and unplaced > 0:
                self.skipped += unplaced
            self._assigned += assigned
            self._closed += 1

    def export_state(self) -> dict:
        """
        Window accounting that has to survive a serialized executor.

        Decimals are strings so the dict can sit in custom_info and be stored as JSON.
        """
        return {
            "skipped": str(self.skipped),
            "closed": self._closed,
            "assigned": str(self._assigned),
        }

    def restore_state(self, state: dict) -> None:
        """Replace window accounting with a snapshot from export_state."""
        if not state:
            return
        self.skipped = Decimal(str(state.get("skipped", "0")))
        self._closed = int(state.get("closed", 0))
        assigned = state.get("assigned", "0")
        self._assigned = Decimal(str("0" if assigned is None else assigned))
