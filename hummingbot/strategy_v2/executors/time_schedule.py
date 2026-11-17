"""
Shared release clock for executors that spread a total across a duration.

TWAPExecutor.create_order_plan uses release_timestamps(start, number_of_orders, order_interval),
which is start + i * interval. The range iceberg schedule uses the same helper for its
nominal release times, then closes the last window at the duration so leftovers are cancelled.
"""

import re
from decimal import Decimal
from typing import List, Sequence, Union

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
        duration: float,
        skip_unplaced: bool,
    ):
        if len(amounts) != len(offsets) or not amounts:
            raise ValueError("A schedule needs one amount for every release time")
        self.amounts = [Decimal(str(amount)) for amount in amounts]
        self.offsets = [float(offset) for offset in offsets]
        self.duration = float(duration)
        self.skip_unplaced = skip_unplaced
        self.skipped = Decimal("0")
        self._closed = 0
        self._filled_checkpoint = Decimal("0")

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
        filled = Decimal(str(filled))
        open_amount = Decimal(str(open_amount))
        while self._closed < len(self.amounts):
            end = self.offsets[self._closed + 1] if self._closed + 1 < len(self.offsets) else self.duration
            if elapsed + 1e-9 < end:
                break
            budget = self.amounts[self._closed]
            filled_in_window = filled - self._filled_checkpoint
            if filled_in_window < 0:
                filled_in_window = Decimal("0")
            shortfall = budget - filled_in_window
            if shortfall < 0:
                shortfall = Decimal("0")
            unplaced = shortfall - open_amount
            if unplaced < 0:
                unplaced = Decimal("0")
            if self.skip_unplaced and outside_cap and unplaced > 0:
                self.skipped += unplaced
            self._filled_checkpoint = filled
            self._closed += 1
