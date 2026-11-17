from decimal import Decimal
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import field_validator, model_validator

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase
from hummingbot.strategy_v2.executors.time_schedule import parse_duration
from hummingbot.strategy_v2.executors.validation import (
    require_at_least,
    require_directional_side,
    require_non_empty,
    require_non_negative,
    require_positive,
    require_trading_pair,
)


class SizeUnit(str, Enum):
    BASE = "BASE"
    QUOTE = "QUOTE"
    CONTRACTS = "CONTRACTS"


class LadderDistribution(str, Enum):
    LINEAR = "LINEAR"
    GEOMETRIC = "GEOMETRIC"
    WEIGHTED = "WEIGHTED"


class ExecutionMode(str, Enum):
    RANGE = "RANGE"
    UP_TO = "UP_TO"


class CapBehavior(str, Enum):
    """What to do when the market is through the cap (buy) or the floor (sell)."""

    REST_AT_CAP = "REST_AT_CAP"
    WAIT = "WAIT"


class CatchUpPolicy(str, Enum):
    """What happens to a scheduled slice that did not get placed because of the cap."""

    CARRY = "CARRY"
    SKIP = "SKIP"


class RangeIcebergExecutorConfig(ExecutorConfigBase):
    """
    Buy or sell a total size as a range ladder or as an up-to iceberg.

    RANGE spreads limit orders across [price_low, price_high]. UP_TO keeps buying at the
    best ask (or selling at the best bid) and never crosses limit_price. With iceberg on,
    only visible_count slices are shown. A duration releases that total on a clock and
    stops when the clock ends, even if size is left.
    """

    type: Literal["range_iceberg_executor"] = "range_iceberg_executor"
    connector_name: str
    trading_pair: str
    side: TradeType
    total_amount: Decimal
    size_unit: SizeUnit = SizeUnit.QUOTE
    mode: ExecutionMode = ExecutionMode.RANGE
    price_low: Optional[Decimal] = None
    price_high: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    cap_behavior: CapBehavior = CapBehavior.REST_AT_CAP
    levels: int = 10
    distribution: LadderDistribution = LadderDistribution.LINEAR
    weight: Decimal = Decimal("0")
    post_only: bool = False
    iceberg: bool = False
    visible_count: int = 1
    slice_amount: Optional[Decimal] = None
    size_jitter_pct: Decimal = Decimal("0")
    price_jitter_pct: Decimal = Decimal("0")
    min_order_interval: float = 0.6
    expiry_seconds: Optional[float] = None
    duration: Optional[Union[float, str]] = None
    schedule_slices: Optional[int] = None
    schedule_interval: Optional[Union[float, str]] = None
    timing_jitter_pct: Decimal = Decimal("0")
    catch_up: CatchUpPolicy = CatchUpPolicy.CARRY
    max_slippage: Optional[Decimal] = None
    reduce_only: bool = False
    leverage: int = 1
    random_seed: Optional[int] = None

    @field_validator("side", mode="before")
    @classmethod
    def parse_side(cls, value):
        if isinstance(value, str):
            return TradeType[value.upper()]
        return value

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.duration is None:
            return None
        return parse_duration(self.duration)

    def resolved_schedule(self) -> Optional[tuple]:
        """
        (slice count, interval seconds) for a duration, or None when the whole size is released at once.
        """
        seconds = self.duration_seconds
        if seconds is None:
            return None
        if self.schedule_slices is not None:
            count = int(self.schedule_slices)
            return count, seconds / count
        interval = parse_duration(self.schedule_interval)
        return max(1, int(seconds / interval)), interval

    @model_validator(mode="after")
    def validate_range(self):
        require_non_empty("connector_name", self.connector_name)
        require_trading_pair("trading_pair", self.trading_pair)
        require_directional_side(self.side)
        require_positive("total_amount", self.total_amount)
        if self.mode == ExecutionMode.UP_TO:
            if self.limit_price is None:
                raise ValueError("limit_price is required for UP_TO mode")
            require_positive("limit_price", self.limit_price)
        else:
            if self.price_low is None or self.price_high is None:
                raise ValueError("price_low and price_high are required for RANGE mode")
            require_positive("price_low", self.price_low)
            require_positive("price_high", self.price_high)
            if self.price_low > self.price_high:
                raise ValueError(
                    f"price_low ({self.price_low}) must be less than or equal to price_high ({self.price_high})"
                )
            if self.visible_count > self.levels:
                raise ValueError(f"visible_count ({self.visible_count}) must not exceed levels ({self.levels})")
        require_at_least("levels", self.levels, 1)
        require_at_least("visible_count", self.visible_count, 1)
        if self.slice_amount is not None:
            require_positive("slice_amount", self.slice_amount)
            if self.slice_amount > self.total_amount:
                raise ValueError(
                    f"slice_amount ({self.slice_amount}) must be lower than or equal to total_amount ({self.total_amount})"
                )
        if self.weight < Decimal("-1") or self.weight > Decimal("1"):
            raise ValueError(f"weight ({self.weight}) must be between -1 and 1")
        require_non_negative("size_jitter_pct", self.size_jitter_pct)
        require_non_negative("price_jitter_pct", self.price_jitter_pct)
        require_non_negative("timing_jitter_pct", self.timing_jitter_pct)
        if self.timing_jitter_pct > 1:
            raise ValueError(f"timing_jitter_pct ({self.timing_jitter_pct}) must be lower than or equal to 1")
        require_non_negative("min_order_interval", self.min_order_interval)
        require_at_least("leverage", self.leverage, 1)
        if self.expiry_seconds is not None:
            require_positive("expiry_seconds", self.expiry_seconds)
        if self.max_slippage is not None:
            require_non_negative("max_slippage", self.max_slippage)
        if self.duration is not None:
            try:
                seconds = parse_duration(self.duration)
            except ValueError as exc:
                raise ValueError(f"duration ({self.duration}) must be seconds or a value like 24h, 30m, 45s") from exc
            require_positive("duration", seconds)
            has_slices = self.schedule_slices is not None
            has_interval = self.schedule_interval is not None
            if has_slices == has_interval:
                raise ValueError("Set schedule_slices or schedule_interval with duration, not both and not neither")
            if has_slices:
                require_at_least("schedule_slices", self.schedule_slices, 1)
            else:
                try:
                    interval = parse_duration(self.schedule_interval)
                except ValueError as exc:
                    raise ValueError(
                        f"schedule_interval ({self.schedule_interval}) must be seconds or a value like 1h"
                    ) from exc
                require_positive("schedule_interval", interval)
        elif self.schedule_slices is not None or self.schedule_interval is not None:
            raise ValueError("schedule_slices and schedule_interval need a duration")
        return self
