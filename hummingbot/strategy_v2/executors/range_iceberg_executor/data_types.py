from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import field_validator, model_validator

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase
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


class RangeIcebergExecutorConfig(ExecutorConfigBase):
    """
    Buy or sell a total size inside a price range.

    With iceberg left off, the levels are working limit orders (paced so a 4 request
    per 2 second venue is not flooded). With iceberg on, only visible_count of those
    levels are shown, and the next level is sent when one fills. Prices never leave
    the range.
    """

    type: Literal["range_iceberg_executor"] = "range_iceberg_executor"
    connector_name: str
    trading_pair: str
    side: TradeType
    total_amount: Decimal
    size_unit: SizeUnit = SizeUnit.QUOTE
    price_low: Decimal
    price_high: Decimal
    levels: int = 10
    distribution: LadderDistribution = LadderDistribution.LINEAR
    weight: Decimal = Decimal("0")
    post_only: bool = False
    iceberg: bool = False
    visible_count: int = 1
    size_jitter_pct: Decimal = Decimal("0")
    price_jitter_pct: Decimal = Decimal("0")
    min_order_interval: float = 0.6
    expiry_seconds: Optional[float] = None
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

    @model_validator(mode="after")
    def validate_range(self):
        require_non_empty("connector_name", self.connector_name)
        require_trading_pair("trading_pair", self.trading_pair)
        require_directional_side(self.side)
        require_positive("total_amount", self.total_amount)
        require_positive("price_low", self.price_low)
        require_positive("price_high", self.price_high)
        if self.price_low > self.price_high:
            raise ValueError(
                f"price_low ({self.price_low}) must be less than or equal to price_high ({self.price_high})"
            )
        require_at_least("levels", self.levels, 1)
        require_at_least("visible_count", self.visible_count, 1)
        if self.visible_count > self.levels:
            raise ValueError(f"visible_count ({self.visible_count}) must not exceed levels ({self.levels})")
        if self.weight < Decimal("-1") or self.weight > Decimal("1"):
            raise ValueError(f"weight ({self.weight}) must be between -1 and 1")
        require_non_negative("size_jitter_pct", self.size_jitter_pct)
        require_non_negative("price_jitter_pct", self.price_jitter_pct)
        require_non_negative("min_order_interval", self.min_order_interval)
        require_at_least("leverage", self.leverage, 1)
        if self.expiry_seconds is not None:
            require_positive("expiry_seconds", self.expiry_seconds)
        if self.max_slippage is not None:
            require_non_negative("max_slippage", self.max_slippage)
        return self
