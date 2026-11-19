from decimal import Decimal
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, field_validator, model_validator

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


class ScheduleStart(str, Enum):
    """When a duration clock starts counting."""

    FIRST_FILL = "FIRST_FILL"
    SUBMIT = "SUBMIT"


class MarginMode(str, Enum):
    """Margin mode sent as openType when it is set. None leaves the account margin unchanged."""

    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


class ExitExecution(str, Enum):
    """How the filled position is closed once the trigger is reached."""

    SINGLE = "SINGLE"
    ICEBERG = "ICEBERG"


class ExitOrderType(str, Enum):
    """
    Order type for a SINGLE take profit.

    LIMIT rests one order at the trigger price. MARKET sends one order when the
    trigger price is reached. A SINGLE stop loss is always a market order. Iceberg
    exits do not use this value.
    """

    LIMIT = "LIMIT"
    MARKET = "MARKET"


class ExitArming(str, Enum):
    """
    When a configured exit becomes eligible.

    ON_FULL_FILL waits until the entry has finished filling. ON_ANY_FILL arms as
    soon as any amount is filled, so a partial fill can be protected. The exit
    size is the filled amount, minus what the exit has already closed.

    A single limit take profit is not a trigger. ON_ANY_FILL posts that limit
    for the filled amount and leaves the entry running. A market take profit,
    an iceberg take profit, and a stop loss are triggers: they cancel the
    remaining entry and close.
    """

    ON_FULL_FILL = "ON_FULL_FILL"
    ON_ANY_FILL = "ON_ANY_FILL"


class ExitState(str, Enum):
    """
    Public exit progress in custom_info["exit_state"].

    NONE means no exit is eligible. ARMED means an exit is waiting for price.
    RESTING means a limit take profit is on the book and has not started the close.
    TRIGGERED means a trigger fired and entry or the other exit is being cancelled.
    CLOSING means close orders are working. DONE means the exit finished.
    """

    NONE = "NONE"
    ARMED = "ARMED"
    RESTING = "RESTING"
    TRIGGERED = "TRIGGERED"
    CLOSING = "CLOSING"
    DONE = "DONE"


class RangeIcebergExitConfig(BaseModel):
    """
    Optional close for the position this executor opens.

    Set percent or price, not both. percent is a fraction of the average entry
    price of the filled amount: 0.02 is 2 percent. For a long, take profit is
    above that average and stop loss is below it. For a short, take profit is
    below and stop loss is above. price is an absolute trigger and does not move
    with the average.

    SINGLE take profit places one limit order at the trigger, or one market order
    when order_type is MARKET. SINGLE stop loss places one market order when the
    trigger is reached.

    ICEBERG closes with the same ladder, schedule, price jitter, and pre-jitter
    reprice as the entry. slices is how many children to use. visible_count is how
    many rest at once. interval spaces those children (seconds, or a string such
    as 30s). Leave interval empty to release every slice as soon as the exit starts.
    The exit clock starts when the exit triggers. It does not wait for the first
    exit fill, and a pause after the trigger freezes it. Set limit_price for an
    up-to close that does not cross that price, or set price_low and price_high
    for a range ladder. Set one of those price styles. A cancelled range rung is
    posted again for its unfilled remainder.

    Take profit and stop loss can both be set. The stop loss can take over any
    unfinished take profit, including a limit that has already filled part of
    the position and an iceberg child that is still resting. It cancels the
    take-profit remainder and closes what is still open. Once a stop loss has
    started, a move back through the take-profit price does not cancel it and
    does not start the take profit.

    A stop-loss iceberg must be able to trade when the trigger prints. For a
    long, the sell floor (limit_price or price_low) has to be at or below the
    trigger. For a short, the buy cap (limit_price or price_high) has to be at
    or above the trigger. A percent trigger is checked when the average fill
    is known. The stop loss always rests at that cap, and if nothing has filled
    when the fallback time ends it closes the rest with a market order.
    """

    percent: Optional[Decimal] = None
    price: Optional[Decimal] = None
    execution: ExitExecution = ExitExecution.SINGLE
    order_type: Optional[ExitOrderType] = None
    slices: int = 1
    visible_count: int = 1
    interval: Optional[Union[float, str]] = None
    limit_price: Optional[Decimal] = None
    price_low: Optional[Decimal] = None
    price_high: Optional[Decimal] = None

    @field_validator("execution", mode="before")
    @classmethod
    def parse_execution(cls, value):
        if isinstance(value, ExitExecution):
            return value
        text = str(getattr(value, "value", value)).upper()
        if text not in (ExitExecution.SINGLE.value, ExitExecution.ICEBERG.value):
            raise ValueError(f"execution ({value}) must be SINGLE or ICEBERG")
        return ExitExecution(text)

    @field_validator("order_type", mode="before")
    @classmethod
    def parse_order_type(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, ExitOrderType):
            return value
        text = str(getattr(value, "value", value)).upper()
        if text not in (ExitOrderType.LIMIT.value, ExitOrderType.MARKET.value):
            raise ValueError(f"order_type ({value}) must be LIMIT or MARKET")
        return ExitOrderType(text)

    @property
    def interval_seconds(self) -> Optional[float]:
        if self.interval is None:
            return None
        return parse_duration(self.interval)

    @model_validator(mode="after")
    def validate_exit(self):
        has_percent = self.percent is not None
        has_price = self.price is not None
        if has_percent == has_price:
            raise ValueError("Set percent or price on an exit, not both and not neither")
        if has_percent:
            require_positive("percent", self.percent)
            if self.percent >= 1:
                raise ValueError(f"percent ({self.percent}) must be lower than 1")
        if has_price:
            require_positive("price", self.price)
        require_at_least("slices", self.slices, 1)
        require_at_least("visible_count", self.visible_count, 1)
        if self.execution == ExitExecution.ICEBERG:
            if self.visible_count > self.slices:
                raise ValueError(
                    f"visible_count ({self.visible_count}) must not exceed slices ({self.slices})"
                )
            has_range = self.price_low is not None or self.price_high is not None
            has_limit = self.limit_price is not None
            if has_range == has_limit:
                raise ValueError(
                    "An iceberg exit needs limit_price, or price_low and price_high, not both and not neither"
                )
            if has_limit:
                require_positive("limit_price", self.limit_price)
            else:
                if self.price_low is None or self.price_high is None:
                    raise ValueError("price_low and price_high are both required for an iceberg exit range")
                require_positive("price_low", self.price_low)
                require_positive("price_high", self.price_high)
                if self.price_low > self.price_high:
                    raise ValueError(
                        f"price_low ({self.price_low}) must be less than or equal to price_high ({self.price_high})"
                    )
            if self.interval is not None:
                try:
                    seconds = parse_duration(self.interval)
                except ValueError as exc:
                    raise ValueError(
                        f"interval ({self.interval}) must be seconds or a value like 30s, 1m"
                    ) from exc
                require_positive("interval", seconds)
        return self


class RangeIcebergExecutorConfig(ExecutorConfigBase):
    """
    Buy or sell a total size as a range ladder or as an up-to iceberg.

    RANGE spreads limit orders across [price_low, price_high]. UP_TO keeps buying at the
    best ask (or selling at the best bid) and never crosses limit_price. With iceberg on,
    only visible_count slices are shown. A duration releases that total on a clock and
    stops when the clock ends, even if size is left. The clock starts at the first fill
    unless schedule_start is SUBMIT, which starts it when the executor is created.

    take_profit and stop_loss are optional and default to off. When either is set, the
    filled position is closed with a single order or with another iceberg. See
    RangeIcebergExitConfig. exit_arming chooses whether that close waits for a full fill.
    An iceberg exit schedule starts when that exit triggers. It does not wait for a
    first exit fill.

    When the entry duration or expiry ends with a fill and an exit is configured,
    the remaining entry is cancelled and the exit is armed on the filled amount.
    Once an exit is armed or active, the entry clock no longer stops the executor.

    early_stop cancels software entry and exit orders and does not close the
    filled position. keep_position=True records POSITION_HOLD and is the API
    shutdown path: software orders are cancelled and the exchange stop is left
    working. A user Stop and keep_position=False cancel it too. A stop loss
    that exhausted its close retries places the exchange stop again before it
    stops, and leaves that stop working.

    protective_exchange_stop defaults to on when the connector implements
    place_position_stop_order, and off otherwise. The exchange stop is a
    backstop past the software stop trigger by protective_stop_buffer_pct
    (default 0.5 percent) in the adverse direction. The software stop waits
    until take-profit and entry cancels are confirmed, then cancels every
    exchange stop before its own close. If those cancels stay open past
    stop_loss_fallback_seconds, the backstop stays and the software close is
    only the size not covered by the live stops and the unconfirmed take profit.
    A larger position at the same backstop price places the full open size and
    keeps the id MEXC returns. A changed stop price is moved with
    change_stop_plan_price and then resized at that price. Cancel-and-place
    is only used when that reprice fails. The place never asks for more than the
    open size minus resting reduce-only close limits. If a move fails, the
    old stop is restored before any take profit is posted again. That restore
    retries at 0.5s, 1s, then 2s, and after four failures waits
    stop_loss_fallback_seconds. The take profit waits at most
    exchange_stop_hold_seconds (default 5) for that stop, including while a
    restore is still failing. After the hold, the take profit is posted and
    the stop keeps retrying. A failed grow leaves the take profit at the size
    the live stop covers. After three failed placements the fast retries
    leave it up, and each slow retry clears it once, places, and posts it
    again. A connector with no place_position_stop_order
    does not hold the take profit and does not cancel one that is already up.
    A stop-loss exit is never held. exchange_stop_released is saved so a
    resumed stop-loss close does not place a new stop. Every id is listed in
    exchange_stop_order_ids. Resuming from custom_info restores an exit_state
    of ARMED, RESTING, or CLOSING, re-posts a missing exit from open_base,
    and keeps the saved exchange stop ids. Only one range iceberg may be
    open on the same account, connector, pair, and position side. An open
    executor is edited in place with update_config (total amount, price range,
    take profit, and stop loss) instead of opening a second one. Those edits
    are stored on custom_info config_edits, in the executor's own size unit,
    and resume applies that object only.     The reported total_amount, price_low,
    price_high, take_profit, and stop_loss are the live values and are not
    applied again. custom_info reports size_unit beside total_amount.
    resume_entries false
    places no entry orders and sizes the restored open amount in base, even
    when the saved config used quote.

    allow_adverse_take_profit permits a take-profit price that is not past the
    average entry. stop_loss_fallback_seconds is how long a stop-loss iceberg
    may go without a new fill before the remainder is a market order.
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
    schedule_start: ScheduleStart = ScheduleStart.FIRST_FILL
    timing_jitter_pct: Decimal = Decimal("0")
    catch_up: CatchUpPolicy = CatchUpPolicy.CARRY
    max_slippage: Optional[Decimal] = None
    reduce_only: bool = False
    leverage: int = 1
    margin_mode: Optional[MarginMode] = None
    random_seed: Optional[int] = None
    take_profit: Optional[RangeIcebergExitConfig] = None
    stop_loss: Optional[RangeIcebergExitConfig] = None
    exit_arming: ExitArming = ExitArming.ON_FULL_FILL
    allow_adverse_take_profit: bool = False
    protective_exchange_stop: Optional[bool] = None
    protective_stop_buffer_pct: Decimal = Decimal("0.005")
    stop_loss_fallback_seconds: float = 30
    exchange_stop_hold_seconds: float = 5

    @field_validator("side", mode="before")
    @classmethod
    def parse_side(cls, value):
        if isinstance(value, str):
            return TradeType[value.upper()]
        return value

    @field_validator("schedule_start", mode="before")
    @classmethod
    def parse_schedule_start(cls, value):
        if isinstance(value, ScheduleStart):
            return value
        text = str(getattr(value, "value", value)).upper()
        if text not in (ScheduleStart.FIRST_FILL.value, ScheduleStart.SUBMIT.value):
            raise ValueError(f"schedule_start ({value}) must be FIRST_FILL or SUBMIT")
        return ScheduleStart(text)

    @field_validator("exit_arming", mode="before")
    @classmethod
    def parse_exit_arming(cls, value):
        if isinstance(value, ExitArming):
            return value
        text = str(getattr(value, "value", value)).upper()
        if text not in (ExitArming.ON_FULL_FILL.value, ExitArming.ON_ANY_FILL.value):
            raise ValueError(f"exit_arming ({value}) must be ON_FULL_FILL or ON_ANY_FILL")
        return ExitArming(text)

    @field_validator("margin_mode", mode="before")
    @classmethod
    def parse_margin_mode(cls, value):
        if value is None:
            return None
        text = str(getattr(value, "value", value)).upper()
        if text not in (MarginMode.ISOLATED.value, MarginMode.CROSS.value):
            raise ValueError(f"margin_mode ({value}) must be ISOLATED or CROSS")
        return MarginMode(text)

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
        require_positive("stop_loss_fallback_seconds", self.stop_loss_fallback_seconds)
        require_positive("exchange_stop_hold_seconds", self.exchange_stop_hold_seconds)
        require_positive("protective_stop_buffer_pct", self.protective_stop_buffer_pct)
        if self.protective_stop_buffer_pct >= 1:
            raise ValueError(
                f"protective_stop_buffer_pct ({self.protective_stop_buffer_pct}) must be lower than 1"
            )
        if self.reduce_only and (self.take_profit is not None or self.stop_loss is not None):
            raise ValueError(
                "take_profit and stop_loss cannot be set when reduce_only is true, "
                "because those exits close the position this executor opens"
            )
        self._prepare_exit("take_profit", self.take_profit, is_stop=False)
        self._prepare_exit("stop_loss", self.stop_loss, is_stop=True)
        self._reject_adverse_take_profit_price()
        self._reject_stop_bounds_on_the_wrong_side()
        if (
            self.take_profit is not None
            and self.stop_loss is not None
            and self.take_profit.price is not None
            and self.stop_loss.price is not None
        ):
            if self.side == TradeType.BUY and self.stop_loss.price >= self.take_profit.price:
                raise ValueError(
                    f"stop_loss price ({self.stop_loss.price}) must be lower than "
                    f"take_profit price ({self.take_profit.price}) for a BUY side"
                )
            if self.side == TradeType.SELL and self.stop_loss.price <= self.take_profit.price:
                raise ValueError(
                    f"stop_loss price ({self.stop_loss.price}) must be higher than "
                    f"take_profit price ({self.take_profit.price}) for a SELL side"
                )
        return self

    def _prepare_exit(self, field: str, exit_config: Optional[RangeIcebergExitConfig], is_stop: bool):
        if exit_config is None or exit_config.execution != ExitExecution.SINGLE:
            return
        if is_stop:
            if exit_config.order_type == ExitOrderType.LIMIT:
                raise ValueError(
                    f"{field} order_type (LIMIT) must be MARKET: a single stop loss is a market order"
                )
            exit_config.order_type = ExitOrderType.MARKET
        elif exit_config.order_type is None:
            exit_config.order_type = ExitOrderType.LIMIT

    def _reject_adverse_take_profit_price(self):
        """
        Reject a take-profit price that is impossible for every fill.

        A buy range fills at or above price_low, so a price at or below that
        low is never a profit. A sell range fills at or below price_high, so a
        price at or above that high is never a profit. An up-to fill can land
        on either side of limit_price enough that the average still clears the
        take profit, so that case is checked against the actual average once a
        fill exists. Set allow_adverse_take_profit to keep a price anyway. A
        percent trigger is always on the profitable side of the average.
        """
        cfg = self.take_profit
        if cfg is None or cfg.price is None or self.allow_adverse_take_profit:
            return
        if self.side == TradeType.BUY and self.mode != ExecutionMode.UP_TO and self.price_low is not None:
            if cfg.price <= self.price_low:
                raise ValueError(
                    f"take_profit price ({cfg.price}) must be above price_low ({self.price_low}) "
                    "for a BUY, because the average entry cannot be below that price. "
                    "Set allow_adverse_take_profit to allow it."
                )
        if self.side == TradeType.SELL and self.mode != ExecutionMode.UP_TO and self.price_high is not None:
            if cfg.price >= self.price_high:
                raise ValueError(
                    f"take_profit price ({cfg.price}) must be below price_high ({self.price_high}) "
                    "for a SELL, because the average entry cannot be above that price. "
                    "Set allow_adverse_take_profit to allow it."
                )

    def _reject_stop_bounds_on_the_wrong_side(self):
        """
        A stop-loss iceberg whose bound sits on the wrong side of an absolute
        trigger would rest away from the market. Percent triggers are checked
        once the average fill is known.
        """
        cfg = self.stop_loss
        if cfg is None or cfg.execution != ExitExecution.ICEBERG or cfg.price is None:
            return
        trigger = cfg.price
        if self.side == TradeType.BUY:
            if cfg.limit_price is not None and cfg.limit_price > trigger:
                raise ValueError(
                    f"stop_loss limit_price ({cfg.limit_price}) must be less than or equal to "
                    f"the trigger ({trigger}) for a BUY, so the sell does not rest above the market"
                )
            if cfg.price_low is not None and cfg.price_low > trigger:
                raise ValueError(
                    f"stop_loss price_low ({cfg.price_low}) must be less than or equal to "
                    f"the trigger ({trigger}) for a BUY, so the sell does not rest above the market"
                )
            return
        if cfg.limit_price is not None and cfg.limit_price < trigger:
            raise ValueError(
                f"stop_loss limit_price ({cfg.limit_price}) must be greater than or equal to "
                f"the trigger ({trigger}) for a SELL, so the buy does not rest below the market"
            )
        if cfg.price_high is not None and cfg.price_high < trigger:
            raise ValueError(
                f"stop_loss price_high ({cfg.price_high}) must be greater than or equal to "
                f"the trigger ({trigger}) for a SELL, so the buy does not rest below the market"
            )
