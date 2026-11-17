import logging
import random
from decimal import Decimal
from typing import List, Optional

from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import RangeIcebergExecutorConfig
from hummingbot.strategy_v2.executors.range_iceberg_executor.ladder import (
    LadderLevel,
    build_range_ladder,
    clamp_price,
    quantize_down,
)
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder


class _Slice:
    def __init__(self, level: LadderLevel):
        self.target_price = level.price
        self.target_amount = level.base_amount
        self.tracked: Optional[TrackedOrder] = None

    @property
    def order_id(self) -> Optional[str]:
        return self.tracked.order_id if self.tracked else None

    @property
    def is_working(self) -> bool:
        if self.tracked is None or self.order_id == "skipped":
            return False
        if self.tracked.order is None:
            return True
        return not self.tracked.is_done

    @property
    def is_filled(self) -> bool:
        return bool(self.tracked and self.tracked.order and self.tracked.is_filled)


class RangeIcebergExecutor(ExecutorBase):
    """
    Range ladder and iceberg execution for any connector.

    TWAP and DCA slice by time or by a fixed price list and send each slice once.
    This executor keeps every child inside a price range, can hide all but a few
    slices, and replaces a slice only after it fills.
    """

    _logger: Optional[HummingbotLogger] = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(self, strategy: StrategyV2Base, config: RangeIcebergExecutorConfig, update_interval: float = 1.0,
                 max_retries: int = 10):
        super().__init__(
            strategy=strategy,
            connectors=[config.connector_name],
            config=config,
            update_interval=update_interval,
            max_retries=max_retries,
        )
        self.config: RangeIcebergExecutorConfig = config
        self._slices: List[_Slice] = []
        self._paused = False
        self._started_at = self._strategy.current_timestamp
        self._last_place_timestamp = Decimal("-1")
        self._rng = random.Random(config.random_seed)
        self._build_plan()

    def _build_plan(self):
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        try:
            levels = build_range_ladder(
                total_amount=self.config.total_amount,
                size_unit=self.config.size_unit,
                price_low=self.config.price_low,
                price_high=self.config.price_high,
                levels=self.config.levels,
                distribution=self.config.distribution,
                weight=self.config.weight,
                side=self.config.side,
                price_step=Decimal(str(rules.min_price_increment)),
                amount_step=Decimal(str(rules.min_base_amount_increment)),
                min_amount=Decimal(str(rules.min_order_size)),
                contract_size=self._contract_size(),
            )
        except ValueError as exc:
            self.logger().error(str(exc))
            self.close_type = CloseType.FAILED
            self._slices = []
            return
        self._slices = [_Slice(level) for level in levels]

    def _contract_size(self) -> Decimal:
        connector = self.connectors[self.config.connector_name]
        getter = getattr(connector, "get_contract_size", None)
        if not callable(getter):
            return Decimal("1")
        try:
            size = Decimal(str(getter(self.config.trading_pair)))
        except Exception:
            return Decimal("1")
        return size if size > 0 else Decimal("1")

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    async def validate_sufficient_balance(self):
        # Limit orders are rejected by the exchange when margin is short. Retries cover that.
        return

    async def control_task(self):
        if self.close_type == CloseType.FAILED and not self._slices:
            self.stop()
            return
        self._refresh_tracked_orders()
        if self._expired():
            self.early_stop(keep_position=False, close_type=CloseType.EXPIRED)
            return
        if self._all_slices_filled():
            self.close_execution_finished()
            return
        if self.status != RunnableStatus.RUNNING or self._paused or self._slippage_blocks_new_orders():
            return
        if not self._rate_limit_allows_place():
            return
        slot = self._next_slot()
        if slot is not None:
            self._place_slice(slot)

    def early_stop(self, keep_position: bool = False, close_type: Optional[CloseType] = None):
        self.close_type = close_type or (CloseType.POSITION_HOLD if keep_position else CloseType.EARLY_STOP)
        self.close_timestamp = self._strategy.current_timestamp
        self._cancel_working_orders()
        self.stop()

    def close_execution_finished(self):
        self.close_type = CloseType.COMPLETED
        self.close_timestamp = self._strategy.current_timestamp
        self.stop()

    def _place_slice(self, slot: _Slice):
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        price = self._jitter_price(slot.target_price, rules.min_price_increment)
        amount = self._jitter_amount(slot.target_amount, rules.min_base_amount_increment, rules.min_order_size)
        if amount < rules.min_order_size:
            slot.tracked = TrackedOrder(order_id="skipped")
            return
        order_type = OrderType.LIMIT_MAKER if self.config.post_only else OrderType.LIMIT
        position_action = PositionAction.CLOSE if self.config.reduce_only else PositionAction.OPEN
        order_id = self.place_order(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_type=order_type,
            side=self.config.side,
            amount=amount,
            price=price,
            position_action=position_action,
        )
        slot.tracked = TrackedOrder(order_id=order_id)
        slot.target_amount = amount
        slot.target_price = price
        self._last_place_timestamp = Decimal(str(self._strategy.current_timestamp))

    def _jitter_price(self, price: Decimal, price_step: Decimal) -> Decimal:
        if self.config.price_jitter_pct <= 0:
            return price
        swing = (Decimal(str(self._rng.random())) * 2) - Decimal("1")
        jittered = price * (Decimal("1") + self.config.price_jitter_pct * swing)
        return clamp_price(jittered, self.config.price_low, self.config.price_high, price_step)

    def _jitter_amount(self, amount: Decimal, amount_step: Decimal, min_amount: Decimal) -> Decimal:
        if self.config.size_jitter_pct <= 0:
            return amount
        swing = (Decimal(str(self._rng.random())) * 2) - Decimal("1")
        jittered = quantize_down(amount * (Decimal("1") + self.config.size_jitter_pct * swing), amount_step)
        if jittered < min_amount:
            return amount
        return jittered

    def _next_slot(self) -> Optional[_Slice]:
        working = sum(1 for slot in self._slices if slot.is_working)
        limit = self.config.visible_count if self.config.iceberg else len(self._slices)
        if working >= limit:
            return None
        return next((slot for slot in self._slices if slot.tracked is None), None)

    def _rate_limit_allows_place(self) -> bool:
        if self._last_place_timestamp < 0:
            return True
        elapsed = Decimal(str(self._strategy.current_timestamp)) - self._last_place_timestamp
        return elapsed >= Decimal(str(self.config.min_order_interval))

    def _slippage_blocks_new_orders(self) -> bool:
        if self.config.max_slippage is None:
            return False
        if self.config.side is TradeType.BUY:
            market_price = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk)
            return market_price > self.config.price_high * (Decimal("1") + self.config.max_slippage)
        market_price = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid)
        return market_price < self.config.price_low * (Decimal("1") - self.config.max_slippage)

    def _expired(self) -> bool:
        if self.config.expiry_seconds is None:
            return False
        return self._strategy.current_timestamp >= self._started_at + self.config.expiry_seconds

    def _all_slices_filled(self) -> bool:
        if not self._slices:
            return False
        return all(slot.is_filled or (slot.order_id == "skipped") for slot in self._slices)

    def _refresh_tracked_orders(self):
        for slot in self._slices:
            if slot.tracked and slot.tracked.order_id and slot.tracked.order_id != "skipped" and slot.tracked.order is None:
                in_flight = self.get_in_flight_order(self.config.connector_name, slot.tracked.order_id)
                if in_flight:
                    slot.tracked.order = in_flight

    def _cancel_working_orders(self):
        for slot in self._slices:
            if slot.is_working and slot.order_id and slot.order_id != "skipped":
                self._strategy.cancel(self.config.connector_name, self.config.trading_pair, slot.order_id)

    def process_order_created_event(self, event_tag: int, market, event):
        slot = self._slot_for(event.order_id)
        if slot and slot.tracked:
            in_flight = self.get_in_flight_order(self.config.connector_name, event.order_id)
            if in_flight:
                slot.tracked.order = in_flight

    def process_order_failed_event(self, event_tag: int, market, event):
        slot = self._slot_for(event.order_id)
        if slot:
            slot.tracked = None
            self._current_retries += 1
            if self._current_retries > self._max_retries:
                self.close_type = CloseType.FAILED
                self.early_stop(close_type=CloseType.FAILED)

    def process_order_completed_event(self, event_tag: int, market, event):
        if self._all_slices_filled():
            self.close_execution_finished()

    def _slot_for(self, order_id: str) -> Optional[_Slice]:
        return next((slot for slot in self._slices if slot.order_id == order_id), None)

    @property
    def filled_base(self) -> Decimal:
        return sum((slot.tracked.executed_amount_base for slot in self._slices if slot.tracked), Decimal("0"))

    @property
    def filled_amount_quote(self) -> Decimal:
        return sum((slot.tracked.executed_amount_quote for slot in self._slices if slot.tracked), Decimal("0"))

    @property
    def total_base(self) -> Decimal:
        return sum((slot.target_amount for slot in self._slices), Decimal("0"))

    @property
    def average_fill_price(self) -> Decimal:
        filled = self.filled_base
        if filled <= 0:
            return Decimal("0")
        return self.filled_amount_quote / filled

    def get_cum_fees_quote(self) -> Decimal:
        return sum((slot.tracked.cum_fees_quote for slot in self._slices if slot.tracked), Decimal("0"))

    def get_net_pnl_quote(self) -> Decimal:
        filled = self.filled_base
        if filled <= 0:
            return Decimal("0")
        mark = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)
        if self.config.side is TradeType.BUY:
            return (mark - self.average_fill_price) * filled - self.cum_fees_quote
        return (self.average_fill_price - mark) * filled - self.cum_fees_quote

    def get_net_pnl_pct(self) -> Decimal:
        quote = self.filled_amount_quote
        if quote <= 0:
            return Decimal("0")
        return self.net_pnl_quote / quote

    def get_custom_info(self) -> dict:
        return {
            "side": self.config.side.name,
            "paused": self._paused,
            "filled_base": str(self.filled_base),
            "filled_quote": str(self.filled_amount_quote),
            "total_base": str(self.total_base),
            "average_fill_price": str(self.average_fill_price),
            "child_order_ids": [slot.order_id for slot in self._slices if slot.order_id],
            "levels": [
                {
                    "price": str(slot.target_price),
                    "amount": str(slot.target_amount),
                    "order_id": slot.order_id,
                    "filled": slot.is_filled,
                }
                for slot in self._slices
            ],
        }
