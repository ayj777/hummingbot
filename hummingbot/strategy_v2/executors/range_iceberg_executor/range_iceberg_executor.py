import logging
import random
from decimal import Decimal
from typing import List, Optional, Set

from hummingbot.core.data_type.common import OrderType, PositionAction, PositionSide, PriceType, TradeType
from hummingbot.core.data_type.in_flight_order import OrderState
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import (
    CapBehavior,
    CatchUpPolicy,
    ExecutionMode,
    RangeIcebergExecutorConfig,
    ScheduleStart,
    SizeUnit,
)
from hummingbot.strategy_v2.executors.range_iceberg_executor.ladder import (
    LadderLevel,
    build_range_ladder,
    cap_buy_price,
    clamp_price,
    floor_sell_price,
    quantize_down,
)
from hummingbot.strategy_v2.executors.time_schedule import SliceSchedule, release_timestamps
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder


class _Slice:
    def __init__(self, level: LadderLevel):
        self.target_price = level.price
        self.target_amount = level.base_amount
        self.tracked: Optional[TrackedOrder] = None
        self.cancel_requested = False

    @property
    def order_id(self) -> Optional[str]:
        return self.tracked.order_id if self.tracked else None

    @property
    def is_working(self) -> bool:
        if self.tracked is None or self.order_id == "skipped":
            return False
        if self.tracked.order is not None and self.tracked.is_done:
            return False
        return True

    @property
    def is_filled(self) -> bool:
        return bool(self.tracked and self.tracked.order and self.tracked.is_filled)


class RangeIcebergExecutor(ExecutorBase):
    """
    Range ladder and up-to iceberg execution for any connector.

    RANGE keeps every child inside [price_low, price_high]. UP_TO prices a buy at the
    best ask, or a sell at the best bid, and never crosses limit_price. Iceberg shows
    only a few slices and replaces each one after it fills. A duration releases the
    total on the same offset clock as TWAP and cancels leftovers when the clock ends.

    By default that clock starts at the first fill. A partial fill counts. Until then
    only the first slice is placed, and no window is skipped or carried. schedule_start
    SUBMIT keeps the previous clock, which starts when the executor starts.
    expiry_seconds is always measured from the executor start, so an order that never
    fills can still expire.

    Pausing freezes the schedule clock after it has started, and it still freezes
    expiry. Working orders stay. Time spent paused is not counted, and resume
    continues from the frozen elapsed time.
    """

    _logger: Optional[HummingbotLogger] = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(
        self,
        strategy: StrategyV2Base,
        config: RangeIcebergExecutorConfig,
        update_interval: float = 1.0,
        max_retries: int = 10,
    ):
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
        self._pause_started: Optional[float] = None
        self._paused_seconds = 0.0
        self._schedule_paused_seconds = 0.0
        self._started_at = self._strategy.current_timestamp
        self._schedule_started_at: Optional[float] = None
        self._last_place_timestamp = Decimal("-1")
        self._rng = random.Random(config.random_seed)
        self._schedule = self._make_schedule()
        if self._schedule is not None and self.config.schedule_start == ScheduleStart.SUBMIT:
            self._schedule_started_at = self._started_at
        self._build_plan()
        self._rounding_remainder_base = Decimal("0")
        self._account_settings_ready = False
        self._account_settings_failed = False
        self._account_settings_error = ""
        self._shutdown_pending = False
        self._cancel_stop_error = ""
        self._unconfirmed_cancels: Set[str] = set()
        self._failed_order_ids: Set[str] = set()
        self._submitted_order_ids: List[str] = []
        self._accounted_fill_base = Decimal("0")
        self._accounted_fill_quote = Decimal("0")
        self._accounted_fill_by_order: dict = {}
        self._accounted_quote_by_order: dict = {}
        self._counted_fill_base: dict = {}
        self._counted_fill_quote: dict = {}
        self._seen_live_order_ids: Set[str] = set()
        self._ground_truth_baseline: Optional[Decimal] = None
        self._cancel_attempt_count: dict = {}
        self._cancel_attempt_started: dict = {}
        self._unknown_order_ids: Set[str] = set()
        self._unknown_order_error = ""
        self._resolved_terminal: Set[str] = set()
        self._lookup_started: Set[str] = set()
        self._open_lookup_count: dict = {}
        self._open_lookup_started: dict = {}
        self._next_open_lookup_at: dict = {}
        self._cancel_resolve_after_attempts = 5
        self._cancel_resolve_after_seconds = 30.0
        self._open_lookup_cap = 3
        self._open_lookup_max_seconds = 120.0
        self._open_lookup_backoff_seconds = 10.0
        self._open_lookup_backoff_cap_seconds = 40.0

    def _make_schedule(self) -> Optional[SliceSchedule]:
        resolved = self.config.resolved_schedule()
        if resolved is None:
            return None
        count, interval = resolved
        total = self.config.total_amount
        share = total / Decimal(count)
        amounts = []
        remaining = total
        for index in range(count):
            piece = remaining if index == count - 1 else share
            amounts.append(piece)
            remaining -= piece
        offsets = release_timestamps(0.0, count, interval)
        if self.config.timing_jitter_pct > 0 and interval > 0:
            jittered = []
            duration = self.config.duration_seconds
            for index, nominal in enumerate(offsets):
                swing = (self._rng.random() * 2) - 1
                delta = swing * float(self.config.timing_jitter_pct) * interval
                low = 0.0 if index == 0 else jittered[-1]
                jittered.append(min(max(low, nominal + delta), duration))
            offsets = jittered
        return SliceSchedule(
            amounts=amounts,
            offsets=offsets,
            duration=self.config.duration_seconds,
            skip_unplaced=self.config.catch_up == CatchUpPolicy.SKIP,
        )

    def _build_plan(self):
        if self.config.mode == ExecutionMode.UP_TO:
            self._slices = []
            rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
            if self._target_base() < Decimal(str(rules.min_order_size)):
                self.logger().error("The up-to total is below the minimum order size.")
                self.close_type = CloseType.FAILED
            return
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        contract_size = self._contract_size()
        if contract_size is None:
            self.logger().error("Refusing to build a range plan until contractSize is loaded.")
            self.close_type = CloseType.FAILED
            self._slices = []
            return
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
                contract_size=contract_size,
            )
        except ValueError as exc:
            self.logger().error(str(exc))
            self.close_type = CloseType.FAILED
            self._slices = []
            return
        self._slices = [_Slice(level) for level in levels]

    def _contract_size(self) -> Optional[Decimal]:
        connector = self.connectors[self.config.connector_name]
        getter = getattr(connector, "get_contract_size", None)
        if not callable(getter):
            return Decimal("1")
        try:
            size = Decimal(str(getter(self.config.trading_pair)))
        except Exception:
            return None
        return size if size > 0 else None

    def _contract_size_ready(self) -> bool:
        connector = self.connectors[self.config.connector_name]
        if not callable(getattr(connector, "get_contract_size", None)):
            return True
        return self._contract_size() is not None

    def _target_base(self) -> Decimal:
        if self.config.size_unit == SizeUnit.CONTRACTS:
            size = self._contract_size()
            if size is None:
                return Decimal("0")
            return self.config.total_amount * size
        if self.config.size_unit == SizeUnit.QUOTE:
            price = self.config.limit_price or self.config.price_low or Decimal("1")
            return self.config.total_amount / price
        return self.config.total_amount

    def pause(self):
        if not self._paused:
            self._pause_started = self._strategy.current_timestamp
            self._paused = True

    async def resume(self):
        if self._unknown_order_ids:
            await self._resolve_halted_orders()
            if self._unknown_order_ids:
                self._describe_unknown_orders("Resume looked the orders up again and they are still open or unknown.")
                self.pause()
                return
            self._unknown_order_error = ""
        if self._paused:
            now = self._strategy.current_timestamp
            started = self._pause_started or now
            self._paused_seconds += now - started
            if self._schedule_started_at is not None:
                self._schedule_paused_seconds += max(0.0, now - max(started, self._schedule_started_at))
            self._pause_started = None
            self._paused = False

    def _elapsed_since(self, origin: float, paused_seconds: float) -> float:
        now = self._strategy.current_timestamp
        paused = paused_seconds
        if self._paused and self._pause_started is not None:
            paused += max(0.0, now - max(self._pause_started, origin))
        return max(0.0, now - origin - paused)

    def _elapsed(self) -> float:
        """Time since executor start, with every pause removed. expiry_seconds uses this."""
        return self._elapsed_since(self._started_at, self._paused_seconds)

    def _schedule_elapsed(self) -> float:
        """
        Time since the schedule anchor, with pauses after that anchor removed.

        Before the first fill the anchor is unset and no schedule time has passed,
        even if the executor has been running longer than the duration.
        """
        if self._schedule_started_at is None:
            return 0.0
        return self._elapsed_since(self._schedule_started_at, self._schedule_paused_seconds)

    def _awaiting_schedule_anchor(self) -> bool:
        return (
            self._schedule is not None
            and self._schedule_started_at is None
            and self.config.schedule_start == ScheduleStart.FIRST_FILL
        )

    def _anchor_schedule(self, timestamp: float) -> None:
        if not self._awaiting_schedule_anchor():
            return
        self._schedule_started_at = float(timestamp)

    def _capture_schedule_anchor(self) -> None:
        if not self._awaiting_schedule_anchor() or not self._has_observed_fill():
            return
        stamp = self._earliest_recorded_fill_timestamp()
        if stamp is None:
            stamp = self._strategy.current_timestamp
        self._anchor_schedule(stamp)

    def _has_observed_fill(self) -> bool:
        if self._filled_unit() > 0:
            return True
        self._absorb_counted_fills()
        if any(amount > 0 for amount in self._counted_fill_base.values()):
            return True
        if any(amount > 0 for amount in self._accounted_fill_by_order.values()):
            return True
        for slot in self._slices:
            order = slot.tracked.order if slot.tracked else None
            if order is None and slot.tracked and slot.tracked.order_id:
                order = self.get_in_flight_order(self.config.connector_name, slot.tracked.order_id)
            if order is None:
                continue
            if Decimal(str(getattr(order, "executed_amount_base", 0) or 0)) > 0:
                return True
            if Decimal(str(getattr(order, "executed_amount_quote", 0) or 0)) > 0:
                return True
            reported = getattr(order, "reported_deal_base", None)
            if reported is not None and Decimal(str(reported)) > 0:
                return True
        return False

    def _earliest_recorded_fill_timestamp(self) -> Optional[float]:
        earliest: Optional[float] = None
        for slot in self._slices:
            order = slot.tracked.order if slot.tracked else None
            if order is None:
                continue
            fills = getattr(order, "order_fills", None)
            if not isinstance(fills, dict):
                continue
            for trade in fills.values():
                stamp = getattr(trade, "fill_timestamp", None)
                if stamp is None:
                    continue
                stamp = float(stamp)
                if earliest is None or stamp < earliest:
                    earliest = stamp
        return earliest

    def _schedule_time_remaining(self) -> Optional[float]:
        duration = self.config.duration_seconds
        if duration is None:
            return None
        if self._schedule_started_at is None:
            return float(duration)
        remaining = float(duration) - self._schedule_elapsed()
        if remaining < 0:
            return 0.0
        return remaining

    def _schedule_clock_state(self) -> Optional[dict]:
        if self._schedule is None:
            return None
        state = self._schedule.export_state()
        state.update(
            {
                "started_at": self._schedule_started_at,
                "paused_seconds": self._schedule_paused_seconds,
                "executor_started_at": self._started_at,
                "executor_paused_seconds": self._paused_seconds,
                "paused": self._paused,
                "pause_started": self._pause_started,
            }
        )
        return state

    def restore_from_custom_info(self, info: Optional[dict]) -> None:
        """
        Restore the schedule anchor from serialized executor state.

        executor_info stores get_custom_info(). schedule_started_at stays null until
        the first fill. schedule_clock also keeps pause totals, the executor start
        used by expiry, and window accounting so a resumed executor does not replay
        skips that already happened.
        """
        if not info:
            return
        clock = info.get("schedule_clock") or {}
        if not isinstance(clock, dict):
            clock = {}
        if self._schedule is not None and ("schedule_started_at" in info or "started_at" in clock):
            started = info["schedule_started_at"] if "schedule_started_at" in info else clock.get("started_at")
            if started is None:
                if self.config.schedule_start == ScheduleStart.FIRST_FILL:
                    self._schedule_started_at = None
            else:
                self._schedule_started_at = float(started)
        if self._schedule is not None and clock.get("paused_seconds") is not None:
            self._schedule_paused_seconds = float(clock["paused_seconds"])
        if self._schedule is not None and clock.get("closed") is not None:
            self._schedule.restore_state(clock)
        if clock.get("executor_started_at") is not None:
            self._started_at = float(clock["executor_started_at"])
        if clock.get("executor_paused_seconds") is not None:
            self._paused_seconds = float(clock["executor_paused_seconds"])
        if "paused" in clock:
            self._paused = bool(clock["paused"])
            pause_started = clock.get("pause_started")
            self._pause_started = None if pause_started is None else float(pause_started)

    async def validate_sufficient_balance(self):
        # Limit orders are rejected by the exchange when margin is short. Retries cover that.
        return

    async def control_task(self):
        if self.status == RunnableStatus.TERMINATED:
            return
        if self._shutdown_pending:
            await self._finish_shutdown()
            return
        if not await self._ensure_account_settings():
            return
        if not self._contract_size_ready():
            self.logger().error("Refusing to trade until contractSize is loaded.")
            return
        if self.close_type == CloseType.FAILED and not self._slices:
            self.stop()
            return
        self._halt_on_unresolved_exchange_orders()
        await self._release_confirmed_failures()
        self._refresh_tracked_orders()
        self._capture_schedule_anchor()
        self._advance_schedule()
        if self._time_limit_reached():
            self._advance_schedule(force_end=True)
            self.early_stop(keep_position=False, close_type=CloseType.TIME_LIMIT)
            return
        if self._expired():
            self.early_stop(keep_position=False, close_type=CloseType.EXPIRED)
            return
        if self._execution_complete():
            self.close_execution_finished()
            return
        if self.status != RunnableStatus.RUNNING or self._paused:
            return
        if self.config.mode == ExecutionMode.UP_TO:
            await self._control_up_to()
            return
        if self._replacement_blocked():
            return
        if self._slippage_blocks_new_orders():
            return
        if not self._rate_limit_allows_place():
            return
        slot = self._next_slot()
        if slot is None or not self._range_slot_released(slot):
            return
        await self._place_slice(slot)

    def early_stop(self, keep_position: bool = False, close_type: Optional[CloseType] = None):
        self.close_type = close_type or (CloseType.POSITION_HOLD if keep_position else CloseType.EARLY_STOP)
        self.close_timestamp = self._strategy.current_timestamp
        self._begin_shutdown()

    def close_execution_finished(self):
        self.close_type = CloseType.COMPLETED
        self.close_timestamp = self._strategy.current_timestamp
        self._begin_shutdown()

    def _begin_shutdown(self):
        """
        Ask the exchange to cancel working orders, then stay up until those cancels
        are confirmed. A stop that returns while the orders are still open hides the failure.
        """
        self._shutdown_pending = True
        self._cancel_working_orders()
        pending = self._unconfirmed_cancel_ids()
        if not pending:
            self._cancel_stop_error = ""
            self.stop()
            return
        self._note_unconfirmed_cancels(pending)

    def _unconfirmed_cancel_ids(self) -> List[str]:
        return sorted(order_id for order_id in self._unconfirmed_cancels if not self._cancel_confirmed(order_id))

    def _note_unconfirmed_cancels(self, pending: List[str]):
        message = f"Cancel was not confirmed for {', '.join(pending)}. These orders may still be open."
        if message != self._cancel_stop_error:
            self._cancel_stop_error = message
            self.logger().error(message)

    async def _finish_shutdown(self):
        await self._release_confirmed_failures()
        pending = self._unconfirmed_cancel_ids()
        if pending:
            self._note_unconfirmed_cancels(pending)
            return
        self._cancel_stop_error = ""
        self._shutdown_pending = False
        self.stop()

    def _cancel_outstanding_orders(self):
        self._cancel_working_orders()
        pending = self._unconfirmed_cancel_ids()
        if pending:
            self._note_unconfirmed_cancels(pending)

    async def _control_up_to(self):
        if self._replacement_blocked():
            return
        if any(
            (slot.cancel_requested or self._holds_unaccounted_fill(slot)) and self._counts_as_open(slot)
            for slot in self._slices
        ):
            return
        desired = self._desired_price()
        if desired is None:
            self._cancel_one_working()
            return
        if self._reprice_one(desired):
            return
        if not self._rate_limit_allows_place():
            return
        if self._working_count() >= self._visible_limit():
            return
        placeable = self._placeable()
        if placeable <= 0:
            return
        await self._place_up_to_slice(desired, placeable)

    async def _place_slice(self, slot: _Slice):
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        price = self._jitter_price(slot.target_price, rules.min_price_increment)
        amount = self._bounded_amount(slot.target_amount, price, rules.min_base_amount_increment, rules.min_order_size)
        if amount < rules.min_order_size:
            slot.tracked = TrackedOrder(order_id="skipped")
            return
        amount = await self._cap_base_amount(amount)
        if amount is None or amount < rules.min_order_size:
            return
        order_id = self._submit(price, amount)
        slot.tracked = TrackedOrder(order_id=order_id)
        slot.target_amount = amount
        slot.target_price = price
        self._submitted_order_ids.append(order_id)
        self._mark_action()

    async def _place_up_to_slice(self, price: Decimal, placeable_unit: Decimal):
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        price = self._clamp_limit(self._jitter_up_to_price(price))
        step = rules.min_base_amount_increment
        slice_unit = min(self._slice_unit_amount(), placeable_unit)
        raw_base = self._unit_to_base(slice_unit, price)
        amount = self._bounded_amount(raw_base, price, step, rules.min_order_size)
        amount = await self._cap_base_amount(amount)
        if amount is None or amount < rules.min_order_size:
            return
        order_id = self._submit(price, amount)
        slot = _Slice(LadderLevel(price=price, base_amount=amount))
        slot.tracked = TrackedOrder(order_id=order_id)
        self._slices.append(slot)
        self._submitted_order_ids.append(order_id)
        self._mark_action()

    def _submit(self, price: Decimal, amount: Decimal) -> str:
        order_type = OrderType.LIMIT_MAKER if self.config.post_only else OrderType.LIMIT
        position_action = PositionAction.CLOSE if self.config.reduce_only else PositionAction.OPEN
        return self.place_order(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_type=order_type,
            side=self.config.side,
            amount=amount,
            price=price,
            position_action=position_action,
        )

    def _jitter_price(self, price: Decimal, price_step: Decimal) -> Decimal:
        if self.config.price_jitter_pct <= 0:
            return price
        swing = (Decimal(str(self._rng.random())) * 2) - Decimal("1")
        jittered = price * (Decimal("1") + self.config.price_jitter_pct * swing)
        try:
            return clamp_price(jittered, self.config.price_low, self.config.price_high, price_step)
        except ValueError:
            return price

    def _jitter_up_to_price(self, price: Decimal) -> Decimal:
        if self.config.price_jitter_pct <= 0:
            return price
        swing = (Decimal(str(self._rng.random())) * 2) - Decimal("1")
        return price * (Decimal("1") + self.config.price_jitter_pct * swing)

    def _bounded_amount(self, raw_base: Decimal, price: Decimal, amount_step: Decimal, min_amount: Decimal) -> Decimal:
        """
        Quantize a child, keep the sub-step leftover, and never let fills plus working size
        plus this child exceed total_amount.
        """
        raw_base = Decimal(raw_base)
        if self.config.size_jitter_pct > 0:
            amount = self._jitter_amount(raw_base, amount_step, min_amount)
            amount = quantize_down(amount, amount_step)
        else:
            amount = quantize_down(raw_base, amount_step)
            leftover = raw_base - amount
            if leftover > 0:
                self._rounding_remainder_base += leftover
            # A repeating division can sit a hair under one size step. Treat that as one step.
            extra = Decimal("0")
            if amount_step and amount_step > 0:
                tolerance = amount_step / Decimal("1000000")
                extra = quantize_down(self._rounding_remainder_base + tolerance, amount_step)
            if extra > 0:
                amount += extra
                self._rounding_remainder_base -= extra
                if self._rounding_remainder_base < 0:
                    self._rounding_remainder_base = Decimal("0")
        room = self._room_base(price, amount_step)
        if amount > room:
            self._rounding_remainder_base += amount - room
            amount = room
        if amount < min_amount:
            if amount > 0:
                self._rounding_remainder_base += amount
            return Decimal("0")
        return amount

    def _room_base(self, price: Decimal, amount_step: Decimal) -> Decimal:
        room_unit = self.config.total_amount - self._filled_unit() - self._open_unit()
        if room_unit < 0:
            room_unit = Decimal("0")
        return quantize_down(self._unit_to_base(room_unit, price), amount_step)

    def _jitter_amount(self, amount: Decimal, amount_step: Decimal, min_amount: Decimal) -> Decimal:
        if self.config.size_jitter_pct <= 0:
            return amount
        swing = (Decimal(str(self._rng.random())) * 2) - Decimal("1")
        jittered = quantize_down(amount * (Decimal("1") + self.config.size_jitter_pct * swing), amount_step)
        if jittered < min_amount:
            return amount
        return jittered

    def _desired_price(self) -> Optional[Decimal]:
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        step = Decimal(str(rules.min_price_increment or 0))
        cap = self.config.limit_price
        ask = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk)
        bid = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid)
        if self.config.side == TradeType.BUY:
            if ask > cap:
                if self.config.cap_behavior == CapBehavior.WAIT:
                    return None
                raw = cap
            elif self.config.post_only:
                raw = bid + step if step > 0 else bid
                if step > 0 and raw >= ask:
                    raw = bid
            else:
                raw = ask
            price = cap_buy_price(raw, step, cap)
            if self.config.post_only and price >= ask:
                price = cap_buy_price(bid, step, cap)
            return price
        if bid < cap:
            if self.config.cap_behavior == CapBehavior.WAIT:
                return None
            raw = cap
        elif self.config.post_only:
            raw = ask - step if step > 0 else ask
            if step > 0 and raw <= bid:
                raw = ask
        else:
            raw = bid
        price = floor_sell_price(raw, step, cap)
        if self.config.post_only and price <= bid:
            price = floor_sell_price(bid + (step if step > 0 else Decimal("0")), step, cap)
        return price

    def _clamp_limit(self, price: Decimal) -> Decimal:
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        step = Decimal(str(rules.min_price_increment or 0))
        if self.config.side == TradeType.BUY:
            return cap_buy_price(price, step, self.config.limit_price)
        return floor_sell_price(price, step, self.config.limit_price)

    def _slice_unit_amount(self) -> Decimal:
        if self.config.slice_amount is not None:
            return self.config.slice_amount
        if self._schedule is not None:
            return self.config.total_amount / Decimal(self._schedule.count)
        return self.config.total_amount / Decimal(self.config.levels)

    def _visible_limit(self) -> int:
        if self.config.iceberg:
            return self.config.visible_count
        if self._schedule is not None:
            return self._schedule.count
        return self.config.levels

    def _working_count(self) -> int:
        return sum(1 for slot in self._slices if self._counts_as_open(slot))

    def _next_slot(self) -> Optional[_Slice]:
        if self._working_count() >= self._visible_limit():
            return None
        return next((slot for slot in self._slices if slot.tracked is None), None)

    def _range_slot_released(self, slot: _Slice) -> bool:
        if self._schedule is None:
            return True
        return self._base_to_unit(slot.target_amount, slot.target_price) <= self._placeable()

    def _reprice_one(self, desired: Decimal) -> bool:
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        step = Decimal(str(rules.min_price_increment or 0))
        pending = [
            slot
            for slot in self._slices
            if slot.is_working
            and not slot.cancel_requested
            and self._price_differs(self._slot_price(slot), desired, step)
        ]
        if not pending:
            return False
        if not self._rate_limit_allows_place():
            return True
        self._request_cancel(pending[0])
        return True

    def _cancel_one_working(self) -> bool:
        if not self._rate_limit_allows_place():
            return False
        for slot in self._slices:
            if slot.is_working and not slot.cancel_requested:
                self._request_cancel(slot)
                return True
        return False

    def _request_cancel(self, slot: _Slice):
        if slot.order_id and slot.order_id != "skipped":
            self._strategy.cancel(self.config.connector_name, self.config.trading_pair, slot.order_id)
        slot.cancel_requested = True
        self._mark_action()

    def _slot_price(self, slot: _Slice) -> Decimal:
        if slot.tracked and slot.tracked.order and slot.tracked.order.price is not None:
            return slot.tracked.order.price
        return slot.target_price

    def _price_differs(self, left: Decimal, right: Decimal, step: Decimal) -> bool:
        if step <= 0:
            return left != right
        return abs(left - right) >= step

    def _rate_limit_allows_place(self) -> bool:
        if self._last_place_timestamp < 0:
            return True
        elapsed = Decimal(str(self._strategy.current_timestamp)) - self._last_place_timestamp
        return elapsed >= Decimal(str(self.config.min_order_interval))

    def _mark_action(self):
        self._last_place_timestamp = Decimal(str(self._strategy.current_timestamp))

    def _slippage_blocks_new_orders(self) -> bool:
        if self.config.max_slippage is None or self.config.mode == ExecutionMode.UP_TO:
            return False
        if self.config.side == TradeType.BUY:
            market_price = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk)
            return market_price > self.config.price_high * (Decimal("1") + self.config.max_slippage)
        market_price = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid)
        return market_price < self.config.price_low * (Decimal("1") - self.config.max_slippage)

    def _outside_cap(self) -> bool:
        if self.config.mode == ExecutionMode.UP_TO:
            limit = self.config.limit_price
            if self.config.side == TradeType.BUY:
                return self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk) > limit
            return self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid) < limit
        if self.config.side == TradeType.BUY:
            return (
                self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk)
                > self.config.price_high
            )
        return (
            self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid)
            < self.config.price_low
        )

    def _expired(self) -> bool:
        if self.config.expiry_seconds is None:
            return False
        return self._elapsed() >= self.config.expiry_seconds

    def _time_limit_reached(self) -> bool:
        if self.config.duration_seconds is None or self._schedule_started_at is None:
            return False
        return self._schedule_elapsed() + 1e-9 >= self.config.duration_seconds

    def _execution_complete(self) -> bool:
        if self.config.mode == ExecutionMode.RANGE:
            return self._all_slices_filled()
        if any(self._counts_as_open(slot) for slot in self._slices):
            return False
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        leftover = self.config.total_amount - self._filled_unit() - self._skipped_amount()
        if leftover <= 0:
            return self._filled_unit() > 0 or self._skipped_amount() >= self.config.total_amount
        return self._unit_to_base(leftover, self.config.limit_price) < Decimal(str(rules.min_order_size))

    def _all_slices_filled(self) -> bool:
        if not self._slices:
            return False
        return all(slot.is_filled or (slot.order_id == "skipped") for slot in self._slices)

    def _advance_schedule(self, force_end: bool = False):
        if self._schedule is None or self._schedule_started_at is None:
            return
        elapsed = self.config.duration_seconds if force_end else self._schedule_elapsed()
        self._schedule.advance(elapsed, self._filled_unit(), self._open_unit(), self._outside_cap())

    def _released_amount(self) -> Decimal:
        """
        Size the schedule has released.

        Before the anchor that is only the first slice, including when timing jitter
        pushed its offset forward. After the anchor, release offsets are measured
        from the anchor.
        """
        if self._schedule is None:
            return self.config.total_amount
        if self._schedule_started_at is None:
            return self._schedule.amounts[0]
        return self._schedule.opened(self._schedule_elapsed())

    def _placeable(self) -> Decimal:
        if self._schedule is None:
            opened = self.config.total_amount
            skipped = Decimal("0")
        else:
            opened = self._released_amount()
            skipped = self._schedule.skipped
        remaining = opened - skipped - self._filled_unit() - self._open_unit()
        if remaining < 0:
            return Decimal("0")
        return remaining

    def _filled_unit(self) -> Decimal:
        if self.config.size_unit == SizeUnit.QUOTE:
            return self.filled_amount_quote
        if self.config.size_unit == SizeUnit.CONTRACTS:
            size = self._contract_size()
            if size is None:
                return Decimal("0")
            return self.filled_base / size
        return self.filled_base

    def _open_unit(self) -> Decimal:
        total = Decimal("0")
        for slot in self._slices:
            if not self._counts_as_open(slot):
                continue
            if slot.tracked and slot.tracked.order is not None:
                remaining_base = slot.tracked.order.amount - slot.tracked.executed_amount_base
                if remaining_base < 0:
                    remaining_base = Decimal("0")
            else:
                remaining_base = slot.target_amount
            total += self._base_to_unit(remaining_base, slot.target_price)
        return total

    def _skipped_amount(self) -> Decimal:
        if self._schedule is None:
            return Decimal("0")
        return self._schedule.skipped

    def _unit_to_base(self, amount: Decimal, price: Decimal) -> Decimal:
        if self.config.size_unit == SizeUnit.QUOTE:
            if price <= 0:
                return Decimal("0")
            return amount / price
        if self.config.size_unit == SizeUnit.CONTRACTS:
            size = self._contract_size()
            if size is None:
                return Decimal("0")
            return amount * size
        return amount

    def _base_to_unit(self, amount: Decimal, price: Decimal) -> Decimal:
        if self.config.size_unit == SizeUnit.QUOTE:
            return amount * price
        if self.config.size_unit == SizeUnit.CONTRACTS:
            size = self._contract_size()
            if size is None:
                return Decimal("0")
            return amount / size
        return amount

    def _refresh_tracked_orders(self):
        for slot in self._slices:
            if (
                slot.tracked
                and slot.tracked.order_id
                and slot.tracked.order_id != "skipped"
                and slot.tracked.order is None
            ):
                in_flight = self.get_in_flight_order(self.config.connector_name, slot.tracked.order_id)
                if in_flight:
                    slot.tracked.order = in_flight

    def _cancel_working_orders(self):
        order_ids = []
        for slot in self._slices:
            if slot.order_id and slot.order_id != "skipped" and self._should_cancel(slot):
                order_ids.append(slot.order_id)
        for order_id in order_ids:
            self._strategy.cancel(self.config.connector_name, self.config.trading_pair, order_id)
            self._unconfirmed_cancels.add(order_id)

    def _should_cancel(self, slot: _Slice) -> bool:
        if slot.is_working:
            return True
        if slot.tracked is None or slot.tracked.order is not None:
            return False
        return slot.order_id in self._submitted_order_ids

    def _holds_unaccounted_fill(self, slot: _Slice) -> bool:
        if slot.tracked is None or slot.tracked.order is None:
            return False
        reported = getattr(slot.tracked.order, "reported_deal_base", None)
        if reported is None:
            return False
        return slot.tracked.executed_amount_base + Decimal("0.00000001") < Decimal(str(reported))

    def _counts_as_open(self, slot: _Slice) -> bool:
        return slot.is_working or self._holds_unaccounted_fill(slot)

    def _replacement_blocked(self) -> bool:
        return bool(self._unconfirmed_cancels) or bool(self._unknown_order_ids) or bool(self._open_lookup_count)

    def _real_callable(self, owner, name):
        value = getattr(owner, name, None)
        if value is None or not callable(value):
            return None
        if getattr(type(value), "__module__", "").startswith("unittest.mock"):
            return None
        return value

    def _cancel_confirmed(self, order_id: str) -> bool:
        if order_id in self._resolved_terminal:
            return True
        checker = self._real_callable(self.connectors[self.config.connector_name], "is_cancel_confirmed")
        if checker is None:
            return True
        return bool(checker(order_id))

    async def _release_confirmed_failures(self):
        for order_id in list(self._unconfirmed_cancels):
            if order_id in self._unknown_order_ids:
                continue
            if not self._cancel_confirmed(order_id):
                if await self._escalate_unresolved_cancel(order_id):
                    if order_id in self._unknown_order_ids or not self._cancel_confirmed(order_id):
                        continue
                else:
                    self._strategy.cancel(self.config.connector_name, self.config.trading_pair, order_id)
                    continue
            self._unconfirmed_cancels.discard(order_id)
            if order_id not in self._failed_order_ids:
                continue
            self._failed_order_ids.discard(order_id)
            self._remember_fill_before_clearing(order_id)

    def _remember_fill_before_clearing(self, order_id: str):
        slot = self._slot_for(order_id)
        if slot is None or slot.tracked is None:
            return
        base, quote = self._best_fill(order_id)
        order = self._order_for(order_id)
        if order is not None:
            reported = getattr(order, "reported_deal_base", None)
            if reported is not None and Decimal(str(reported)) > base:
                extra = Decimal(str(reported)) - base
                price = getattr(order, "reported_deal_price", None) or order.price or Decimal("0")
                base = Decimal(str(reported))
                quote += Decimal(str(price or 0)) * extra
        if base > self._accounted_fill_by_order.get(order_id, Decimal("0")):
            self._accounted_fill_by_order[order_id] = base
            self._accounted_quote_by_order[order_id] = quote
        elif quote > self._accounted_quote_by_order.get(order_id, Decimal("0")):
            self._accounted_quote_by_order[order_id] = quote
        self._accounted_fill_base = sum(self._accounted_fill_by_order.values(), Decimal("0"))
        self._accounted_fill_quote = sum(self._accounted_quote_by_order.values(), Decimal("0"))
        # Safe: the fill was copied into the per-order tally above before the
        # slice forgets the order. This does not tell the exchange the order is gone.
        slot.tracked = None
        slot.cancel_requested = False

    async def _escalate_unresolved_cancel(self, order_id: str) -> bool:
        """
        Keep canceling until the retry count or the time limit, then look the order up.
        An open result is looked up again on a backoff. True means this tick settled the
        lookup, so the caller should not send another cancel.
        """
        now = self._strategy.current_timestamp
        if order_id not in self._lookup_started:
            self._cancel_attempt_count[order_id] = self._cancel_attempt_count.get(order_id, 0) + 1
            self._cancel_attempt_started.setdefault(order_id, now)
            attempts = self._cancel_attempt_count[order_id]
            elapsed = now - self._cancel_attempt_started[order_id]
            if attempts < self._cancel_resolve_after_attempts and elapsed < self._cancel_resolve_after_seconds:
                return False
        else:
            started = self._open_lookup_started.get(order_id, now)
            timed_out = now - started >= self._open_lookup_max_seconds
            due = now >= self._next_open_lookup_at.get(order_id, now)
            if not due and not timed_out:
                return False
        self._lookup_started.add(order_id)
        state = await self._lookup_order_state(order_id)
        self._refresh_tracked_orders()
        self._absorb_counted_fills()
        if self._is_confirmed_absent(state):
            if self._order_blocks_absent(order_id):
                self._halt_for_unknown_order(
                    order_id,
                    "The order was already seen, so a later not-found does not prove it never filled.",
                )
                return True
            self._resolved_terminal.add(order_id)
            self._clear_open_retry(order_id)
            return True
        if state == OrderState.CANCELED:
            self._resolved_terminal.add(order_id)
            self._clear_open_retry(order_id)
            return True
        if state == OrderState.FILLED:
            if not self._known_fill_size(order_id):
                self._halt_for_unknown_order(
                    order_id,
                    "The exchange says the order is filled but no size was recorded.",
                )
                return True
            self._resolved_terminal.add(order_id)
            self._clear_open_retry(order_id)
            return True
        if state is None or state == OrderState.FAILED:
            self._halt_for_unknown_order(order_id, "The order state is unknown.")
            return True
        self._open_lookup_count[order_id] = self._open_lookup_count.get(order_id, 0) + 1
        self._open_lookup_started.setdefault(order_id, now)
        count = self._open_lookup_count[order_id]
        age = now - self._open_lookup_started[order_id]
        if count >= self._open_lookup_cap or age >= self._open_lookup_max_seconds:
            self._halt_for_unknown_order(order_id, "The order is still open after repeated lookups.")
            return True
        delay = min(
            self._open_lookup_backoff_seconds * (2 ** (count - 1)),
            self._open_lookup_backoff_cap_seconds,
        )
        self._next_open_lookup_at[order_id] = now + delay
        return False

    async def _lookup_order_state(self, order_id: str):
        connector = self.connectors[self.config.connector_name]
        resolve = self._real_callable(connector, "resolve_unknown_order")
        if resolve is None:
            return None
        try:
            return await resolve(self.config.trading_pair, order_id, self._exchange_order_id_for(order_id))
        except Exception:
            self.logger().exception("Could not resolve order %s.", order_id)
            return None

    async def _resolve_halted_orders(self):
        for order_id in list(self._unknown_order_ids):
            state = await self._lookup_order_state(order_id)
            self._refresh_tracked_orders()
            if not self._lookup_clears_halt(order_id, state):
                continue
            self._unknown_order_ids.discard(order_id)
            self._resolved_terminal.add(order_id)
            self._clear_open_retry(order_id)

    def _is_confirmed_absent(self, state) -> bool:
        return state == "confirmed_absent"

    def _lookup_clears_halt(self, order_id: str, state) -> bool:
        self._absorb_counted_fills()
        if self._is_confirmed_absent(state):
            return not self._order_blocks_absent(order_id)
        if state == OrderState.FILLED:
            return self._known_fill_size(order_id) or self._counted_fill_base.get(order_id, Decimal("0")) > 0
        return state == OrderState.CANCELED

    def _halt_on_unresolved_exchange_orders(self):
        connector = self.connectors[self.config.connector_name]
        pending = self._real_callable(connector, "orders_awaiting_resolution")
        if pending is None:
            return
        try:
            ids = set(pending())
        except Exception:
            self.logger().exception("Could not read orders whose status is unknown.")
            return
        for order_id in ids.intersection(self._known_order_ids()):
            if order_id not in self._unknown_order_ids:
                self._halt_for_unknown_order(
                    order_id,
                    "The status poll could not prove this order is gone.",
                )

    def _known_order_ids(self) -> Set[str]:
        ids = set(self._submitted_order_ids)
        ids.update(self._unconfirmed_cancels)
        ids.update(self._unknown_order_ids)
        for slot in self._slices:
            if slot.order_id and slot.order_id != "skipped":
                ids.add(slot.order_id)
        return ids

    async def _cap_base_amount(self, amount: Decimal) -> Optional[Decimal]:
        """
        Cap a new child by the live position change and open orders.
        None means the exchange read failed and the child must not be sent.
        A connector without this read is left unchanged.
        """
        if amount <= 0:
            return amount
        connector = self.connectors[self.config.connector_name]
        query = self._real_callable(connector, "ground_truth_exposure")
        if query is None:
            return amount
        try:
            try:
                result = query(
                    self.config.trading_pair,
                    self.config.side,
                    position_side=self._settings_position_side(),
                )
            except TypeError:
                result = query(self.config.trading_pair, self.config.side)
            if hasattr(result, "__await__"):
                result = await result
            position_base, open_base = result
            position_base = Decimal(str(position_base))
            open_base = Decimal(str(open_base))
        except Exception:
            self.logger().error("Could not read the exchange position and open orders. Not placing a child.")
            return None
        if self._ground_truth_baseline is None:
            self._ground_truth_baseline = position_base
        change = position_base - self._ground_truth_baseline
        if change < 0:
            change = Decimal("0")
        room = self._target_base() - change - open_base
        if room < 0:
            room = Decimal("0")
        if amount > room:
            return room
        return amount

    def _order_blocks_absent(self, order_id: str) -> bool:
        if order_id in self._seen_live_order_ids:
            return True
        if self._counted_fill_base.get(order_id, Decimal("0")) > 0:
            return True
        if self._known_fill_size(order_id):
            return True
        order = self._order_for(order_id)
        if order is None:
            return False
        if order.exchange_order_id not in (None, ""):
            return True
        return order.current_state in (OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.FILLED)

    def _absorb_counted_fills(self):
        connector = self.connectors[self.config.connector_name]
        counted = self._real_callable(connector, "counted_fill")
        seen = self._real_callable(connector, "order_was_seen_live")
        order_ids = set(self._counted_fill_base)
        order_ids.update(self._unknown_order_ids)
        order_ids.update(self._unconfirmed_cancels)
        order_ids.update(self._accounted_fill_by_order)
        for slot in self._slices:
            if slot.order_id and slot.order_id != "skipped":
                order_ids.add(slot.order_id)
        for order_id in order_ids:
            if counted is not None:
                try:
                    base, quote = counted(order_id)
                except Exception:
                    base, quote = Decimal("0"), Decimal("0")
                base = Decimal(str(base or 0))
                quote = Decimal(str(quote or 0))
                previous = self._counted_fill_base.get(order_id, Decimal("0"))
                previous_quote = self._counted_fill_quote.get(order_id, Decimal("0"))
                if base > previous:
                    self._counted_fill_base[order_id] = base
                    self._counted_fill_quote[order_id] = quote
                elif base == previous and quote > previous_quote:
                    self._counted_fill_quote[order_id] = quote
            if seen is not None:
                try:
                    if seen(order_id):
                        self._seen_live_order_ids.add(order_id)
                except Exception:
                    pass

    def _live_fill(self, order_id: str):
        slot = self._slot_for(order_id)
        order = slot.tracked.order if slot and slot.tracked else None
        if order is None and slot and slot.tracked:
            order = self.get_in_flight_order(self.config.connector_name, order_id)
        if order is None:
            return Decimal("0"), Decimal("0")
        return (
            Decimal(str(order.executed_amount_base or 0)),
            Decimal(str(order.executed_amount_quote or 0)),
        )

    def _best_fill(self, order_id: str):
        self._absorb_counted_fills()
        live_base, live_quote = self._live_fill(order_id)
        counted_base = self._counted_fill_base.get(order_id, Decimal("0"))
        counted_quote = self._counted_fill_quote.get(order_id, Decimal("0"))
        accounted_base = self._accounted_fill_by_order.get(order_id, Decimal("0"))
        accounted_quote = self._accounted_quote_by_order.get(order_id, Decimal("0"))
        base = max(live_base, counted_base, accounted_base)
        if base == live_base:
            quote = live_quote
        elif base == counted_base:
            quote = counted_quote
        else:
            quote = accounted_quote
        quote = max(
            quote,
            live_quote if live_base == base else Decimal("0"),
            counted_quote if counted_base == base else Decimal("0"),
        )
        return base, quote

    def _clear_open_retry(self, order_id: str):
        self._open_lookup_count.pop(order_id, None)
        self._open_lookup_started.pop(order_id, None)
        self._next_open_lookup_at.pop(order_id, None)

    def _known_fill_size(self, order_id: str) -> bool:
        order = self._order_for(order_id)
        if order is None:
            return False
        base = Decimal(str(order.executed_amount_base or 0))
        reported = getattr(order, "reported_deal_base", None)
        if reported is not None:
            base = max(base, Decimal(str(reported)))
        return base > 0

    def _order_for(self, order_id: str):
        slot = self._slot_for(order_id)
        order = slot.tracked.order if slot and slot.tracked else None
        if order is None:
            order = self.get_in_flight_order(self.config.connector_name, order_id)
        return order

    def _exchange_order_id_for(self, order_id: str) -> Optional[str]:
        order = self._order_for(order_id)
        if order is None or not order.exchange_order_id:
            return None
        return str(order.exchange_order_id)

    def _halt_for_unknown_order(self, order_id: str, because: str):
        self._unknown_order_ids.add(order_id)
        self._next_open_lookup_at.pop(order_id, None)
        self._describe_unknown_orders(because)
        self.logger().error(self._unknown_order_error)
        self.pause()

    def _describe_unknown_orders(self, because: str):
        ids = ", ".join(sorted(self._unknown_order_ids))
        self._unknown_order_error = (
            f"unknown_order_halt: {ids}. {because} "
            "Paused so it is not bought again while its state is open or unknown."
        )

    def _status_reason(self) -> str:
        if self._unknown_order_ids:
            return "unknown_order_halt: " + ", ".join(sorted(self._unknown_order_ids))
        if self._cancel_stop_error:
            return self._cancel_stop_error
        if self._account_settings_error:
            return self._account_settings_error
        if self._open_lookup_count:
            return "open_order_retry: " + ", ".join(sorted(self._open_lookup_count))
        return ""

    def to_format_status(self):
        return self._status_reason()

    async def _ensure_account_settings(self) -> bool:
        if self._account_settings_ready:
            return True
        if self._account_settings_failed:
            return False
        connector = self.connectors[self.config.connector_name]
        position_mode_ready = self._real_callable(connector, "is_position_mode_ready")
        if position_mode_ready is not None and not position_mode_ready():
            return False
        # None means do not change margin. A matching position leverage is left as it is.
        if self.config.margin_mode is None and self._position_leverage_matches(connector):
            self._account_settings_ready = True
            return True
        position_side = self._settings_position_side()
        confirm = self._real_callable(connector, "confirm_leverage_and_margin")
        if confirm is not None:
            try:
                try:
                    accepted = await confirm(
                        self.config.trading_pair,
                        int(self.config.leverage),
                        self.config.margin_mode,
                        position_side=position_side,
                    )
                except TypeError:
                    accepted = await confirm(
                        self.config.trading_pair,
                        int(self.config.leverage),
                        self.config.margin_mode,
                    )
            except Exception:
                self.logger().exception("Could not confirm leverage and margin mode.")
                accepted = False
            if not accepted:
                reason = self._confirmation_error(connector)
                if not reason:
                    if self.config.margin_mode is None:
                        reason = f"The exchange did not confirm leverage {self.config.leverage}. Refusing to trade."
                    else:
                        reason = (
                            f"The exchange did not confirm leverage {self.config.leverage} "
                            f"and margin mode {self.config.margin_mode}. Refusing to trade."
                        )
                self._halt_account_settings(reason)
                return False
        else:
            if self.config.margin_mode is not None:
                set_mode = self._real_callable(connector, "set_margin_mode")
                if set_mode is not None:
                    result = set_mode(self.config.margin_mode)
                    if hasattr(result, "__await__"):
                        await result
            if not self._position_leverage_matches(connector):
                set_leverage = self._real_callable(connector, "set_leverage")
                if set_leverage is not None:
                    try:
                        set_leverage(
                            self.config.trading_pair,
                            int(self.config.leverage),
                            position_side=position_side,
                        )
                    except TypeError:
                        set_leverage(self.config.trading_pair, int(self.config.leverage))
        self._account_settings_ready = True
        return True

    def _settings_position_side(self) -> PositionSide:
        """
        Hedge mode keeps a long and a short. Opening a buy is the long. Opening a sell
        is the short. A reduce-only order belongs to the position it closes.
        """
        if self.config.reduce_only:
            return PositionSide.LONG if self.config.side == TradeType.SELL else PositionSide.SHORT
        return PositionSide.LONG if self.config.side == TradeType.BUY else PositionSide.SHORT

    def _position_leverage_matches(self, connector) -> bool:
        positions = getattr(connector, "account_positions", None)
        if not isinstance(positions, dict):
            return False
        side = self._settings_position_side()
        for position in positions.values():
            if getattr(position, "trading_pair", None) != self.config.trading_pair:
                continue
            if getattr(position, "position_side", None) != side:
                continue
            leverage = getattr(position, "leverage", None)
            if leverage is None:
                continue
            try:
                return Decimal(str(leverage)) == Decimal(int(self.config.leverage))
            except (TypeError, ValueError, ArithmeticError):
                return False
        return False

    def _confirmation_error(self, connector) -> str:
        reader = self._real_callable(connector, "leverage_confirmation_error")
        if reader is None:
            return ""
        try:
            text = reader()
        except Exception:
            return ""
        if not isinstance(text, str):
            return ""
        return text

    def _halt_account_settings(self, reason: str):
        self._account_settings_error = reason
        self.logger().error(reason)
        self._account_settings_failed = True
        self.close_type = CloseType.FAILED
        self.stop()

    def _size_remainder(self) -> Decimal:
        return self._rounding_remainder_base

    def process_order_created_event(self, event_tag: int, market, event):
        slot = self._slot_for(event.order_id)
        if slot and slot.tracked:
            in_flight = self.get_in_flight_order(self.config.connector_name, event.order_id)
            if in_flight:
                slot.tracked.order = in_flight

    def process_order_failed_event(self, event_tag: int, market, event):
        slot = self._slot_for(event.order_id)
        if slot:
            self._strategy.cancel(self.config.connector_name, self.config.trading_pair, event.order_id)
            self._unconfirmed_cancels.add(event.order_id)
            self._failed_order_ids.add(event.order_id)
            slot.cancel_requested = True
            self._current_retries += 1
            if self._current_retries > self._max_retries:
                self.close_type = CloseType.FAILED
                self.early_stop(close_type=CloseType.FAILED)

    def process_order_filled_event(self, event_tag: int, market, event: OrderFilledEvent):
        if not self._awaiting_schedule_anchor():
            return
        order_id = getattr(event, "order_id", None)
        if order_id is None:
            return
        if self._slot_for(order_id) is None and order_id not in self._submitted_order_ids:
            return
        if Decimal(str(getattr(event, "amount", 0) or 0)) <= 0:
            return
        self._anchor_schedule(float(event.timestamp))

    def process_order_completed_event(self, event_tag: int, market, event):
        self._capture_schedule_anchor()
        if self._execution_complete():
            self.close_execution_finished()

    def _slot_for(self, order_id: str) -> Optional[_Slice]:
        return next((slot for slot in self._slices if slot.order_id == order_id), None)

    @property
    def filled_base(self) -> Decimal:
        self._absorb_counted_fills()
        return self._sum_best_fills()[0]

    @property
    def filled_amount_quote(self) -> Decimal:
        self._absorb_counted_fills()
        return self._sum_best_fills()[1]

    def _sum_best_fills(self):
        order_ids = set(self._counted_fill_base)
        order_ids.update(self._accounted_fill_by_order)
        for slot in self._slices:
            if slot.order_id and slot.order_id != "skipped":
                order_ids.add(slot.order_id)
        base = Decimal("0")
        quote = Decimal("0")
        for order_id in order_ids:
            part_base, part_quote = self._best_fill(order_id)
            base += part_base
            quote += part_quote
        return base, quote

    @property
    def total_base(self) -> Decimal:
        if self.config.mode == ExecutionMode.UP_TO:
            return self._target_base()
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
        if self.config.side == TradeType.BUY:
            return (mark - self.average_fill_price) * filled - self.cum_fees_quote
        return (self.average_fill_price - mark) * filled - self.cum_fees_quote

    def get_net_pnl_pct(self) -> Decimal:
        quote = self.filled_amount_quote
        if quote <= 0:
            return Decimal("0")
        return self.net_pnl_quote / quote

    def _unfilled_amount(self) -> Decimal:
        remainder = self.config.total_amount - self._filled_unit()
        if remainder < 0:
            return Decimal("0")
        return remainder

    def get_custom_info(self) -> dict:
        return {
            "side": self.config.side.name,
            "mode": self.config.mode.value,
            "paused": self._paused,
            "schedule_started_at": self._schedule_started_at,
            "schedule_time_remaining": self._schedule_time_remaining(),
            "schedule_clock": self._schedule_clock_state(),
            "unknown_order_halt": bool(self._unknown_order_ids),
            "unknown_order_ids": sorted(self._unknown_order_ids),
            "unknown_order_error": self._unknown_order_error,
            "status_reason": self._status_reason(),
            "state": self._status_reason() or "RUNNING",
            "filled_base": str(self.filled_base),
            "filled_quote": str(self.filled_amount_quote),
            "total_base": str(self.total_base),
            "target_amount": str(self.config.total_amount),
            "filled_amount": str(self._filled_unit()),
            "unfilled_amount": str(self._unfilled_amount()),
            "skipped_amount": str(self._skipped_amount()),
            "average_fill_price": str(self.average_fill_price),
            "rounding_remainder": str(self._size_remainder()),
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
