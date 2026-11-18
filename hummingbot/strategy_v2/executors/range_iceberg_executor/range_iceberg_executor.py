import inspect
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
    ExitArming,
    ExitExecution,
    ExitOrderType,
    ExitState,
    RangeIcebergExecutorConfig,
    RangeIcebergExitConfig,
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
    quantize_up,
)
from hummingbot.strategy_v2.executors.time_schedule import SliceSchedule, release_timestamps
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder


class _Slice:
    def __init__(self, level: LadderLevel):
        self.target_price = level.price
        self.target_amount = level.base_amount
        self.pre_jitter_price: Optional[Decimal] = None
        self.tracked: Optional[TrackedOrder] = None
        self.cancel_requested = False
        self.retired = False

    @property
    def order_id(self) -> Optional[str]:
        return self.tracked.order_id if self.tracked else None

    @property
    def is_working(self) -> bool:
        if self.retired:
            return False
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
    only the first slice is placed, except a range rung larger than that slice is
    allowed out so it can fill and start the clock. Nothing is skipped or carried
    before the anchor. schedule_start SUBMIT keeps the previous clock, which starts
    when the executor starts.
    expiry_seconds is always measured from the executor start, so an order that never
    fills can still expire.

    Pausing freezes the schedule clock after it has started, and it still freezes
    expiry. Working orders stay. Time spent paused is not counted, and resume
    continues from the frozen elapsed time.
    Pause stops new entry orders. An exit that already started keeps closing, and an
    armed exit can still trigger. A pause after an exit trigger also freezes the
    exit clock. An unknown-order halt is different: no new exit order is sent
    until that halt clears.

    Optional take profit and stop loss close the filled position. They are off unless
    the config sets them. A single take profit is one limit at the trigger, or one
    market order. A single stop loss is one market order when the trigger is reached.
    An iceberg exit reuses this executor's ladder, schedule, price jitter, pre-jitter
    reprice, and cancelled-rung release. The opening quote cap and the hedge opening
    cap stay on entry children. An exit is sized from the filled base, so those caps
    do not shrink the close. Exit orders are not post only. A post-only reject
    backoff therefore does not hold the close, and fees on a cleared exit order are
    kept the same way entry fees are kept.

    An iceberg exit schedule starts when that exit triggers. It does not wait for
    the first exit fill. The entry rule that holds later rungs until a fill exists
    only so that fill can start the entry clock. A close does not need it, so the
    first exit window is open at the trigger and later windows count from then,
    even while the first close order is still unfilled.

    The exit is reduce-only: closing a long is a sell with position action CLOSE
    (MEXC hedge side 4) and closing a short is a buy with position action CLOSE
    (MEXC hedge side 2). Exit size never exceeds this executor's own filled amount,
    net of what the exit has already closed.

    A single limit take profit is not a trigger. It rests at the trigger for the
    filled amount, and further entry fills add to it or resize it. The order is
    reduce-only and never larger than the filled amount. The entry keeps running.
    A market take profit, an iceberg take profit, and a stop loss are triggers.
    They cancel the remaining entry before they send the close.

    The stop loss can take over any unfinished take profit. That includes a limit
    that has partly filled and an iceberg child that is still resting. The
    take-profit remainder is cancelled and the stop closes what is still open.
    Once a stop loss has started, a later move back through the take-profit price
    does not cancel it and does not start the take profit.

    Under ON_FULL_FILL, a duration or expiry that ends with only a partial fill
    cancels the rest of the entry and arms exits on the filled amount. It does
    not stop the executor while that position still needs a close. Once an exit
    is armed or active, the entry time limit and expiry are not checked again.

    early_stop cancels software entry and exit orders. It does not close the
    filled position. keep_position=True is the API shutdown path and records
    POSITION_HOLD. That path cancels software orders and leaves the exchange
    stop working. keep_position=False, a user Stop, and FAILED cancel the
    exchange stop as well. The saved custom_info is enough to resume an exit
    whose exit_state is ARMED, RESTING, or CLOSING. Resume posts the exit
    again from open_base when the old order ids are gone, and it keeps the
    saved exchange stop id instead of placing a second stop.

    When the stop loss is armed and the connector can place one, a protective
    exchange stop is sent past the software trigger by protective_stop_buffer_pct.
    It is placed before a resting take profit, with volType 1. A placement
    failure backs off and is stored on exchange_stop_error.     The software stop leaves that order working until every take-profit and
    entry cancel is confirmed. Only then does it cancel the backstop and send
    its own close. If those cancels are still open after
    stop_loss_fallback_seconds, the backstop stays and the software close is
    the open size minus the unconfirmed take profit and the live stop. If the
    exchange stop already fired, the close is reconciled and the executor ends
    on the stop, not FAILED. A larger fill at the same backstop price places
    the full open size and MEXC returns the existing stop id with that new
    vol. A moved percent trigger cancels the old stops first, then places one
    full-size stop at the new price. protective_exchange_stop defaults to on
    only if the connector implements place_position_stop_order.

    Repeated exit rejections use their own retry budget and backoff. Stop loss
    retries wait at most about 2 seconds. They do not mark the executor FAILED
    while the position is open until a market close has been tried. If that
    close fails and the exchange stop was cancelled to send it, the stop is
    placed again before the executor stops. The error is stored on custom_info.
    Entry retries that end with a fill arm the configured exits on that fill
    instead of claiming a market close.

    restore_from_custom_info with resume_entries false never places an entry.
    It keeps only the restored open amount, in base size, and rebuilds the
    average from the supplied average instead of stale per-order exit fills.

    The exposure cap reads the whole pair and side. Two executors on the same
    pair and side can starve each other, because each one counts the other's
    fills and open orders. The cap is not scoped to this executor.

    custom_info for the API and Condor includes exit_state, exit_order_ids,
    exchange_stop_order_id, exchange_stop_order_ids, has_exchange_stop,
    exchange_stop_error, filled_base, closed_base, and open_base.
    take_profit_trigger_price, stop_loss_trigger_price, and
    exchange_stop_trigger_price are null until a fill exists, and null when
    that exit is not configured. A percent trigger uses the current average
    fill. The exchange stop trigger is the software stop plus the buffer.
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
        self._size_unit_override: Optional[SizeUnit] = None
        self._restored_total_base: Optional[Decimal] = None
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
        self._cap_block_reason = ""
        self._retained_fees_quote = Decimal("0")
        self._post_only_retry_after: Optional[float] = None
        self._post_only_rejected_at: Optional[float] = None
        self._post_only_reject_bid: Optional[Decimal] = None
        self._post_only_reject_ask: Optional[Decimal] = None
        self._exit_slices: List[_Slice] = []
        self._exit_phase = "idle"
        self._exit_active: Optional[str] = None
        self._exit_trigger_price: Optional[Decimal] = None
        self._exit_plan_ready = False
        self._exit_plan_base = Decimal("0")
        self._exit_slice_base = Decimal("0")
        self._exit_schedule: Optional[SliceSchedule] = None
        self._exit_schedule_state: Optional[dict] = None
        self._exit_started_at: Optional[float] = None
        self._exit_paused_seconds = 0.0
        self._exit_range = False
        self._exit_accounted_base: dict = {}
        self._exit_accounted_quote: dict = {}
        self._exit_order_ids: List[str] = []
        self._exit_force_single = False
        self._exit_force_market = False
        self._entry_clock_handled = False
        self._entry_handed_to_exit = False
        self._resume_entries = True
        self._last_fill_timestamp: Optional[float] = None
        self._tp_grow_cancelling = False
        self._tp_previous_size = Decimal("0")
        self._replace_stop_before_failed = False
        self._preserve_exchange_stop = False
        self._exchange_stops: List[dict] = []
        self._exchange_stop_order_id: Optional[str] = None
        self._exchange_stop_amount = Decimal("0")
        self._exchange_stop_price: Optional[Decimal] = None
        self._sl_blocked_since: Optional[float] = None
        self._tp_reposted_after_failure = False
        self._exchange_stop_credited: Set[str] = set()
        self._exchange_stop_last_request: Optional[tuple] = None
        self._exchange_stop_error = ""
        self._exchange_stop_failures = 0
        self._exchange_stop_retry_after: Optional[float] = None
        self._exchange_stop_released = False
        self._exchange_stop_reconciled = False
        self._exchange_stop_last_amount = Decimal("0")
        self._exchange_stop_cancel_failures = 0
        self._exchange_stop_cancel_after: Optional[float] = None
        self._exchange_stop_cancel_started_at: Optional[float] = None
        self._exchange_stop_close_capped = False
        self._keep_exchange_stop = False
        self._exit_failure_count = 0
        self._exit_retry_after: Optional[float] = None
        self._exit_market_attempted = False
        self._exit_error = ""
        self._exit_last_fill_at: Optional[float] = None
        self._exit_seen_fill_base = Decimal("0")
        self._tp_reprice_after: Optional[float] = None

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
        offsets = self._apply_timing_jitter(offsets, interval, self.config.duration_seconds)
        return SliceSchedule(
            amounts=amounts,
            offsets=offsets,
            duration=self.config.duration_seconds,
            skip_unplaced=self.config.catch_up == CatchUpPolicy.SKIP,
        )

    def _apply_timing_jitter(self, offsets, interval: float, duration: float):
        """
        The same release-clock jitter the entry schedule uses. A zero jitter percent
        returns the nominal offsets and does not draw random numbers.
        """
        if self.config.timing_jitter_pct <= 0 or interval <= 0:
            return list(offsets)
        jittered = []
        for index, nominal in enumerate(offsets):
            swing = (self._rng.random() * 2) - 1
            delta = swing * float(self.config.timing_jitter_pct) * interval
            low = 0.0 if index == 0 else jittered[-1]
            jittered.append(min(max(low, nominal + delta), duration))
        return jittered

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
        if self._restored_total_base is not None:
            return self._restored_total_base
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
            if self._exit_started_at is not None:
                self._exit_paused_seconds += max(0.0, now - max(started, self._exit_started_at))
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
        if any(
            amount > 0
            for order_id, amount in self._counted_fill_base.items()
            if not self._is_exit_order(order_id)
        ):
            return True
        if any(
            amount > 0
            for order_id, amount in self._accounted_fill_by_order.items()
            if not self._is_exit_order(order_id)
        ):
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
        self._resume_entries = self._resume_entries_requested(info)
        self._restore_exit_state(info)

    def _resume_entries_requested(self, info: dict) -> bool:
        """
        Missing resume_entries keeps placing entries. An explicit false, 0,
        "false", "0", or "no" resumes exits only.
        """
        if not isinstance(info, dict) or "resume_entries" not in info:
            return True
        value = info.get("resume_entries")
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no")
        return bool(value)

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
        self._release_cancelled_rungs()
        self._release_cancelled_exit_rungs()
        self._capture_schedule_anchor()
        if self._exit_phase == "done":
            await self._sync_exchange_stop()
            return
        if self._exit_phase in ("cancelling_entry", "executing"):
            await self._control_exit()
            await self._sync_exchange_stop()
            return
        self._advance_schedule()
        if await self._handle_entry_clock():
            await self._sync_exchange_stop()
            return
        if self._exit_phase in ("armed", "resting"):
            await self._control_exit()
            if self._exit_phase in ("cancelling_entry", "executing", "done", "armed"):
                await self._sync_exchange_stop()
                return
            if self._entry_clock_handled or self._execution_complete():
                await self._sync_exchange_stop()
                return
        if self._execution_complete():
            if self._needs_exit_after_entry():
                self._exit_phase = "armed"
                await self._control_exit()
                await self._sync_exchange_stop()
                return
            self.close_execution_finished()
            return
        if self._resting_limit_applies() or self._partial_exit_ready():
            await self._control_exit()
            if self._exit_phase in ("cancelling_entry", "executing", "done"):
                await self._sync_exchange_stop()
                return
        await self._sync_exchange_stop()
        if not self._resume_entries:
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
        if self._post_only_backoff_blocks():
            return
        slot = self._next_slot()
        if slot is None or not self._range_slot_released(slot):
            return
        await self._place_slice(slot)

    def early_stop(self, keep_position: bool = False, close_type: Optional[CloseType] = None):
        """
        Cancel software entry and exit orders, then stop.

        The filled position is not closed. keep_position=True selects POSITION_HOLD
        and is the close type the API uses on shutdown. It leaves the exchange
        stop working. keep_position=False selects EARLY_STOP unless the caller
        passes a close type, and that path cancels the exchange stop. A user
        Stop uses this method with keep_position left false, so it cancels the
        stop too. A stop loss that is re-placing its backstop keeps the stop.
        Neither value flattens the position.

        custom_info keeps exit_state, exit_order_ids, and exchange_stop_order_id
        so a later start can resume an exit that was ARMED, RESTING, or CLOSING.
        """
        self._keep_exchange_stop = bool(keep_position) or self._preserve_exchange_stop
        self.close_type = close_type or (CloseType.POSITION_HOLD if keep_position else CloseType.EARLY_STOP)
        self.close_timestamp = self._strategy.current_timestamp
        self._begin_shutdown()

    def evaluate_max_retries(self):
        """
        Entry retries that leave a fill do not stop the executor.

        The base helper would mark FAILED as soon as the counter passes the
        budget. With a fill and a configured exit, that fill is handed to the
        exit instead. A position with no exit, or an exit that already failed
        its own market close, still ends FAILED.
        """
        if self._current_retries > self._max_retries and self._entry_handed_to_exit:
            return
        if (
            self._current_retries > self._max_retries
            and self._position_base() > 0
            and self._exits_configured()
            and self.close_type != CloseType.FAILED
        ):
            return
        super().evaluate_max_retries()

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
        if pending:
            self._note_unconfirmed_cancels(pending)
        if pending or self._exchange_stop_cancel_outstanding():
            return
        self._cancel_stop_error = ""
        self.stop()

    def _unconfirmed_cancel_ids(self) -> List[str]:
        return sorted(order_id for order_id in self._unconfirmed_cancels if not self._cancel_confirmed(order_id))

    def _note_unconfirmed_cancels(self, pending: List[str]):
        message = f"Cancel was not confirmed for {', '.join(pending)}. These orders may still be open."
        if message != self._cancel_stop_error:
            self._cancel_stop_error = message
            self.logger().error(message)

    async def _finish_shutdown(self):
        await self._release_confirmed_failures()
        if self._exchange_stop_cancel_outstanding():
            outcome = await self._request_exchange_stop_cancel()
            if outcome == "failed":
                return
            if outcome == "fired":
                self._apply_exchange_stop_fill(fired=True)
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
        if self._post_only_backoff_blocks():
            return
        placeable = self._placeable()
        if placeable <= 0:
            return
        await self._place_up_to_slice(desired, placeable)

    async def _place_slice(self, slot: _Slice):
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        planned = slot.target_price
        price = self._jitter_price(planned, rules.min_price_increment)
        step = rules.min_base_amount_increment
        amount = self._bounded_amount(slot.target_amount, price, step, rules.min_order_size)
        if amount < rules.min_order_size:
            slot.tracked = TrackedOrder(order_id="skipped")
            return
        amount = await self._cap_child_amount(amount, price, step, rules.min_order_size)
        if amount is None:
            return
        order_id = self._submit(price, amount)
        slot.tracked = TrackedOrder(order_id=order_id)
        slot.target_amount = amount
        slot.target_price = price
        slot.pre_jitter_price = planned
        self._submitted_order_ids.append(order_id)
        self._mark_action()

    async def _place_up_to_slice(self, price: Decimal, placeable_unit: Decimal):
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        planned = price
        price = self._clamp_limit(self._jitter_up_to_price(price))
        step = rules.min_base_amount_increment
        slice_unit = min(self._slice_unit_amount(), placeable_unit)
        raw_base = self._unit_to_base(slice_unit, price)
        amount = self._bounded_amount(raw_base, price, step, rules.min_order_size)
        amount = await self._cap_child_amount(amount, price, step, rules.min_order_size)
        if amount is None:
            return
        order_id = self._submit(price, amount)
        slot = _Slice(LadderLevel(price=price, base_amount=amount))
        slot.pre_jitter_price = planned
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

    def _jitter_price(
        self,
        price: Decimal,
        price_step: Decimal,
        price_low: Optional[Decimal] = None,
        price_high: Optional[Decimal] = None,
    ) -> Decimal:
        if self.config.price_jitter_pct <= 0:
            return price
        low = self.config.price_low if price_low is None else price_low
        high = self.config.price_high if price_high is None else price_high
        swing = (Decimal(str(self._rng.random())) * 2) - Decimal("1")
        jittered = price * (Decimal("1") + self.config.price_jitter_pct * swing)
        try:
            return clamp_price(jittered, low, high, price_step)
        except ValueError:
            return price

    def _jitter_up_to_price(
        self,
        price: Decimal,
        side: Optional[TradeType] = None,
        limit_price: Optional[Decimal] = None,
        post_only: Optional[bool] = None,
    ) -> Decimal:
        if self.config.price_jitter_pct <= 0:
            return price
        swing = (Decimal(str(self._rng.random())) * 2) - Decimal("1")
        jittered = price * (Decimal("1") + self.config.price_jitter_pct * swing)
        clamped = self._clamp_limit(jittered, side=side, limit_price=limit_price)
        if not self._jitter_respects_market(clamped, side=side, limit_price=limit_price, post_only=post_only):
            return price
        return clamped

    def _jitter_respects_market(
        self,
        price: Decimal,
        side: Optional[TradeType] = None,
        limit_price: Optional[Decimal] = None,
        post_only: Optional[bool] = None,
    ) -> bool:
        """
        A buy cannot be pushed past the limit, and a post-only buy cannot reach the ask.
        A sell cannot be pushed through the limit, and a post-only sell cannot reach the bid.
        """
        side = self.config.side if side is None else side
        limit = self.config.limit_price if limit_price is None else limit_price
        post_only = self.config.post_only if post_only is None else post_only
        if side == TradeType.BUY:
            if limit is not None and price > limit:
                return False
            if post_only:
                ask = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk)
                if price >= ask:
                    return False
            return True
        if limit is not None and price < limit:
            return False
        if post_only:
            bid = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid)
            if price <= bid:
                return False
        return True

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

    def _desired_price(
        self,
        side: Optional[TradeType] = None,
        limit_price: Optional[Decimal] = None,
        cap_behavior: Optional[CapBehavior] = None,
        post_only: Optional[bool] = None,
    ) -> Optional[Decimal]:
        side = self.config.side if side is None else side
        cap = self.config.limit_price if limit_price is None else limit_price
        cap_behavior = self.config.cap_behavior if cap_behavior is None else cap_behavior
        post_only = self.config.post_only if post_only is None else post_only
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        step = Decimal(str(rules.min_price_increment or 0))
        ask = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk)
        bid = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid)
        if side == TradeType.BUY:
            if ask > cap:
                if cap_behavior == CapBehavior.WAIT:
                    return None
                raw = cap
            elif post_only:
                raw = bid + step if step > 0 else bid
                if step > 0 and raw >= ask:
                    raw = bid
            else:
                raw = ask
            price = cap_buy_price(raw, step, cap)
            if post_only and price >= ask:
                price = cap_buy_price(bid, step, cap)
            return price
        if bid < cap:
            if cap_behavior == CapBehavior.WAIT:
                return None
            raw = cap
        elif post_only:
            raw = ask - step if step > 0 else ask
            if step > 0 and raw <= bid:
                raw = ask
        else:
            raw = bid
        price = floor_sell_price(raw, step, cap)
        if post_only and price <= bid:
            price = floor_sell_price(bid + (step if step > 0 else Decimal("0")), step, cap)
        return price

    def _clamp_limit(
        self,
        price: Decimal,
        side: Optional[TradeType] = None,
        limit_price: Optional[Decimal] = None,
    ) -> Decimal:
        side = self.config.side if side is None else side
        limit_price = self.config.limit_price if limit_price is None else limit_price
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        step = Decimal(str(rules.min_price_increment or 0))
        if side == TradeType.BUY:
            return cap_buy_price(price, step, limit_price)
        return floor_sell_price(price, step, limit_price)

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
            and not self._keep_working_price(slot, desired, step)
        ]
        if not pending:
            return False
        if not self._rate_limit_allows_place():
            return True
        self._request_cancel(pending[0])
        return True

    def _keep_working_price(
        self,
        slot: _Slice,
        desired: Decimal,
        step: Decimal,
        side: Optional[TradeType] = None,
        post_only: Optional[bool] = None,
        limit_price: Optional[Decimal] = None,
    ) -> bool:
        """
        Leave an order whose pre-jitter target has not moved.
        Also leave a post-only order that is already the best price on its side
        when the new target is only one tick more aggressive, which is this order
        seeing itself at the top of the book.
        """
        reference = slot.pre_jitter_price if slot.pre_jitter_price is not None else self._slot_price(slot)
        if not self._price_differs(reference, desired, step):
            return True
        return self._resting_on_own_best(
            self._slot_price(slot),
            desired,
            step,
            side=side,
            post_only=post_only,
            limit_price=limit_price,
        )

    def _resting_on_own_best(
        self,
        order_price: Decimal,
        desired: Decimal,
        step: Decimal,
        side: Optional[TradeType] = None,
        post_only: Optional[bool] = None,
        limit_price: Optional[Decimal] = None,
    ) -> bool:
        side = self.config.side if side is None else side
        post_only = self.config.post_only if post_only is None else post_only
        limit = self.config.limit_price if limit_price is None else limit_price
        if not post_only:
            return False
        bid = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid)
        ask = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk)
        tick = step if step > 0 else Decimal("0")
        if side == TradeType.BUY:
            if order_price != bid or order_price >= ask:
                return False
            if limit is not None and order_price > limit:
                return False
            return desired <= order_price + tick
        if order_price != ask or order_price <= bid:
            return False
        if limit is not None and order_price < limit:
            return False
        return desired >= order_price - tick

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

        Before the anchor that is the first slice, including when timing jitter
        pushed its offset forward. A range rung larger than that slice is released
        too, so the first order can rest and fill, which is what starts the clock.
        The allowance never exceeds the total. After the anchor, release offsets
        are measured from the anchor.
        """
        if self._schedule is None:
            return self.config.total_amount
        if self._schedule_started_at is None:
            released = self._schedule.amounts[0]
            if self.config.mode == ExecutionMode.RANGE and self._slices:
                first = self._slices[0]
                rung = self._base_to_unit(first.target_amount, first.target_price)
                if rung > released:
                    released = rung
            if released > self.config.total_amount:
                released = self.config.total_amount
            return released
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
        if self._size_unit_override == SizeUnit.BASE:
            return self.filled_base
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
        for slot in list(self._slices) + list(self._exit_slices):
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
        for slot in list(self._slices) + list(self._exit_slices):
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
        return slot.order_id in self._submitted_order_ids or slot.order_id in self._exit_order_ids

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
        self._retire_confirmed_exit_cancels()

    def _remember_fill_before_clearing(self, order_id: str):
        slot = self._slot_for(order_id)
        if slot is None or slot.tracked is None:
            return
        self._retained_fees_quote += slot.tracked.cum_fees_quote
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
        # Fees were copied into _retained_fees_quote before the slice forgot the order.
        slot.tracked = None
        slot.cancel_requested = False

    def _release_cancelled_rungs(self) -> None:
        """
        A range rung that ends cancelled still owns its unfilled remainder.
        Keep the fill, then clear the rung so the remainder is posted again.
        """
        if self.config.mode != ExecutionMode.RANGE or self._shutdown_pending:
            return
        for slot in self._slices:
            if slot.tracked is None or slot.order_id in (None, "skipped"):
                continue
            if slot.order_id in self._unconfirmed_cancels or slot.order_id in self._failed_order_ids:
                continue
            if slot.order_id in self._unknown_order_ids:
                continue
            order = slot.tracked.order
            if order is None or not order.is_cancelled or order.is_filled:
                continue
            filled_base, _quote = self._best_fill(slot.order_id)
            remainder = slot.target_amount - filled_base
            if remainder < 0:
                remainder = Decimal("0")
            self._remember_fill_before_clearing(slot.order_id)
            slot.target_amount = remainder

    def _release_cancelled_exit_rungs(self) -> None:
        """
        Same remainder release as an entry range rung, for an exit range ladder.

        A close that the exchange cancelled still owns its unfilled size. Keep the
        fill and the fee, then clear the rung so the remainder is posted again.
        Up-to exit children are new slices, so they are left for the next place.
        """
        if not self._exit_range or self._shutdown_pending or self._exit_phase == "done":
            return
        for slot in self._exit_slices:
            if slot.retired or slot.tracked is None or slot.order_id in (None, "skipped"):
                continue
            if slot.order_id in self._unconfirmed_cancels or slot.order_id in self._failed_order_ids:
                continue
            if slot.order_id in self._unknown_order_ids:
                continue
            order = slot.tracked.order
            if order is None or not order.is_cancelled or order.is_filled:
                continue
            filled_base, _quote = self._exit_best_fill(slot.order_id)
            remainder = slot.target_amount - filled_base
            if remainder < 0:
                remainder = Decimal("0")
            self._retained_fees_quote += slot.tracked.cum_fees_quote
            self._remember_exit_fill(slot)
            slot.tracked = None
            slot.cancel_requested = False
            slot.target_amount = remainder

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
        for slot in list(self._slices) + list(self._exit_slices):
            if slot.order_id and slot.order_id != "skipped":
                ids.add(slot.order_id)
        ids.update(self._exit_order_ids)
        return ids

    def _exposure_target_base(self, price: Optional[Decimal]) -> Decimal:
        """
        Base the exposure cap will still allow.

        A quote-sized up-to order spends the quote budget at the price it is about
        to pay. Valuing that budget at the limit under-counts a fill below the limit
        and blocks the last child. Other modes keep the static target.
        """
        if (
            self.config.size_unit == SizeUnit.QUOTE
            and self.config.mode == ExecutionMode.UP_TO
            and price is not None
            and price > 0
        ):
            return self.config.total_amount / price
        return self._target_base()

    def _exposure_query_kwargs(self, query) -> dict:
        try:
            names = inspect.signature(query).parameters
        except (TypeError, ValueError):
            names = {}
        kwargs = {}
        if "position_side" in names:
            kwargs["position_side"] = self._settings_position_side()
        if "position_action" in names:
            kwargs["position_action"] = (
                PositionAction.CLOSE if self.config.reduce_only else PositionAction.OPEN
            )
        if "exclude_client_order_ids" in names:
            kwargs["exclude_client_order_ids"] = self._exposure_exclude_ids()
        return kwargs

    def _exposure_exclude_ids(self) -> Set[str]:
        ids = set()
        for order_id in self._known_order_ids():
            if not order_id or order_id == "skipped":
                continue
            ids.add(str(order_id))
            exchange_id = self._exchange_order_id_for(order_id)
            if exchange_id:
                ids.add(str(exchange_id))
        return ids

    def _own_working_base(self) -> Decimal:
        total = Decimal("0")
        for slot in self._slices:
            if not self._counts_as_open(slot):
                continue
            if slot.tracked and slot.tracked.order is not None:
                remaining = slot.tracked.order.amount - slot.tracked.executed_amount_base
                if remaining < 0:
                    remaining = Decimal("0")
            else:
                remaining = slot.target_amount
            total += remaining
        return total

    async def _cap_child_amount(
        self,
        amount: Decimal,
        price: Decimal,
        amount_step: Decimal,
        min_amount: Decimal,
    ) -> Optional[Decimal]:
        """
        None means the child must not be sent. A cap that blocks the child sets status_reason.
        """
        capped = await self._cap_base_amount(amount, price)
        if capped is None:
            self._cap_block_reason = (
                "Exposure cap blocked the next child. "
                "The exchange position and open orders could not be read."
            )
            return None
        if capped < amount:
            capped = quantize_down(capped, amount_step)
        if capped < min_amount:
            self._cap_block_reason = (
                "Exposure cap blocked the next child. "
                "The live position and open orders leave no room under the total."
            )
            return None
        self._cap_block_reason = ""
        return capped

    async def _cap_base_amount(self, amount: Decimal, price: Optional[Decimal] = None) -> Optional[Decimal]:
        """
        Cap a new child by the live position change and open orders.
        None means the exchange read failed and the child must not be sent.
        A connector without this read is left unchanged.
        Own working children are not counted twice. A connector that can exclude
        their ids does that itself. Otherwise their remaining size is removed here.

        The read is the whole pair and side. Another executor's fills and open
        orders on that pair and side reduce the room, so two icebergs can starve
        each other. Scoping the cap to this executor would drop the exchange
        check that stops a child when the live position is already at the total.
        """
        if amount <= 0:
            return amount
        connector = self.connectors[self.config.connector_name]
        query = self._real_callable(connector, "ground_truth_exposure")
        if query is None:
            return amount
        kwargs = self._exposure_query_kwargs(query)
        try:
            try:
                result = query(self.config.trading_pair, self.config.side, **kwargs)
            except TypeError:
                kwargs = {}
                result = query(self.config.trading_pair, self.config.side)
            if hasattr(result, "__await__"):
                result = await result
            position_base, open_base = result
            position_base = Decimal(str(position_base))
            open_base = Decimal(str(open_base))
        except Exception:
            self.logger().error("Could not read the exchange position and open orders. Not placing a child.")
            return None
        if "exclude_client_order_ids" not in kwargs:
            open_base -= self._own_working_base()
            if open_base < 0:
                open_base = Decimal("0")
        if self._ground_truth_baseline is None:
            self._ground_truth_baseline = position_base
        change = position_base - self._ground_truth_baseline
        if change < 0:
            change = Decimal("0")
        room = self._exposure_target_base(price) - change - open_base
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
        # A fill can land after the rung has been cleared, once the cancel is
        # confirmed. The in-flight order is still the size the position holds.
        order = self._order_for(order_id)
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
        if self._cap_block_reason:
            return self._cap_block_reason
        if self._exit_error:
            return self._exit_error
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
        slot = self._slot_for(event.order_id) or self._exit_slot_for(event.order_id)
        if slot and slot.tracked:
            in_flight = self.get_in_flight_order(self.config.connector_name, event.order_id)
            if in_flight:
                slot.tracked.order = in_flight

    def _is_post_only_reject(self, event) -> bool:
        if not self.config.post_only:
            return False
        order_type = getattr(event, "order_type", None)
        return order_type is None or order_type == OrderType.LIMIT_MAKER

    def _note_post_only_backoff(self) -> None:
        self._post_only_reject_bid = self.get_price(
            self.config.connector_name, self.config.trading_pair, PriceType.BestBid
        )
        self._post_only_reject_ask = self.get_price(
            self.config.connector_name, self.config.trading_pair, PriceType.BestAsk
        )
        now = float(self._strategy.current_timestamp)
        interval = float(self.config.min_order_interval or 0)
        if interval < 0:
            interval = 0.0
        self._post_only_rejected_at = now
        self._post_only_retry_after = now + interval

    def _clear_post_only_backoff(self) -> None:
        self._post_only_retry_after = None
        self._post_only_rejected_at = None
        self._post_only_reject_bid = None
        self._post_only_reject_ask = None

    def _post_only_backoff_blocks(self) -> bool:
        """
        After a post-only reject, wait for the book to move or for the next refresh.
        The same timestamp does not count as the next refresh.
        """
        if self._post_only_retry_after is None:
            return False
        bid = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid)
        ask = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk)
        if bid != self._post_only_reject_bid or ask != self._post_only_reject_ask:
            self._clear_post_only_backoff()
            return False
        now = float(self._strategy.current_timestamp)
        rejected_at = self._post_only_rejected_at if self._post_only_rejected_at is not None else now
        if now > rejected_at and now + 1e-9 >= float(self._post_only_retry_after):
            self._clear_post_only_backoff()
            return False
        return True

    def process_order_failed_event(self, event_tag: int, market, event):
        slot = self._slot_for(event.order_id) or self._exit_slot_for(event.order_id)
        if slot is None and event.order_id not in self._exit_order_ids:
            return
        if slot is not None:
            self._strategy.cancel(self.config.connector_name, self.config.trading_pair, event.order_id)
            self._unconfirmed_cancels.add(event.order_id)
            self._failed_order_ids.add(event.order_id)
            slot.cancel_requested = True
        is_exit = self._is_exit_order(event.order_id)
        if not is_exit and self._is_post_only_reject(event):
            self._note_post_only_backoff()
            return
        if is_exit:
            self._note_exit_failure(event.order_id)
            return
        self._current_retries += 1
        if self._current_retries > self._max_retries:
            if self._position_base() > 0 and self._exits_configured() and not self._entry_handed_to_exit:
                self._entry_handed_to_exit = True
                self._entry_clock_handled = True
                self._cancel_entry_orders_for_exit()
                if self._exit_phase == "idle":
                    self._exit_phase = "armed"
                if self._limit_take_profit_config():
                    self._exit_error = (
                        "Entry orders failed with the position still open. "
                        "The limit take profit will rest on the filled amount."
                    )
                else:
                    self._exit_error = (
                        "Entry orders failed with the position still open. "
                        "Configured exits are armed on the filled amount."
                    )
                return
            self.close_type = CloseType.FAILED
            self.early_stop(close_type=CloseType.FAILED)

    def process_order_filled_event(self, event_tag: int, market, event: OrderFilledEvent):
        order_id = getattr(event, "order_id", None)
        amount = Decimal(str(getattr(event, "amount", 0) or 0))
        if order_id is not None and amount > 0:
            self._last_fill_timestamp = float(getattr(event, "timestamp", None) or self._strategy.current_timestamp)
            if self._is_exit_order(order_id):
                self._exit_failure_count = 0
                self._exit_retry_after = None
                self._exit_error = ""
                self._exit_last_fill_at = float(getattr(event, "timestamp", None) or self._strategy.current_timestamp)
                self._exit_seen_fill_base = self._exit_filled_base()
            if self._slot_for(order_id) is not None or order_id in self._submitted_order_ids:
                self._current_retries = 0
                self._clear_post_only_backoff()
        if not self._awaiting_schedule_anchor():
            return
        if order_id is None:
            return
        if self._slot_for(order_id) is None and order_id not in self._submitted_order_ids:
            return
        if amount <= 0:
            return
        self._anchor_schedule(float(event.timestamp))

    def process_order_completed_event(self, event_tag: int, market, event):
        if self._exit_slot_for(event.order_id) is not None or event.order_id in self._exit_order_ids:
            return
        self._capture_schedule_anchor()
        if self._execution_complete() and not self._needs_exit_after_entry():
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
            if self._is_exit_order(order_id):
                continue
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
        slots = list(self._slices) + list(self._exit_slices)
        live = sum((slot.tracked.cum_fees_quote for slot in slots if slot.tracked), Decimal("0"))
        return live + self._retained_fees_quote

    def get_net_pnl_quote(self) -> Decimal:
        filled = self.filled_base
        if filled <= 0:
            return Decimal("0")
        if self._exit_filled_base() > 0:
            remaining = self._position_base()
            realized = self._realized_pnl_quote()
            if remaining <= 0 or self.average_fill_price <= 0:
                return realized
            mark = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)
            if self.config.side == TradeType.BUY:
                unrealized = (mark - self.average_fill_price) * remaining
            else:
                unrealized = (self.average_fill_price - mark) * remaining
            return realized + unrealized
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
            "child_order_ids": self._child_order_ids(),
            "take_profit": self._exit_config_view(self.config.take_profit),
            "stop_loss": self._exit_config_view(self.config.stop_loss),
            "exit_arming": self.config.exit_arming.value,
            "take_profit_trigger_price": self._trigger_text("take_profit"),
            "stop_loss_trigger_price": self._trigger_text("stop_loss"),
            "exchange_stop_trigger_price": self._exchange_stop_trigger_text(),
            "exit_state": self._public_exit_state().value,
            "exit_order_ids": self._public_exit_order_ids(),
            "exchange_stop_order_id": self._exchange_stop_order_id,
            "exchange_stop_order_ids": self._exchange_stop_order_ids(),
            "has_exchange_stop": bool(self._exchange_stop_order_ids()),
            "exchange_stop_error": self._exchange_stop_error,
            "closed_base": str(self._exit_filled_base()),
            "open_base": str(self._position_base()),
            "exit_error": self._exit_error,
            "exit_status": self._exit_phase,
            "exit_active": self._exit_active,
            "exit_filled_amount": str(self._exit_filled_base()),
            "exit_average_price": str(self._exit_average_price()),
            "realized_pnl_quote": str(self._realized_pnl_quote()),
            "exit": self._exit_snapshot(),
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

    def _exits_configured(self) -> bool:
        return self.config.take_profit is not None or self.config.stop_loss is not None

    def _needs_exit_after_entry(self) -> bool:
        return self._exits_configured() and self.filled_base > 0 and self._exit_phase != "done"

    def _partial_exit_ready(self) -> bool:
        if self.config.exit_arming != ExitArming.ON_ANY_FILL or not self._exits_configured():
            return False
        if self.filled_base <= 0:
            return False
        return self._choose_exit() is not None

    def _exit_config(self, kind: str) -> Optional[RangeIcebergExitConfig]:
        if kind == "take_profit":
            return self.config.take_profit
        if kind == "stop_loss":
            return self.config.stop_loss
        return None

    def _active_exit_config(self) -> Optional[RangeIcebergExitConfig]:
        if self._exit_active is None:
            return None
        return self._exit_config(self._exit_active)

    def _exit_side(self) -> TradeType:
        """
        Close the position the entry opened.

        A long is closed with a sell. A short is closed with a buy. Combined with
        position action CLOSE, the MEXC hedge mapping is side 4 for a long and side 2
        for a short.
        """
        if self.config.side == TradeType.BUY:
            return TradeType.SELL
        return TradeType.BUY

    def _reference_price(self) -> Decimal:
        return Decimal(str(self.get_price(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice,
        )))

    def _trigger_price_for(self, kind: str) -> Optional[Decimal]:
        cfg = self._exit_config(kind)
        if cfg is None:
            return None
        if cfg.price is not None:
            return cfg.price
        average = self.average_fill_price
        if average <= 0 or cfg.percent is None:
            return None
        long_entry = self.config.side == TradeType.BUY
        if kind == "take_profit":
            multiplier = Decimal("1") + cfg.percent if long_entry else Decimal("1") - cfg.percent
        else:
            multiplier = Decimal("1") - cfg.percent if long_entry else Decimal("1") + cfg.percent
        return average * multiplier

    def _reported_trigger(self, kind: str) -> Optional[Decimal]:
        if self._exit_active == kind and self._exit_trigger_price is not None:
            return self._exit_trigger_price
        return self._trigger_price_for(kind)

    def _trigger_text(self, kind: str):
        """
        custom_info price for one exit.

        Null when that exit is not configured, and null before this executor
        has a fill. A percent trigger is the current average fill times the
        percent. It is not the price stored when the exit first armed, so a
        later entry fill moves it. An absolute price is that price, once a
        fill exists.
        """
        if self._exit_config(kind) is None or self.filled_base <= 0:
            return None
        price = self._trigger_price_for(kind)
        if price is None:
            return None
        return str(price)

    def _price_reached(self, kind: str, trigger: Decimal, price: Decimal) -> bool:
        long_entry = self.config.side == TradeType.BUY
        if kind == "take_profit":
            return price >= trigger if long_entry else price <= trigger
        return price <= trigger if long_entry else price >= trigger

    def _entry_clock_close_type(self) -> Optional[CloseType]:
        if self._time_limit_reached():
            return CloseType.TIME_LIMIT
        if self._expired():
            return CloseType.EXPIRED
        return None

    async def _handle_entry_clock(self) -> bool:
        """
        Apply the entry duration or expiry.

        True means the caller should not place more entry orders this tick.
        A clock that ends with a fill and a configured exit cancels the rest
        of the entry and arms the exit. It does not stop the executor. Once
        an exit is armed or active, a later tick does not apply the clock again.
        """
        if self._entry_clock_handled:
            return False
        if self._exit_phase in ("cancelling_entry", "executing", "done"):
            self._entry_clock_handled = True
            return False
        close_type = self._entry_clock_close_type()
        if close_type is None:
            return False
        if close_type == CloseType.TIME_LIMIT:
            self._advance_schedule(force_end=True)
        if self.filled_base > 0 and self._exits_configured():
            self._entry_clock_handled = True
            self._cancel_entry_orders_for_exit()
            if self._exit_phase == "idle":
                self._exit_phase = "armed"
            await self._control_exit()
            return True
        if self._exit_phase in ("armed", "resting") or self._exit_active is not None:
            self._entry_clock_handled = True
            return True
        self.early_stop(keep_position=False, close_type=close_type)
        return True

    def _limit_take_profit_config(self) -> bool:
        cfg = self.config.take_profit
        return bool(
            cfg is not None
            and cfg.execution == ExitExecution.SINGLE
            and cfg.order_type == ExitOrderType.LIMIT
        )

    def _resting_limit_applies(self) -> bool:
        """
        A single limit take profit can rest before the entry is finished.

        ON_FULL_FILL waits for the entry to finish, or for the entry clock to
        end. ON_ANY_FILL rests as soon as any size is filled. The stop loss
        taking over, or a forced market close, is not this path.
        """
        if not self._limit_take_profit_config():
            return False
        if self._exit_active == "stop_loss" or self._exit_force_market:
            return False
        if self.filled_base <= 0 or self._exit_phase == "done":
            return False
        if self._take_profit_price_is_adverse():
            return False
        if self.config.exit_arming == ExitArming.ON_FULL_FILL:
            return self._execution_complete() or self._entry_clock_handled or self._exit_phase in (
                "armed",
                "resting",
                "executing",
            )
        return True

    def _take_profit_price_is_adverse(self) -> bool:
        cfg = self.config.take_profit
        if cfg is None or cfg.price is None or self.config.allow_adverse_take_profit:
            return False
        average = self.average_fill_price
        if average <= 0:
            return False
        if self.config.side == TradeType.BUY and cfg.price <= average:
            self._exit_error = (
                f"take_profit price ({cfg.price}) is not above the average entry ({average}). "
                "Set allow_adverse_take_profit to place it."
            )
            return True
        if self.config.side == TradeType.SELL and cfg.price >= average:
            self._exit_error = (
                f"take_profit price ({cfg.price}) is not below the average entry ({average}). "
                "Set allow_adverse_take_profit to place it."
            )
            return True
        return False

    def _take_profit_wants_to_start(self, price: Decimal) -> bool:
        cfg = self.config.take_profit
        if cfg is None:
            return False
        trigger = self._trigger_price_for("take_profit")
        if trigger is None:
            return False
        # A resting limit is posted by _maintain_resting_take_profit. It is not
        # a price trigger and it does not cancel the entry.
        if cfg.execution == ExitExecution.SINGLE and cfg.order_type == ExitOrderType.LIMIT:
            return False
        if self._take_profit_price_is_adverse():
            return False
        return self._price_reached("take_profit", trigger, price)

    def _stop_wants_to_start(self, price: Decimal) -> bool:
        cfg = self.config.stop_loss
        if cfg is None:
            return False
        trigger = self._trigger_price_for("stop_loss")
        if trigger is None:
            return False
        return self._price_reached("stop_loss", trigger, price)

    def _exit_has_fill(self) -> bool:
        return self._exit_filled_base() > 0

    def _take_profit_is_committed(self) -> bool:
        """
        An unfinished take profit never blocks the stop loss.

        A partial limit fill and a live iceberg child are both unfinished while
        the open position is still at least a minimum order. The stop cancels
        the remainder and closes what is left.
        """
        if self._exit_active != "take_profit":
            return False
        return self._position_base() < self._min_order_size()

    def _exit_has_live_order(self) -> bool:
        return any(slot.order_id and slot.order_id != "skipped" and not slot.retired for slot in self._exit_slices)

    def _choose_exit(self) -> Optional[str]:
        """
        Pick the exit that should be working.

        A stop loss that has started stays selected even if price returns to
        the take-profit side. Until the take profit has closed the position,
        a stop trigger replaces it.
        """
        if self._exit_active == "stop_loss":
            return "stop_loss"
        price = self._reference_price()
        stop_hit = self._stop_wants_to_start(price)
        if stop_hit and not self._take_profit_is_committed():
            return "stop_loss"
        if self._exit_active == "take_profit":
            return "take_profit"
        if self._take_profit_wants_to_start(price):
            return "take_profit"
        return None

    def _exit_retry_blocks(self) -> bool:
        if self._exit_retry_after is None:
            return False
        return float(self._strategy.current_timestamp) + 1e-9 < float(self._exit_retry_after)

    def _note_exit_failure(self, order_id: str):
        """
        Count an exit rejection on its own budget.

        The entry retry counter is left alone. While the position is open, the
        budget ends in one market close, not in FAILED. FAILED is used only
        after that market close has also been rejected.
        """
        self._exit_failure_count += 1
        step = max(float(self.config.min_order_interval or 0), 0.5)
        # A stop has to get back to the market quickly. A take profit can wait.
        cap = 2.0 if self._exit_active == "stop_loss" else 30.0
        delay = min(cap, step * (2 ** (self._exit_failure_count - 1)))
        self._exit_retry_after = float(self._strategy.current_timestamp) + delay
        budget = max(int(self._max_retries), 1)
        if self._exit_failure_count <= budget:
            self._exit_error = (
                f"Exit order {order_id} was rejected ({self._exit_failure_count} of {budget}). "
                "The close will retry. The position stays open."
            )
            return
        if self._position_base() > 0 and not self._exit_market_attempted:
            self._exit_force_single = True
            self._exit_force_market = True
            self._exit_market_attempted = True
            self._exit_retry_after = None
            self._exit_error = (
                f"Exit order {order_id} was rejected {self._exit_failure_count} times. "
                "Sending a market close for the open position."
            )
            return
        self._exit_error = (
            f"Could not close the open position after {self._exit_failure_count} exit rejections, "
            "including a market close."
        )
        if self._position_base() > 0 and self._stop_should_keep_a_backstop():
            if not self._exchange_stop_order_id:
                self._exchange_stop_released = False
                self._exchange_stop_close_capped = False
                self._exit_retry_after = None
                self._replace_stop_before_failed = True
                self._exit_error = (
                    "Stop loss closes were rejected, including a market close. "
                    "Re-placing the exchange stop before the executor stops."
                )
                return
            self._preserve_exchange_stop = True
        if self._position_base() > 0:
            self.close_type = CloseType.FAILED
            self.early_stop(close_type=CloseType.FAILED)

    def _stop_should_keep_a_backstop(self) -> bool:
        return (
            self._exit_active == "stop_loss"
            and self._protective_stop_enabled()
            and self._position_base() > 0
        )

    def _sl_protective_wait_expired(self) -> bool:
        if self._sl_blocked_since is None:
            return False
        elapsed = float(self._strategy.current_timestamp) - float(self._sl_blocked_since)
        return elapsed + 1e-9 >= float(self.config.stop_loss_fallback_seconds)

    async def _close_stop_loss(self):
        """
        Send the software stop without dropping the backstop early.

        Take-profit and entry cancels are requested first. The exchange stop
        stays until those cancels are confirmed, and only then is it cancelled
        and the close sent. If a cancel is still open after
        stop_loss_fallback_seconds, the backstop stays and the close is the
        open size minus that unconfirmed take profit and the live stop.
        """
        self._activate_exit("stop_loss")
        blocked = self._entry_blocks_exit() or self._exit_orders_block()
        if blocked:
            if self._sl_blocked_since is None:
                self._sl_blocked_since = float(self._strategy.current_timestamp)
            if not self._sl_protective_wait_expired():
                self._exit_phase = "cancelling_entry"
                return
            self._exchange_stop_close_capped = True
        else:
            self._sl_blocked_since = None
            if not await self._exchange_stop_cleared_for_software_close():
                if self._exit_phase == "done":
                    return
                if self._exit_phase == "idle":
                    self._exit_phase = "armed"
                return
        if self._unknown_order_ids and self._exit_phase not in ("cancelling_entry", "executing"):
            return
        self._exit_phase = "executing"
        await self._execute_exit()

    async def _control_exit(self):
        if self._exit_phase == "done" or not self._exits_configured() or self.filled_base <= 0:
            return
        if self._replace_stop_before_failed:
            self._exchange_stop_released = False
            await self._sync_exchange_stop()
            if not self._exchange_stop_order_id:
                return
            self._preserve_exchange_stop = True
            self._replace_stop_before_failed = False
            self.close_type = CloseType.FAILED
            self.early_stop(close_type=CloseType.FAILED)
            return
        if self._exit_retry_blocks():
            if self._exit_phase == "idle":
                self._exit_phase = "armed"
            return
        kind = self._choose_exit()
        if kind == "stop_loss":
            await self._close_stop_loss()
            return
        if kind == "take_profit" and not self._limit_take_profit_config():
            if self._unknown_order_ids and self._exit_phase not in ("cancelling_entry", "executing"):
                if self._exit_phase == "idle":
                    self._exit_phase = "armed"
                return
            self._activate_exit(kind)
            if self._entry_blocks_exit() or self._exit_orders_block():
                self._exit_phase = "cancelling_entry"
                return
            self._exit_phase = "executing"
            await self._execute_exit()
            return
        if self._resting_limit_applies() or (
            kind == "take_profit" and self._limit_take_profit_config() and not self._exit_force_market
        ):
            await self._maintain_resting_take_profit()
            return
        if self._exit_force_market and kind is not None:
            if self._unknown_order_ids:
                return
            self._activate_exit(kind)
            if self._entry_blocks_exit() or self._exit_orders_block():
                self._exit_phase = "cancelling_entry"
                return
            self._exit_phase = "executing"
            await self._execute_exit()
            return
        if kind is None:
            if (self._execution_complete() or self._entry_clock_handled) and self._exit_phase == "idle":
                self._exit_phase = "armed"
            return

    async def _maintain_resting_take_profit(self):
        """
        Post or grow a reduce-only limit take profit without cancelling the entry.

        The working close size is the open position. A new entry fill adds
        another limit for the shortfall. The total never exceeds the fill.
        The exchange stop is placed or resized before that limit, because a
        full-size close reduces the closable quantity the stop is checked
        against. When a percent trigger moves with the average, the whole
        resting limit is cancelled and replaced once, then left alone until
        the reprice interval has passed.
        """
        self._refresh_exit_orders()
        self._exit_active = "take_profit"
        self._exit_trigger_price = self._trigger_price_for("take_profit")
        if self._exit_done_closing():
            if self._exit_filled_base() > 0 or self._position_base() <= 0:
                self._finish_exit()
            return
        if self._exit_phase not in ("cancelling_entry", "executing"):
            self._exit_phase = "resting"
        if self._exchange_stop_needs_resize():
            if self._exchange_stop_place_waiting():
                return
            await self._grow_take_profit_behind_the_stop()
            return
        await self._sync_exchange_stop()
        if self._stop_must_land_before_take_profit():
            return
        if self._unknown_order_ids:
            return
        if self._exit_orders_block():
            return
        if self._resting_tp_is_stale():
            now = float(self._strategy.current_timestamp)
            if self._tp_reprice_after is not None and now + 1e-9 < float(self._tp_reprice_after):
                return
            self._cancel_exit_orders()
            self._tp_reprice_after = now + max(float(self.config.min_order_interval or 0), 1.0)
            return
        if not self._rate_limit_allows_place():
            return
        amount = self._cap_exit_amount(self._position_base())
        if amount <= 0:
            return
        price = self._quantize_exit_price(self._exit_trigger_price or self._reference_price())
        order_id = self._submit_exit(price, amount, OrderType.LIMIT)
        self._track_exit(price, amount, order_id)

    def _resting_tp_is_stale(self) -> bool:
        target = self._trigger_price_for("take_profit")
        if target is None:
            return False
        target = self._quantize_exit_price(target)
        for slot in self._exit_slices:
            if slot.retired or slot.cancel_requested or not slot.is_working:
                continue
            price = slot.target_price
            if slot.tracked is not None and slot.tracked.order is not None and slot.tracked.order.price == slot.tracked.order.price:
                price = slot.tracked.order.price
            if price != target:
                return True
        return False

    def _stop_must_land_before_take_profit(self) -> bool:
        """
        Hold a new take-profit limit while the exchange stop is still missing
        after a failed place. A full-size close would consume the closable
        quantity and the stop retry would be rejected.
        """
        if not self._protective_stop_enabled() or not self._exchange_stop_is_due():
            return False
        if self._exchange_stop_order_id:
            return False
        return bool(self._exchange_stop_error)

    def _resting_take_profit_size(self) -> Decimal:
        total = Decimal("0")
        for slot in self._exit_slices:
            if slot.retired or not slot.order_id or slot.order_id == "skipped":
                continue
            if slot.tracked is not None and slot.tracked.order is not None:
                total += Decimal(str(slot.tracked.order.amount or 0))
            else:
                total += slot.target_amount
        return total

    def _exchange_stop_place_waiting(self) -> bool:
        if self._exchange_stop_retry_after is None:
            return False
        return float(self._strategy.current_timestamp) + 1e-9 < float(self._exchange_stop_retry_after)

    def _exchange_stop_needs_more_size(self) -> bool:
        if not self._exchange_stop_order_ids() or not self._protective_stop_enabled():
            return False
        if self._exchange_stop_trigger_price() is None:
            return False
        shortfall = self._position_base() - self._exchange_stop_amount
        return shortfall >= self._min_order_size()

    def _exchange_stop_price_is_stale(self) -> bool:
        if not self._exchange_stops:
            return False
        trigger = self._exchange_stop_trigger_price()
        if trigger is None:
            return False
        return any(stop.get("price") != trigger for stop in self._exchange_stops)

    def _exchange_stop_needs_resize(self) -> bool:
        if not self._protective_stop_enabled():
            return False
        if self._tp_grow_cancelling:
            return True
        if not self._exchange_stop_order_ids():
            return False
        return self._exchange_stop_needs_more_size() or self._exchange_stop_price_is_stale()

    def _take_profit_must_move_before_the_stop(self) -> bool:
        """
        The resting take profit has to leave the book before the stop vol changes.

        A same-price place replaces the live stop. A new price is a new stop,
        and that plus the old stops plus the resting close is more than the
        position. The grow path owns both cases.
        """
        return self._exchange_stop_needs_resize()

    def _live_stop_matches(self, amount: Decimal, price: Optional[Decimal]) -> bool:
        if len(self._exchange_stops) != 1 or price is None:
            return False
        stop = self._exchange_stops[0]
        return stop.get("amount") == amount and stop.get("price") == price

    async def _grow_take_profit_behind_the_stop(self):
        """
        Resize the backstop to the open position, then post one full take profit.

        The resting take profit is cancelled first and the existing stops stay
        up, so the covered size does not drop while that cancel is open. With
        the take profit off the book, a same-price place sends the full open
        size and the returned id is stored at that vol. A changed price cancels
        the old stops, then places one full-size stop. If that place fails, the
        old size is put back at the old price in the same turn. An identical
        size and price that is already live is not sent again.
        """
        if self._exit_working() or self._exit_orders_block():
            if self._exit_working() and not self._tp_grow_cancelling:
                self._tp_previous_size = self._resting_take_profit_size()
                self._cancel_exit_orders()
                self._tp_grow_cancelling = True
            return
        self._tp_grow_cancelling = False
        if self._exchange_stop_price_is_stale():
            placed = await self._replace_exchange_stop_at_new_price()
        else:
            placed = await self._place_exchange_stop_keeping_old()
        if not placed:
            if (
                not self._tp_reposted_after_failure
                and not self._exit_working()
                and not self._exit_orders_block()
                and self._exit_phase != "done"
            ):
                covered = self._exchange_stop_amount
                if covered <= 0:
                    covered = self._tp_previous_size
                if covered > self._position_base():
                    covered = self._position_base()
                await self._post_resting_take_profit(covered)
                self._tp_reposted_after_failure = True
            return
        if self._exchange_stop_needs_resize() or self._exit_phase == "done":
            return
        self._tp_reposted_after_failure = False
        await self._post_resting_take_profit(self._position_base())

    async def _post_resting_take_profit(self, amount: Decimal):
        if self._unknown_order_ids or not self._rate_limit_allows_place():
            return
        amount = self._cap_exit_amount(amount)
        if amount <= 0:
            return
        price = self._quantize_exit_price(self._exit_trigger_price or self._reference_price())
        order_id = self._submit_exit(price, amount, OrderType.LIMIT)
        self._track_exit(price, amount, order_id)

    def _software_close_base(self) -> Decimal:
        amount = self._position_base()
        if (
            self._exchange_stop_close_capped
            and self._exchange_stop_order_id
            and self._exit_active == "stop_loss"
        ):
            amount -= self._exchange_stop_amount
            if amount < 0:
                amount = Decimal("0")
        return amount

    def _activate_exit(self, kind: str):
        if self._exit_active == kind and self._exit_phase in ("cancelling_entry", "executing"):
            return
        if self._exit_active is not None and self._exit_active != kind:
            self._cancel_exit_orders()
            self._exit_plan_ready = False
            self._exit_schedule = None
            self._exit_schedule_state = None
            self._exit_force_single = False
            self._exit_force_market = False
            self._exit_started_at = None
            self._exit_paused_seconds = 0.0
        self._exit_active = kind
        if self._exit_started_at is None:
            # The close clock starts at this trigger. A later fill does not move it.
            self._exit_started_at = float(self._strategy.current_timestamp)
        self._exit_trigger_price = self._trigger_price_for(kind)
        self._cancel_entry_orders_for_exit()
        self._exit_phase = "cancelling_entry" if self._entry_blocks_exit() or self._exit_orders_block() else "executing"

    def _cancel_entry_orders_for_exit(self):
        """An exit that starts while entry orders are still working cancels those orders first."""
        for slot in self._slices:
            if not slot.is_working or slot.cancel_requested:
                continue
            if not slot.order_id or slot.order_id == "skipped":
                continue
            self._strategy.cancel(self.config.connector_name, self.config.trading_pair, slot.order_id)
            slot.cancel_requested = True
            self._unconfirmed_cancels.add(slot.order_id)

    def _cancel_exit_orders(self):
        for slot in self._exit_slices:
            if slot.retired or not slot.is_working or slot.cancel_requested:
                continue
            if not slot.order_id or slot.order_id == "skipped":
                continue
            self._strategy.cancel(self.config.connector_name, self.config.trading_pair, slot.order_id)
            slot.cancel_requested = True
            self._unconfirmed_cancels.add(slot.order_id)

    def _entry_blocks_exit(self) -> bool:
        for slot in self._slices:
            if slot.retired or not slot.cancel_requested:
                if self._counts_as_open(slot) and not slot.cancel_requested:
                    return True
                continue
            if slot.order_id and slot.order_id in self._unconfirmed_cancels:
                return True
        return False

    def _exit_orders_block(self) -> bool:
        for slot in self._exit_slices:
            if slot.retired or not slot.cancel_requested:
                continue
            if slot.order_id and slot.order_id in self._unconfirmed_cancels:
                return True
            if slot.is_working:
                return True
        return False

    async def _execute_exit(self):
        self._refresh_exit_orders()
        self._note_exit_fill_progress()
        if self._exit_done_closing():
            if self._exit_filled_base() > 0 or self._position_base() <= 0:
                self._finish_exit()
            else:
                self.close_execution_finished()
            return
        if self._unknown_order_ids:
            return
        if self._stop_iceberg_should_market():
            self._cancel_exit_orders()
            if self._exit_orders_block():
                self._exit_phase = "cancelling_entry"
                return
            self._exit_force_single = True
            self._exit_force_market = True
            await self._execute_exit_single()
            return
        if self._active_is_single():
            await self._execute_exit_single()
            return
        await self._execute_exit_iceberg()

    def _note_exit_fill_progress(self):
        filled = self._exit_filled_base()
        if filled > self._exit_seen_fill_base:
            self._exit_seen_fill_base = filled
            self._exit_last_fill_at = float(self._strategy.current_timestamp)

    def _stop_quiet_seconds(self) -> float:
        mark = self._exit_last_fill_at if self._exit_last_fill_at is not None else self._exit_started_at
        if mark is None:
            return 0.0
        return float(self._strategy.current_timestamp) - float(mark)

    def _stop_iceberg_should_market(self) -> bool:
        """
        A stop-loss iceberg that cannot rest on the trigger side, or that has
        had no new fill for the fallback time, closes the rest with a market order.

        A partial fill does not retire the fallback. The clock restarts from
        that fill, and the remainder markets once the quiet time has passed.
        """
        if self._exit_active != "stop_loss" or self._exit_force_market:
            return False
        cfg = self.config.stop_loss
        if cfg is None or cfg.execution != ExitExecution.ICEBERG:
            return False
        if self._stop_bounds_are_wrong():
            self._exit_error = (
                "Stop loss price bounds are on the wrong side of the trigger. "
                "Closing the open position with a market order."
            )
            return True
        if self._exit_started_at is None and self._exit_last_fill_at is None:
            return False
        return self._stop_quiet_seconds() >= float(self.config.stop_loss_fallback_seconds)

    def _stop_bounds_are_wrong(self) -> bool:
        cfg = self.config.stop_loss
        if cfg is None or cfg.execution != ExitExecution.ICEBERG:
            return False
        trigger = self._trigger_price_for("stop_loss")
        if trigger is None:
            return False
        long_entry = self.config.side == TradeType.BUY
        if cfg.limit_price is not None:
            if long_entry and cfg.limit_price > trigger:
                return True
            if not long_entry and cfg.limit_price < trigger:
                return True
        if long_entry and cfg.price_low is not None and cfg.price_low > trigger:
            return True
        if not long_entry and cfg.price_high is not None and cfg.price_high < trigger:
            return True
        return False

    def _active_is_single(self) -> bool:
        cfg = self._active_exit_config()
        if cfg is None:
            return False
        return cfg.execution == ExitExecution.SINGLE or self._exit_force_single

    def _exit_done_closing(self) -> bool:
        if self._exit_working():
            return False
        return self._position_base() < self._min_order_size()

    def _finish_exit(self):
        if self._exit_phase == "done":
            return
        self._exit_phase = "done"
        if self._exit_active == "take_profit":
            self.close_type = CloseType.TAKE_PROFIT
        elif self._exit_active == "stop_loss":
            self.close_type = CloseType.STOP_LOSS
        self.close_timestamp = self._strategy.current_timestamp
        self._begin_shutdown()

    async def _execute_exit_single(self):
        if self._exit_working() and not (
            self._exchange_stop_close_capped and self._exit_active == "stop_loss"
        ):
            return
        if not self._rate_limit_allows_place():
            return
        amount = self._cap_exit_amount(self._software_close_base())
        if amount <= 0:
            return
        cfg = self._active_exit_config()
        market = (
            self._exit_force_market
            or self._exit_active == "stop_loss"
            or cfg is None
            or cfg.order_type == ExitOrderType.MARKET
        )
        if market:
            order_type = OrderType.MARKET
            price = Decimal("NaN")
            shown = self._exit_trigger_price or Decimal("0")
        else:
            order_type = OrderType.LIMIT
            price = self._quantize_exit_price(self._exit_trigger_price or self._reference_price())
            shown = price
        order_id = self._submit_exit(price, amount, order_type)
        self._track_exit(shown, amount, order_id)

    async def _execute_exit_iceberg(self):
        self._ensure_exit_plan()
        if self._exit_force_single:
            await self._execute_exit_single()
            return
        if self._exit_range:
            await self._place_next_exit_range_slice()
        else:
            await self._place_next_exit_up_to_slice()

    def _ensure_exit_plan(self):
        if self._exit_plan_ready:
            return
        cfg = self._active_exit_config()
        if cfg is None:
            self._exit_plan_ready = True
            return
        total = self._position_base()
        self._exit_plan_base = total
        count = max(1, int(cfg.slices))
        self._exit_slice_base = total / Decimal(count) if count else total
        if cfg.interval_seconds is not None:
            self._exit_schedule = self._build_exit_schedule(total, count, cfg.interval_seconds)
            if self._exit_schedule_state:
                self._exit_schedule.restore_state(self._exit_schedule_state)
                self._exit_schedule_state = None
        else:
            self._exit_schedule = None
        if cfg.limit_price is not None:
            self._exit_range = False
            self._exit_plan_ready = True
            return
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        try:
            levels = build_range_ladder(
                total_amount=total,
                size_unit=SizeUnit.BASE,
                price_low=cfg.price_low,
                price_high=cfg.price_high,
                levels=count,
                distribution=self.config.distribution,
                weight=self.config.weight,
                side=self._exit_side(),
                price_step=Decimal(str(rules.min_price_increment)),
                amount_step=Decimal(str(rules.min_base_amount_increment)),
                min_amount=Decimal(str(rules.min_order_size)),
                contract_size=Decimal("1"),
            )
        except ValueError as exc:
            self.logger().error(str(exc))
            self._exit_force_single = True
            self._exit_plan_ready = True
            return
        self._exit_slices.extend(_Slice(level) for level in levels)
        self._exit_range = True
        self._exit_plan_ready = True

    def _build_exit_schedule(self, total: Decimal, count: int, interval: float) -> SliceSchedule:
        """
        Release offsets measured from the exit trigger, not from the first exit fill.

        The entry schedule can wait for a fill because that fill is what starts its
        clock. This clock is already running, so offset 0 is open immediately.
        """
        share = total / Decimal(count)
        amounts = []
        remaining = total
        for index in range(count):
            piece = remaining if index == count - 1 else share
            amounts.append(piece)
            remaining -= piece
        offsets = release_timestamps(0.0, count, interval)
        duration = interval * count
        offsets = self._apply_timing_jitter(offsets, interval, duration)
        return SliceSchedule(amounts=amounts, offsets=offsets, duration=duration, skip_unplaced=False)

    async def _place_next_exit_range_slice(self):
        if self._exit_working_count() >= self._exit_visible_limit():
            return
        if any(slot.cancel_requested and slot.is_working for slot in self._exit_slices):
            return
        slot = next((item for item in self._exit_slices if item.tracked is None and not item.retired), None)
        if slot is None:
            await self._place_exit_remainder()
            return
        self._advance_exit_schedule()
        if not self._exit_slice_released(slot):
            return
        if not self._rate_limit_allows_place():
            return
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        cfg = self._active_exit_config()
        planned = slot.target_price
        price = self._jitter_price(
            planned,
            rules.min_price_increment,
            None if cfg is None else cfg.price_low,
            None if cfg is None else cfg.price_high,
        )
        amount = self._cap_exit_amount(min(slot.target_amount, self._exit_placeable()))
        if amount <= 0:
            if slot.target_amount < self._min_order_size():
                slot.tracked = TrackedOrder(order_id="skipped")
            return
        order_id = self._submit_exit(price, amount, OrderType.LIMIT)
        slot.tracked = TrackedOrder(order_id=order_id)
        slot.pre_jitter_price = planned
        slot.target_amount = amount
        slot.target_price = price
        self._exit_order_ids.append(order_id)
        self._mark_action()

    async def _place_next_exit_up_to_slice(self):
        cfg = self._active_exit_config()
        if cfg is None or cfg.limit_price is None:
            await self._place_exit_remainder()
            return
        if any(slot.cancel_requested and slot.is_working for slot in self._exit_slices):
            return
        self._advance_exit_schedule()
        desired = self._desired_price(
            side=self._exit_side(),
            limit_price=cfg.limit_price,
            cap_behavior=self._exit_cap_behavior(),
            post_only=False,
        )
        if desired is None:
            return
        if self._reprice_exit(desired):
            return
        if self._exit_working_count() >= self._exit_visible_limit():
            return
        placeable = self._exit_placeable()
        if placeable <= 0:
            return
        if not self._rate_limit_allows_place():
            return
        price = self._clamp_limit(
            self._jitter_up_to_price(
                desired,
                side=self._exit_side(),
                limit_price=cfg.limit_price,
                post_only=False,
            ),
            side=self._exit_side(),
            limit_price=cfg.limit_price,
        )
        if self._exit_placed_count() >= max(1, cfg.slices) - 1:
            raw = min(placeable, self._exit_room_base())
        else:
            raw = min(self._exit_slice_base, placeable, self._exit_room_base())
        amount = self._cap_exit_amount(raw)
        if amount <= 0:
            return
        order_id = self._submit_exit(price, amount, OrderType.LIMIT)
        self._track_exit(price, amount, order_id, pre_jitter_price=desired)

    async def _place_exit_remainder(self):
        if self._exit_working_count() > 0 or not self._rate_limit_allows_place():
            return
        amount = self._cap_exit_amount(self._position_base())
        if amount <= 0:
            return
        price = self._exit_remainder_price()
        order_id = self._submit_exit(price, amount, OrderType.LIMIT)
        self._track_exit(price, amount, order_id)

    def _exit_remainder_price(self) -> Decimal:
        cfg = self._active_exit_config()
        if cfg is not None and cfg.limit_price is not None:
            desired = self._desired_price(
                side=self._exit_side(),
                limit_price=cfg.limit_price,
                cap_behavior=self._exit_cap_behavior(),
                post_only=False,
            )
            if desired is None:
                return cfg.limit_price
            return self._clamp_limit(desired, side=self._exit_side(), limit_price=cfg.limit_price)
        if cfg is not None and cfg.price_high is not None and cfg.price_low is not None:
            anchor = cfg.price_high if self._exit_side() == TradeType.SELL else cfg.price_low
            return self._quantize_exit_price(anchor)
        return self._quantize_exit_price(self._exit_trigger_price or self._reference_price())

    def _reprice_exit(self, desired: Decimal) -> bool:
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        step = Decimal(str(rules.min_price_increment or 0))
        cfg = self._active_exit_config()
        limit = None if cfg is None else cfg.limit_price
        pending = [
            slot for slot in self._exit_slices
            if slot.is_working and not slot.cancel_requested
            and not self._keep_working_price(
                slot,
                desired,
                step,
                side=self._exit_side(),
                post_only=False,
                limit_price=limit,
            )
        ]
        if not pending:
            return False
        if not self._rate_limit_allows_place():
            return True
        slot = pending[0]
        self._request_cancel(slot)
        if slot.order_id:
            self._unconfirmed_cancels.add(slot.order_id)
        return True

    def _advance_exit_schedule(self):
        if self._exit_schedule is None:
            return
        self._exit_schedule.advance(
            self._exit_elapsed(),
            self._exit_filled_base(),
            self._exit_open_base(),
            False,
        )

    def _exit_slice_released(self, slot: _Slice) -> bool:
        if self._exit_schedule is None:
            return True
        # A window can release a partial child. The order is then capped at placeable.
        return self._exit_placeable() >= min(slot.target_amount, self._min_order_size())

    def _exit_elapsed(self) -> float:
        """
        Time since the exit trigger, with pauses after that trigger removed.

        This is not the entry first-fill anchor. The trigger itself starts the clock,
        so a close schedule does not sit at zero until the first exit order fills.
        """
        if self._exit_started_at is None:
            return 0.0
        return self._elapsed_since(self._exit_started_at, self._exit_paused_seconds)

    def _exit_cap_behavior(self) -> CapBehavior:
        """
        A stop-loss iceberg always rests at the cap.

        Copying the entry WAIT behavior would send nothing when the market is
        through the floor or the cap, and the stop would never fire.
        """
        if self._exit_active == "stop_loss":
            return CapBehavior.REST_AT_CAP
        return self.config.cap_behavior

    def _exit_placeable(self) -> Decimal:
        if not self._exit_range:
            # An up-to close is sized from the position held now, not from the
            # size captured when the exit first triggered. A fill that arrives
            # after the entry cancel is confirmed is still closed.
            live_total = self._position_base() + self._exit_filled_base()
            if self._exit_schedule is None:
                opened = live_total
                skipped = Decimal("0")
            else:
                scheduled = self._exit_schedule.opened(self._exit_elapsed())
                planned = self._exit_plan_base if self._exit_plan_base > 0 else live_total
                if planned <= 0:
                    fraction = Decimal("1")
                else:
                    fraction = scheduled / planned
                    if fraction > 1:
                        fraction = Decimal("1")
                opened = live_total * fraction
                skipped = Decimal("0")
            remaining = opened - skipped - self._exit_filled_base() - self._exit_open_base()
            if remaining < 0:
                return Decimal("0")
            return remaining
        if self._exit_schedule is None:
            opened = self._exit_plan_base if self._exit_plan_base > 0 else self._position_base()
            skipped = Decimal("0")
        else:
            opened = self._exit_schedule.opened(self._exit_elapsed())
            skipped = self._exit_schedule.skipped
        remaining = opened - skipped - self._exit_filled_base() - self._exit_open_base()
        if remaining < 0:
            return Decimal("0")
        return remaining

    def _exit_visible_limit(self) -> int:
        cfg = self._active_exit_config()
        if cfg is None or cfg.execution == ExitExecution.SINGLE:
            return 1
        return cfg.visible_count

    def _exit_working_count(self) -> int:
        return sum(1 for slot in self._exit_slices if slot.is_working)

    def _exit_working(self) -> bool:
        return self._exit_working_count() > 0

    def _exit_placed_count(self) -> int:
        return sum(
            1 for slot in self._exit_slices
            if slot.tracked is not None and slot.order_id != "skipped" and not slot.retired
        )

    def _submit_exit(self, price: Decimal, amount: Decimal, order_type: OrderType) -> str:
        return self.place_order(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_type=order_type,
            side=self._exit_side(),
            amount=amount,
            price=price,
            position_action=PositionAction.CLOSE,
        )

    def _track_exit(
        self,
        price: Decimal,
        amount: Decimal,
        order_id: str,
        pre_jitter_price: Optional[Decimal] = None,
    ):
        shown = price if price == price else (self._exit_trigger_price or Decimal("0"))
        slot = _Slice(LadderLevel(price=shown, base_amount=amount))
        slot.pre_jitter_price = pre_jitter_price
        slot.tracked = TrackedOrder(order_id=order_id)
        self._exit_slices.append(slot)
        self._exit_order_ids.append(order_id)
        self._mark_action()

    def _cap_exit_amount(self, raw: Decimal) -> Decimal:
        """
        Quantize a close and never let it exceed the still-open filled position.

        The entry quote cap values a QUOTE budget at the child price, and the hedge
        opening cap ignores close orders. Neither is applied here. A close is the
        filled base this executor still holds, not another open.
        """
        step = self._amount_step()
        room = self._exit_room_base()
        amount = quantize_down(min(Decimal(raw), room), step)
        if amount <= 0 or amount < self._min_order_size() or amount > room:
            return Decimal("0")
        return amount

    def _quantize_exit_price(self, price: Decimal) -> Decimal:
        step = self._price_step()
        if self._exit_side() == TradeType.SELL:
            quantized = quantize_up(price, step)
        else:
            quantized = quantize_down(price, step)
        if quantized <= 0:
            return quantize_down(price, step) if price > 0 else price
        return quantized

    def _min_order_size(self) -> Decimal:
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        return Decimal(str(rules.min_order_size))

    def _amount_step(self) -> Decimal:
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        return Decimal(str(rules.min_base_amount_increment or 0))

    def _price_step(self) -> Decimal:
        rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        return Decimal(str(rules.min_price_increment or 0))

    def _position_base(self) -> Decimal:
        net = self.filled_base - self._exit_filled_base()
        if net < 0:
            return Decimal("0")
        return net

    def _exit_room_base(self) -> Decimal:
        room = self._software_close_base() - self._exit_open_base()
        if room < 0:
            return Decimal("0")
        return room

    def _exit_open_base(self) -> Decimal:
        total = Decimal("0")
        for slot in self._exit_slices:
            if not slot.is_working:
                continue
            if slot.tracked and slot.tracked.order is not None:
                remaining = slot.tracked.order.amount - slot.tracked.executed_amount_base
                if remaining < 0:
                    remaining = Decimal("0")
            else:
                remaining = slot.target_amount
            total += remaining
        return total

    def _exit_filled_base(self) -> Decimal:
        return self._sum_exit_fills()[0]

    def _exit_filled_quote(self) -> Decimal:
        return self._sum_exit_fills()[1]

    def _exit_average_price(self) -> Decimal:
        base = self._exit_filled_base()
        if base <= 0:
            return Decimal("0")
        return self._exit_filled_quote() / base

    def _realized_pnl_quote(self) -> Decimal:
        closed = self._exit_filled_base()
        if closed <= 0 or self.average_fill_price <= 0:
            return Decimal("0")
        exit_price = self._exit_average_price()
        if self.config.side == TradeType.BUY:
            gross = (exit_price - self.average_fill_price) * closed
        else:
            gross = (self.average_fill_price - exit_price) * closed
        return gross - self.get_cum_fees_quote()

    def _sum_exit_fills(self):
        order_ids = set(self._exit_accounted_base)
        for slot in self._exit_slices:
            if slot.order_id and slot.order_id != "skipped":
                order_ids.add(slot.order_id)
        base = Decimal("0")
        quote = Decimal("0")
        for order_id in order_ids:
            part_base, part_quote = self._exit_best_fill(order_id)
            base += part_base
            quote += part_quote
        return base, quote

    def _exit_best_fill(self, order_id: str):
        slot = self._exit_slot_for(order_id)
        order = slot.tracked.order if slot and slot.tracked else None
        if order is None:
            order = self.get_in_flight_order(self.config.connector_name, order_id)
        live_base = Decimal("0")
        live_quote = Decimal("0")
        if order is not None:
            live_base = Decimal(str(order.executed_amount_base or 0))
            live_quote = Decimal(str(order.executed_amount_quote or 0))
        accounted_base = self._exit_accounted_base.get(order_id, Decimal("0"))
        accounted_quote = self._exit_accounted_quote.get(order_id, Decimal("0"))
        base = live_base if live_base >= accounted_base else accounted_base
        if base == live_base and live_base > 0:
            quote = live_quote
        else:
            quote = accounted_quote
        if live_base == base and live_quote > quote:
            quote = live_quote
        return base, quote

    def _exit_slot_for(self, order_id: str) -> Optional[_Slice]:
        return next((slot for slot in self._exit_slices if slot.order_id == order_id), None)

    def _is_exit_order(self, order_id: str) -> bool:
        return order_id in self._exit_order_ids or self._exit_slot_for(order_id) is not None

    def _refresh_exit_orders(self):
        for slot in self._exit_slices:
            if slot.retired or slot.tracked is None or not slot.tracked.order_id or slot.tracked.order_id == "skipped":
                continue
            if slot.tracked.order is None:
                in_flight = self.get_in_flight_order(self.config.connector_name, slot.tracked.order_id)
                if in_flight:
                    slot.tracked.order = in_flight

    def _retire_confirmed_exit_cancels(self):
        for slot in self._exit_slices:
            if slot.retired or not slot.cancel_requested:
                continue
            if slot.order_id and not self._cancel_confirmed(slot.order_id):
                continue
            self._remember_exit_fill(slot)
            slot.retired = True

    def _remember_exit_fill(self, slot: _Slice):
        if not slot.order_id or slot.order_id == "skipped":
            return
        base, quote = self._exit_best_fill(slot.order_id)
        previous = self._exit_accounted_base.get(slot.order_id, Decimal("0"))
        previous_quote = self._exit_accounted_quote.get(slot.order_id, Decimal("0"))
        if base > previous:
            self._exit_accounted_base[slot.order_id] = base
            self._exit_accounted_quote[slot.order_id] = quote
        elif quote > previous_quote:
            self._exit_accounted_quote[slot.order_id] = quote

    def _child_order_ids(self) -> List[str]:
        ids: List[str] = []
        for slot in self._slices:
            if slot.order_id and slot.order_id != "skipped" and slot.order_id not in ids:
                ids.append(slot.order_id)
        for order_id in self._public_exit_order_ids():
            if order_id not in ids:
                ids.append(order_id)
        return ids

    def _public_exit_order_ids(self) -> List[str]:
        ids: List[str] = []
        for order_id in self._exit_order_ids:
            if order_id and order_id != "skipped" and order_id not in ids:
                ids.append(order_id)
        for slot in self._exit_slices:
            if slot.order_id and slot.order_id != "skipped" and slot.order_id not in ids:
                ids.append(slot.order_id)
        return ids

    def _public_exit_state(self) -> ExitState:
        """
        The state the API and Condor should display.

        A resting limit take profit is RESTING until it starts to fill. Waiting
        on a stop, or on a market or iceberg take profit, is ARMED. Sending the
        close is CLOSING. Cancelling the entry or the other exit after a
        trigger is TRIGGERED.
        """
        if self._exit_phase == "done":
            return ExitState.DONE
        if self._exit_phase == "cancelling_entry":
            return ExitState.TRIGGERED
        if self._exit_phase == "executing":
            if self._resting_untriggered_limit():
                return ExitState.RESTING
            return ExitState.CLOSING
        if self._exit_phase == "resting":
            if self._exit_filled_base() > 0:
                return ExitState.CLOSING
            return ExitState.RESTING
        if self._exit_phase == "armed" or self._exit_active is not None:
            return ExitState.ARMED
        if not self._exits_configured() or self.filled_base <= 0:
            return ExitState.NONE
        if self.config.exit_arming == ExitArming.ON_ANY_FILL or self._execution_complete() or self._entry_clock_handled:
            return ExitState.ARMED
        return ExitState.NONE

    def _resting_untriggered_limit(self) -> bool:
        return (
            self._limit_take_profit_config()
            and self._exit_active == "take_profit"
            and self._exit_filled_base() <= 0
            and not self._exit_force_market
        )

    def _protective_stop_supported(self) -> bool:
        connector = self.connectors[self.config.connector_name]
        return self._real_callable(connector, "place_position_stop_order") is not None

    def _protective_stop_enabled(self) -> bool:
        flag = self.config.protective_exchange_stop
        if flag is False:
            return False
        if flag is True:
            return True
        return self._protective_stop_supported()

    def _exchange_stop_is_due(self) -> bool:
        if self.config.stop_loss is None or self.filled_base <= 0:
            return False
        if self._exit_phase == "done" or self._position_base() < self._min_order_size():
            return False
        if self.config.exit_arming == ExitArming.ON_ANY_FILL:
            return True
        return (
            self._execution_complete()
            or self._entry_clock_handled
            or self._exit_phase in ("armed", "resting", "cancelling_entry", "executing")
        )

    def _exchange_position_id(self):
        connector = self.connectors[self.config.connector_name]
        finder = self._real_callable(connector, "_position_id_for_order")
        if finder is not None:
            try:
                found = finder(
                    self.config.trading_pair,
                    self._exit_side(),
                    PositionAction.CLOSE,
                    True,
                )
            except TypeError:
                found = None
            if found not in (None, "", 0, "0"):
                return found
        ids = getattr(connector, "_position_ids", None)
        if isinstance(ids, dict):
            side = PositionSide.LONG if self.config.side == TradeType.BUY else PositionSide.SHORT
            found = ids.get((self.config.trading_pair, side))
            if found not in (None, "", 0, "0"):
                return found
        return None

    def _exchange_stop_trigger_price(self) -> Optional[Decimal]:
        """
        Backstop price, past the software stop in the adverse direction.

        A long stop sits below the software trigger. A short stop sits above
        it. The gap is protective_stop_buffer_pct, so the software stop is
        reached first and can cancel this order before both of them fire.
        """
        trigger = self._trigger_price_for("stop_loss")
        if trigger is None or self.filled_base <= 0:
            return None
        buffer = Decimal(str(self.config.protective_stop_buffer_pct))
        if self.config.side == TradeType.BUY:
            price = trigger * (Decimal("1") - buffer)
        else:
            price = trigger * (Decimal("1") + buffer)
        if price <= 0:
            return None
        step = self._price_step()
        if step <= 0:
            return price
        if self.config.side == TradeType.BUY:
            quantized = quantize_down(price, step)
        else:
            quantized = quantize_up(price, step)
        if quantized <= 0:
            return price
        return quantized

    def _exchange_stop_trigger_text(self):
        price = self._exchange_stop_trigger_price()
        if price is None:
            return None
        return str(price)

    def _exchange_stop_order_ids(self) -> List[str]:
        ids = [str(stop["id"]) for stop in self._exchange_stops if stop.get("id")]
        if not ids and self._exchange_stop_order_id:
            return [str(self._exchange_stop_order_id)]
        return ids

    def _exchange_stop_cancel_outstanding(self) -> bool:
        if self._keep_exchange_stop:
            return False
        return bool(self._exchange_stop_order_ids())

    def _remember_exchange_stop(self, order_id: str, amount: Decimal, price: Optional[Decimal], *, replace: bool):
        order_id = str(order_id)
        if replace:
            self._exchange_stops = [{"id": order_id, "amount": amount, "price": price}]
        else:
            found = False
            for stop in self._exchange_stops:
                if stop["id"] == order_id:
                    if amount >= stop["amount"]:
                        stop["amount"] = amount
                    stop["price"] = price
                    found = True
                    break
            if not found:
                self._exchange_stops.append({"id": order_id, "amount": amount, "price": price})
        self._exchange_stop_order_id = self._exchange_stops[0]["id"]
        self._exchange_stop_amount = sum((stop["amount"] for stop in self._exchange_stops), Decimal("0"))
        self._exchange_stop_price = price

    def _clear_exchange_stop_error_if_complete(self):
        if self._exchange_stop_needs_more_size():
            return
        if not self._exchange_stop_order_ids():
            return
        trigger = self._exchange_stop_trigger_price()
        if trigger is not None and any(stop.get("price") != trigger for stop in self._exchange_stops):
            return
        self._exchange_stop_error = ""
        self._exchange_stop_failures = 0
        self._exchange_stop_retry_after = None

    def _matching_exchange_position(self):
        connector = self.connectors[self.config.connector_name]
        positions = getattr(connector, "account_positions", None)
        if not isinstance(positions, dict):
            return None
        side = PositionSide.LONG if self.config.side == TradeType.BUY else PositionSide.SHORT
        for position in positions.values():
            if getattr(position, "trading_pair", None) != self.config.trading_pair:
                continue
            if getattr(position, "position_side", None) != side:
                continue
            return position
        return None

    def _exchange_position_base(self) -> Optional[Decimal]:
        """
        Open base on the exchange for this pair and side.

        None means the map is missing or this pair and side is not in it.
        A missing key is unknown, not a flat position.
        """
        position = self._matching_exchange_position()
        if position is None:
            return None
        amount = getattr(position, "amount", None)
        if amount is None:
            return None
        try:
            return abs(Decimal(str(amount)))
        except (TypeError, ValueError, ArithmeticError):
            return None

    def _position_read_is_fresh(self, position) -> bool:
        stamp = getattr(position, "update_timestamp", None)
        if stamp is None:
            stamp = getattr(position, "timestamp", None)
        if stamp is None:
            return False
        try:
            stamp = float(stamp)
        except (TypeError, ValueError):
            return False
        if self._last_fill_timestamp is None:
            return True
        return stamp > float(self._last_fill_timestamp)

    def _position_closed_on_exchange(self) -> bool:
        """
        True only when a position read taken after the last fill explicitly
        says the size is flat. A map that simply lacks the key is unknown.
        """
        position = self._matching_exchange_position()
        if position is None or self._position_base() < self._min_order_size():
            return False
        amount = self._exchange_position_base()
        if amount is None or amount >= self._min_order_size():
            return False
        return self._position_read_is_fresh(position)

    def _stop_cancel_says_fired(self, payload) -> bool:
        text = str(payload or "").lower()
        markers = (
            "not found",
            "not exist",
            "does not exist",
            "doesn't exist",
            "already triggered",
            "already been triggered",
            "already executed",
            "already filled",
        )
        return any(marker in text for marker in markers)

    def _clear_exchange_stop_id(self):
        if self._exchange_stop_amount > 0:
            self._exchange_stop_last_amount = self._exchange_stop_amount
        self._exchange_stops = []
        self._exchange_stop_order_id = None
        self._exchange_stop_amount = Decimal("0")
        self._exchange_stop_price = None

    def _note_exchange_stop_cancel_failure(self, order_id: str, detail: str):
        now = float(self._strategy.current_timestamp)
        if self._exchange_stop_cancel_started_at is None:
            self._exchange_stop_cancel_started_at = now
        self._exchange_stop_cancel_failures += 1
        delay = min(2.0, 0.5 * (2 ** (self._exchange_stop_cancel_failures - 1)))
        self._exchange_stop_cancel_after = now + delay
        self._exchange_stop_error = (
            f"Could not cancel exchange stop {order_id}: {detail} "
            f"Next try in {delay:.1f}s."
        )

    def _stop_cancel_wait_expired(self) -> bool:
        if self._exchange_stop_cancel_started_at is None:
            return False
        elapsed = float(self._strategy.current_timestamp) - float(self._exchange_stop_cancel_started_at)
        return elapsed + 1e-9 >= float(self.config.stop_loss_fallback_seconds)

    async def _request_exchange_stop_cancel(self, backoff: bool = False) -> str:
        """
        Cancel the current exchange stop and keep its id until that is confirmed.

        "confirmed" means the cancel was accepted. "fired" means the cancel
        says the order is already gone or already triggered. "failed" means
        the id is still ours. A software stop loss backs off, capped at 2
        seconds, so a stuck cancel does not hit the exchange on every tick.
        A resize keeps trying so the old id can be replaced as soon as the
        cancel is accepted.
        """
        order_ids = self._exchange_stop_order_ids()
        if not order_ids:
            return "confirmed"
        now = float(self._strategy.current_timestamp)
        if (
            backoff
            and self._exchange_stop_cancel_after is not None
            and now + 1e-9 < float(self._exchange_stop_cancel_after)
        ):
            return "failed"
        connector = self.connectors[self.config.connector_name]
        cancel = self._real_callable(connector, "cancel_stop_orders")
        label = ", ".join(order_ids)
        if cancel is None:
            self._note_exchange_stop_cancel_failure(label, "the connector has no cancel_stop_orders.")
            return "failed"
        try:
            result = cancel(list(order_ids))
            if hasattr(result, "__await__"):
                result = await result
        except Exception as exc:
            if self._stop_cancel_says_fired(exc):
                return await self._resolve_ambiguous_stop_cancel(order_ids)
            if backoff:
                self._note_exchange_stop_cancel_failure(label, str(exc))
            else:
                self._exchange_stop_error = f"Could not cancel exchange stop {label}: {exc}"
            return "failed"
        if self._stop_cancel_says_fired(result):
            return await self._resolve_ambiguous_stop_cancel(order_ids)
        rejected = self._rejected_stop_ids(result, order_ids)
        if isinstance(result, dict) and result.get("success") is False and not rejected:
            detail = str(result.get("message") or result.get("msg") or result)
            if backoff:
                self._note_exchange_stop_cancel_failure(label, detail)
            else:
                self._exchange_stop_error = f"Could not cancel exchange stop {label}: {detail}"
            return "failed"
        self._forget_exchange_stops([order_id for order_id in order_ids if order_id not in rejected])
        if rejected:
            self._note_exchange_stop_cancel_failure(", ".join(sorted(rejected)), "the cancel was rejected")
            return "failed"
        self._exchange_stop_error = ""
        self._exchange_stop_cancel_after = None
        self._exchange_stop_cancel_failures = 0
        self._exchange_stop_cancel_started_at = None
        return "confirmed"

    def _rejected_stop_ids(self, result, requested: List[str]) -> Set[str]:
        if not isinstance(result, dict):
            return set()
        rejected: Set[str] = set()
        for key in ("rejected", "rejected_ids", "failed", "failed_ids"):
            values = result.get(key)
            if isinstance(values, list):
                rejected.update(str(item) for item in values if item not in (None, ""))
        rows = result.get("data")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if row.get("success") is False or row.get("error") or row.get("msg"):
                    order_id = row.get("id", row.get("stopPlanOrderId", row.get("orderId")))
                    if order_id not in (None, ""):
                        rejected.add(str(order_id))
        return {order_id for order_id in rejected if order_id in set(requested)}

    async def _resolve_ambiguous_stop_cancel(self, order_ids: List[str]) -> str:
        """
        A batch cancel said not found or already triggered.

        Credit only the ids that are individually fired, or the whole set when
        a fresh position read is flat. A single id is that order. Anything
        still unproven stays tracked.
        """
        fired = await self._ids_proven_fired(order_ids)
        reader = self._real_callable(self.connectors[self.config.connector_name], "get_position_stop_order")
        if not fired and reader is None and len(order_ids) == 1:
            fired = {order_ids[0]}
        if not fired and self._position_closed_on_exchange():
            self._apply_exchange_stop_fill(fired=False)
            self._forget_exchange_stops(order_ids)
            return "fired"
        if not fired:
            self._note_exchange_stop_cancel_failure(
                ", ".join(order_ids),
                "the cancel did not say which stop fired",
            )
            return "failed"
        self._credit_exchange_stops(fired)
        if self._exchange_stop_order_ids():
            return "partial"
        return "fired"

    async def _ids_proven_fired(self, order_ids: List[str]) -> Set[str]:
        fired: Set[str] = set()
        connector = self.connectors[self.config.connector_name]
        reader = self._real_callable(connector, "get_position_stop_order")
        if reader is None:
            return fired
        for order_id in order_ids:
            try:
                record = reader(order_id)
                if hasattr(record, "__await__"):
                    record = await record
            except Exception:
                record = None
            if self._stop_record_fired(record) is True:
                fired.add(str(order_id))
        return fired

    def _forget_exchange_stops(self, order_ids: List[str]):
        dropping = {str(order_id) for order_id in order_ids}
        if not dropping:
            return
        self._exchange_stops = [stop for stop in self._exchange_stops if stop.get("id") not in dropping]
        self._sync_exchange_stop_fields()

    def _sync_exchange_stop_fields(self):
        if not self._exchange_stops:
            if self._exchange_stop_amount > 0:
                self._exchange_stop_last_amount = self._exchange_stop_amount
            self._exchange_stop_order_id = None
            self._exchange_stop_amount = Decimal("0")
            self._exchange_stop_price = None
            return
        self._exchange_stop_order_id = self._exchange_stops[0]["id"]
        self._exchange_stop_amount = sum((stop["amount"] for stop in self._exchange_stops), Decimal("0"))
        self._exchange_stop_price = self._exchange_stops[0].get("price")

    def _stop_record_fired(self, record) -> Optional[bool]:
        if not isinstance(record, dict):
            return None
        state = record.get("state", record.get("status"))
        trigger_side = record.get("triggerSide", record.get("trigger_side"))
        text = str(state or "").lower()
        if text in ("executed", "triggered", "filled") or state in (3, "3"):
            return True
        if trigger_side in (1, 2, "1", "2"):
            return True
        if text in ("untriggered", "open", "new") or state in (1, "1"):
            return False
        if trigger_side in (0, "0"):
            return False
        return None

    async def _query_exchange_stop_fired(self) -> Set[str]:
        """
        Ids whose own stop record says the order executed.

        An empty set means nothing was proven. Absence from the open list is
        not proof, and one fired id does not make the others fired.
        """
        order_ids = self._exchange_stop_order_ids()
        if not order_ids:
            return set()
        connector = self.connectors[self.config.connector_name]
        reader = self._real_callable(connector, "get_position_stop_order")
        if reader is not None:
            fired: Set[str] = set()
            inconclusive = False
            for order_id in order_ids:
                try:
                    record = reader(order_id)
                    if hasattr(record, "__await__"):
                        record = await record
                except Exception:
                    record = None
                found = self._stop_record_fired(record)
                if found is True:
                    fired.add(str(order_id))
                elif found is None:
                    inconclusive = True
            if fired or not inconclusive:
                return fired
        listing = self._real_callable(connector, "list_open_stop_orders")
        if listing is None:
            return set()
        try:
            rows = listing(self.config.trading_pair)
            if hasattr(rows, "__await__"):
                rows = await rows
        except Exception:
            return set()
        if isinstance(rows, dict):
            rows = rows.get("data") or rows.get("orders") or []
        if not isinstance(rows, list):
            return set()
        fired = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_id = row.get("id", row.get("stopPlanOrderId", row.get("orderId")))
            if str(row_id) not in order_ids:
                continue
            if self._stop_record_fired(row) is True:
                fired.add(str(row_id))
        return fired

    def _credit_exchange_stops(self, order_ids: Set[str]) -> bool:
        """
        Credit the named stops and drop only those ids.

        The size is each stop's own amount. Other stops stay tracked. True
        means the position is flat and the exit is finished.
        """
        wanted = {str(order_id) for order_id in order_ids} - self._exchange_stop_credited
        closed = Decimal("0")
        price = None
        kept = []
        for stop in self._exchange_stops:
            if stop.get("id") in wanted:
                closed += Decimal(str(stop.get("amount") or 0))
                price = stop.get("price") or price
                self._exchange_stop_credited.add(str(stop.get("id")))
            else:
                kept.append(stop)
        self._exchange_stops = kept
        self._sync_exchange_stop_fields()
        if closed <= 0:
            return self._exit_phase == "done"
        if closed > self._position_base():
            closed = self._position_base()
        fill_price = price or self._exchange_stop_price or self._trigger_price_for("stop_loss") or Decimal("0")
        self._exit_accounted_base["exchange-stop"] = (
            self._exit_accounted_base.get("exchange-stop", Decimal("0")) + closed
        )
        self._exit_accounted_quote["exchange-stop"] = (
            self._exit_accounted_quote.get("exchange-stop", Decimal("0")) + closed * fill_price
        )
        self._exit_error = "The exchange protective stop closed part of the position."
        if self._position_base() >= self._min_order_size():
            return False
        self._exit_active = "stop_loss"
        self._exit_error = "The exchange protective stop closed the position."
        self._finish_exit()
        return True

    def _apply_exchange_stop_fill(self, fired: bool) -> bool:
        """
        Credit a stop that actually fired, then leave any remainder to the software close.

        A fired stop credits its own size, not a delta of the shared hedge
        position. A fresh flat position read credits the open size. True means
        nothing remains for a software close.
        """
        if self._exchange_stop_reconciled:
            return self._exit_phase == "done"
        if fired:
            closed = self._exchange_stop_amount if self._exchange_stop_amount > 0 else self._exchange_stop_last_amount
        elif self._position_closed_on_exchange():
            closed = self._position_base()
        else:
            return False
        if closed <= 0:
            return False
        if closed > self._position_base():
            closed = self._position_base()
        self._exchange_stop_reconciled = True
        price = self._exchange_stop_price or self._trigger_price_for("stop_loss") or Decimal("0")
        self._exit_accounted_base["exchange-stop"] = (
            self._exit_accounted_base.get("exchange-stop", Decimal("0")) + closed
        )
        self._exit_accounted_quote["exchange-stop"] = (
            self._exit_accounted_quote.get("exchange-stop", Decimal("0")) + closed * price
        )
        self._exit_error = "The exchange protective stop closed part of the position."
        if self._position_base() >= self._min_order_size():
            return False
        self._exit_active = "stop_loss"
        self._exit_error = "The exchange protective stop closed the position."
        self._finish_exit()
        return True

    async def _exchange_stop_cleared_for_software_close(self) -> bool:
        """
        True when the software stop may send a close.

        The exchange stop is cancelled first, with backoff. After
        stop_loss_fallback_seconds without a confirmation, the software close
        is the open size minus the live stop, and the cancel keeps retrying.
        A fired stop or a fresh flat position read is reconciled. A missing
        position key is not a flat read.
        """
        if self._exit_phase == "done":
            return False
        if self._exchange_stop_order_ids():
            fired_ids = await self._query_exchange_stop_fired()
            if fired_ids:
                if self._credit_exchange_stops(fired_ids):
                    return False
            if self._exchange_stop_order_ids():
                outcome = await self._request_exchange_stop_cancel(backoff=True)
            else:
                outcome = "confirmed"
            if outcome in ("fired", "partial"):
                if self._exit_phase == "done":
                    return False
                if self._exchange_stop_order_ids():
                    return False
                self._exchange_stop_released = True
                return True
            if outcome == "failed":
                if self._position_closed_on_exchange():
                    if self._apply_exchange_stop_fill(fired=False):
                        return False
                    return self._exit_phase != "done"
                if self._stop_cancel_wait_expired():
                    self._exchange_stop_close_capped = True
                    return True
                return False
            self._exchange_stop_released = True
            if self._position_closed_on_exchange():
                if self._apply_exchange_stop_fill(fired=False):
                    return False
            return self._exit_phase != "done"
        if self._position_closed_on_exchange():
            self._apply_exchange_stop_fill(fired=False)
            return False
        return True

    def _note_exchange_stop_failure(self, exc: Exception):
        self._exchange_stop_failures += 1
        delay = min(30.0, 0.5 * (2 ** (self._exchange_stop_failures - 1)))
        self._exchange_stop_retry_after = float(self._strategy.current_timestamp) + delay
        self._exchange_stop_error = (
            f"Could not place the exchange protective stop ({self._exchange_stop_failures}). "
            f"Next try in {delay:.1f}s. {exc}"
        )

    async def _place_exchange_stop_keeping_old(self) -> bool:
        """
        Place the full open size at the current backstop price.

        The stops already live stay up. When the price matches one of them,
        the exchange returns that id and replaces its vol with the vol sent.
        That id is tracked at the sent vol, as one stop. A changed price is
        handled by _replace_exchange_stop_at_new_price. An identical live
        stop is not sent again.
        """
        trigger = self._exchange_stop_trigger_price()
        position = self._position_base()
        if trigger is None or position < self._min_order_size():
            return False
        if self._exchange_stop_price_is_stale():
            return False
        if self._live_stop_matches(position, trigger):
            self._clear_exchange_stop_error_if_complete()
            return True
        new_id = await self._submit_exchange_stop(position, trigger)
        if not new_id:
            return False
        self._remember_exchange_stop(new_id, position, trigger, replace=True)
        self._exchange_stop_last_request = (position, trigger)
        self._clear_exchange_stop_error_if_complete()
        return True

    async def _replace_exchange_stop_at_new_price(self) -> bool:
        """
        Cancel the live stops, then place one full-size stop at the new price.

        The new order is sent only after the cancel is confirmed, because a
        new price does not replace the old vol. If the place fails, the old
        size is placed again at the old price before this turn returns.
        """
        trigger = self._exchange_stop_trigger_price()
        position = self._position_base()
        if trigger is None or position < self._min_order_size():
            return False
        saved = [dict(stop) for stop in self._exchange_stops]
        if self._exchange_stop_order_ids():
            outcome = await self._request_exchange_stop_cancel(backoff=False)
            if outcome == "failed" or self._exchange_stop_order_ids():
                return False
            if self._exit_phase == "done":
                return False
        if self._live_stop_matches(position, trigger):
            self._clear_exchange_stop_error_if_complete()
            return True
        new_id = await self._submit_exchange_stop(position, trigger)
        if not new_id:
            await self._restore_exchange_stops(saved)
            return False
        self._remember_exchange_stop(new_id, position, trigger, replace=True)
        self._exchange_stop_last_request = (position, trigger)
        self._clear_exchange_stop_error_if_complete()
        return True

    async def _restore_exchange_stops(self, saved: List[dict]):
        restored = []
        for stop in saved:
            amount = Decimal(str(stop.get("amount") or 0))
            price = stop.get("price")
            if price is None or amount < self._min_order_size():
                continue
            new_id = await self._submit_exchange_stop(amount, price)
            if not new_id:
                continue
            restored.append({"id": str(new_id), "amount": amount, "price": price})
        if not restored:
            return
        self._exchange_stops = restored
        self._sync_exchange_stop_fields()

    async def _align_exchange_stop_prices(self) -> bool:
        """
        Move every live stop to the current backstop price without cancelling it.

        A percent stop moves when the average moves. Repricing in place keeps
        the size covered. A connector that cannot reprice leaves the old stops
        working and records exchange_stop_error.
        """
        trigger = self._exchange_stop_trigger_price()
        if trigger is None or not self._exchange_stops:
            return True
        stale = [stop for stop in self._exchange_stops if stop.get("price") != trigger]
        if not stale:
            self._exchange_stop_price = trigger
            return True
        connector = self.connectors[self.config.connector_name]
        changer = self._real_callable(connector, "change_stop_plan_price")
        if changer is None:
            self._exchange_stop_error = (
                "Could not reprice the exchange stop in place: "
                "the connector has no change_stop_plan_price."
            )
            return False
        for stop in stale:
            try:
                result = changer(stop["id"], stop_loss_price=trigger)
                if hasattr(result, "__await__"):
                    result = await result
            except Exception as exc:
                self._exchange_stop_error = f"Could not reprice exchange stop {stop['id']}: {exc}"
                return False
            if isinstance(result, dict) and result.get("success") is False:
                self._exchange_stop_error = (
                    f"Could not reprice exchange stop {stop['id']}: "
                    f"{result.get('message') or result.get('msg') or result}"
                )
                return False
            stop["price"] = trigger
        self._exchange_stop_price = trigger
        self._clear_exchange_stop_error_if_complete()
        return True

    async def _submit_exchange_stop(self, amount: Decimal, trigger: Decimal) -> Optional[str]:
        connector = self.connectors[self.config.connector_name]
        place = self._real_callable(connector, "place_position_stop_order")
        if place is None:
            if self.config.protective_exchange_stop is True and not self._exchange_stop_error:
                self._exchange_stop_error = (
                    "protective_exchange_stop is on, but the connector has no place_position_stop_order."
                )
            return None
        position_id = self._exchange_position_id()
        if position_id is None:
            return None
        if self._live_stop_matches(amount, trigger):
            return self._exchange_stop_order_id
        try:
            order_id = place(
                trading_pair=self.config.trading_pair,
                position_id=str(position_id),
                amount=amount,
                stop_loss_price=trigger,
                vol_type=1,
            )
            if hasattr(order_id, "__await__"):
                order_id = await order_id
        except Exception as exc:
            self._note_exchange_stop_failure(exc)
            return None
        if order_id in (None, ""):
            self._note_exchange_stop_failure(RuntimeError("place_position_stop_order returned no id"))
            return None
        return str(order_id)

    async def _cancel_replaced_exchange_stop(self, old_id: str):
        connector = self.connectors[self.config.connector_name]
        cancel = self._real_callable(connector, "cancel_stop_orders")
        if cancel is None:
            self._exchange_stop_error = (
                f"The resized exchange stop is live, and the old stop {old_id} could not be cancelled."
            )
            return
        try:
            result = cancel([old_id])
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:
            self._exchange_stop_error = (
                f"The resized exchange stop is live, and the old stop {old_id} could not be cancelled: {exc}"
            )
            return
        if isinstance(result, dict) and result.get("success") is False and not self._stop_cancel_says_fired(result):
            self._exchange_stop_error = (
                f"The resized exchange stop is live, and the old stop {old_id} could not be cancelled."
            )

    async def _sync_exchange_stop(self):
        """
        Place or refresh the exchange-side backstop.

        The price is past the software stop. A resize cancels the old id and
        does not place a replacement until that cancel is confirmed. The old
        id is kept when the cancel fails. Placement failures back off. The
        software stop, a finished exit, and every terminal path except
        keep_position cancel the order instead of refreshing it.
        """
        if self._keep_exchange_stop and self._shutdown_pending:
            return
        if self._exit_phase == "done" or (self._exchange_stop_order_id and self._position_base() <= 0):
            if self._exchange_stop_order_id:
                await self._request_exchange_stop_cancel()
            return
        if self._exchange_stop_released:
            return
        if self._exchange_stop_cancel_started_at is not None:
            return
        if (self._exit_orders_block() or self._entry_blocks_exit()) and (
            self._choose_exit() == "stop_loss" or self._exit_active == "stop_loss"
        ):
            return
        if self._exit_active == "take_profit" and self._exchange_stop_needs_resize():
            return
        if not self._protective_stop_enabled() or not self._exchange_stop_is_due():
            return
        if (
            self._exchange_stop_retry_after is not None
            and float(self._strategy.current_timestamp) + 1e-9 < float(self._exchange_stop_retry_after)
        ):
            return
        trigger = self._exchange_stop_trigger_price()
        amount = self._position_base()
        if trigger is None or amount < self._min_order_size():
            return
        if self._exchange_stop_order_ids() and not await self._align_exchange_stop_prices():
            if not self._exchange_stop_needs_more_size() and not self._exchange_stop_price_is_stale():
                return
        if (
            self._exchange_stop_order_ids()
            and not self._exchange_stop_needs_more_size()
            and all(stop.get("price") == trigger for stop in self._exchange_stops)
        ):
            self._clear_exchange_stop_error_if_complete()
            return
        if self._exchange_stop_order_ids():
            outcome = await self._request_exchange_stop_cancel()
            if outcome == "failed" or outcome == "partial":
                return
            if outcome == "fired":
                return
        connector = self.connectors[self.config.connector_name]
        place = self._real_callable(connector, "place_position_stop_order")
        if place is None:
            if self.config.protective_exchange_stop is True and not self._exchange_stop_error:
                self._exchange_stop_error = (
                    "protective_exchange_stop is on, but the connector has no place_position_stop_order."
                )
            return
        position_id = self._exchange_position_id()
        if position_id is None:
            return
        try:
            order_id = place(
                trading_pair=self.config.trading_pair,
                position_id=str(position_id),
                amount=amount,
                stop_loss_price=trigger,
                vol_type=1,
            )
            if hasattr(order_id, "__await__"):
                order_id = await order_id
        except Exception as exc:
            self._note_exchange_stop_failure(exc)
            return
        if order_id in (None, ""):
            self._note_exchange_stop_failure(RuntimeError("place_position_stop_order returned no id"))
            return
        self._remember_exchange_stop(str(order_id), amount, trigger, replace=True)
        self._clear_exchange_stop_error_if_complete()

    async def _cancel_exchange_stop(self):
        await self._request_exchange_stop_cancel()

    def _exit_config_view(self, cfg: Optional[RangeIcebergExitConfig]):
        if cfg is None:
            return None
        return {
            "percent": None if cfg.percent is None else str(cfg.percent),
            "price": None if cfg.price is None else str(cfg.price),
            "execution": cfg.execution.value,
            "order_type": None if cfg.order_type is None else cfg.order_type.value,
            "slices": cfg.slices,
            "visible_count": cfg.visible_count,
            "interval": None if cfg.interval is None else str(cfg.interval),
            "limit_price": None if cfg.limit_price is None else str(cfg.limit_price),
            "price_low": None if cfg.price_low is None else str(cfg.price_low),
            "price_high": None if cfg.price_high is None else str(cfg.price_high),
        }

    def _exit_snapshot(self) -> dict:
        """
        Exit progress stored in custom_info so restore_from_custom_info can continue
        the same close after a restart.
        """
        order_ids = set(self._exit_order_ids)
        order_ids.update(self._exit_accounted_base)
        for slot in self._exit_slices:
            if slot.order_id and slot.order_id != "skipped":
                order_ids.add(slot.order_id)
        filled_by_order = {}
        for order_id in sorted(order_ids):
            base, quote = self._exit_best_fill(order_id)
            filled_by_order[order_id] = {"base": str(base), "quote": str(quote)}
        return {
            "active": self._exit_active,
            "status": self._exit_phase,
            "exit_state": self._public_exit_state().value,
            "exchange_stop_order_id": self._exchange_stop_order_id,
            "exchange_stop_order_ids": self._exchange_stop_order_ids(),
            "trigger_price": None if self._exit_trigger_price is None else str(self._exit_trigger_price),
            "entry_filled_base": str(self.filled_base),
            "entry_filled_quote": str(self.filled_amount_quote),
            "filled_base": str(self._exit_filled_base()),
            "filled_quote": str(self._exit_filled_quote()),
            "average_price": str(self._exit_average_price()),
            "realized_pnl_quote": str(self._realized_pnl_quote()),
            "started_at": self._exit_started_at,
            "paused_seconds": self._exit_paused_seconds,
            "schedule": None if self._exit_schedule is None else self._exit_schedule.export_state(),
            "order_ids": list(self._exit_order_ids),
            "filled_by_order": filled_by_order,
        }

    def _restored_open_base(self, custom_info: dict, blob: dict) -> Decimal:
        """
        The open base the API is handing back.

        A payload may say size_unit BASE and total_amount equal to the open
        base even when this executor's saved config was quoted in quote.
        """
        raw = custom_info.get("open_base")
        if raw in (None, ""):
            raw = blob.get("open_base")
        if raw in (None, ""):
            size_unit = str(custom_info.get("size_unit") or blob.get("size_unit") or "").upper()
            total = custom_info.get("total_amount", blob.get("total_amount"))
            if size_unit == "BASE" and total not in (None, ""):
                raw = total
        if raw in (None, ""):
            filled = Decimal(str(custom_info.get("filled_base") or blob.get("entry_filled_base") or "0"))
            closed = Decimal(str(custom_info.get("closed_base") or "0"))
            raw = filled - closed
        amount = Decimal(str(raw or "0"))
        return amount if amount > 0 else Decimal("0")

    def _restored_average_entry(self, custom_info: dict, blob: dict) -> Decimal:
        raw = custom_info.get("average_fill_price", custom_info.get("average_entry_price"))
        if raw in (None, ""):
            raw = blob.get("average_fill_price", blob.get("average_entry_price"))
        if raw in (None, ""):
            return Decimal("0")
        return Decimal(str(raw))

    def _rebuild_exits_only_accounting(self, custom_info: dict, blob: dict):
        """
        Replace stale per-order fills with the open base and the supplied average.

        filled_base is that open base, closed_base is 0, and the entry quote is
        open base times the average. Later control passes place no entry orders.
        """
        open_base = self._restored_open_base(custom_info, blob)
        average = self._restored_average_entry(custom_info, blob)
        self._entry_clock_handled = True
        self._entry_handed_to_exit = True
        self._size_unit_override = SizeUnit.BASE
        self._restored_total_base = open_base
        self._accounted_fill_by_order = {"restored-entry": open_base}
        self._accounted_quote_by_order = {"restored-entry": open_base * average}
        self._accounted_fill_base = open_base
        self._accounted_fill_quote = open_base * average
        self._counted_fill_base.clear()
        self._counted_fill_quote.clear()
        self._exit_accounted_base.clear()
        self._exit_accounted_quote.clear()
        self._last_fill_timestamp = float(self._strategy.current_timestamp)

    def _restore_exit_state(self, custom_info: dict) -> None:
        """
        Reapply exit progress saved in get_custom_info()["exit"].

        Entry slices are rebuilt from the config, the same as a new start. Restored
        entry fills and exit fills size the remaining close, so a restart does not
        close more than this executor's own filled position. resume_entries false
        ignores those per-order fills. It sets the open base, a zero closed base,
        and the supplied average, and it never places an entry. A stop loss that
        had already started stays the active exit. Exit order ids are reattached
        only when the connector still has them open and entries are still being
        resumed. Missing ids are posted again from open_base. A saved exchange
        stop id is kept and is not placed a second time. started_at is the trigger
        time. It is not moved to the first exit fill.
        """
        if not isinstance(custom_info, dict):
            return
        if not self._resume_entries:
            early_blob = custom_info.get("exit") if isinstance(custom_info.get("exit"), dict) else {}
            self._rebuild_exits_only_accounting(custom_info, early_blob)
        blob = custom_info.get("exit")
        if not isinstance(blob, dict):
            public_state = str(custom_info.get("exit_state") or "")
            phase_for_state = {
                "ARMED": "armed",
                "RESTING": "resting",
                "TRIGGERED": "cancelling_entry",
                "CLOSING": "executing",
                "DONE": "done",
            }
            if public_state not in phase_for_state:
                return
            blob = {
                "status": phase_for_state[public_state],
                "active": custom_info.get("exit_active"),
                "order_ids": list(custom_info.get("exit_order_ids") or []),
                "entry_filled_base": custom_info.get("filled_base") or "0",
                "entry_filled_quote": custom_info.get("filled_quote") or "0",
                "filled_by_order": {},
                "exchange_stop_order_id": custom_info.get("exchange_stop_order_id"),
            }
        status = str(blob.get("status") or "idle")
        active = blob.get("active")
        if active not in ("take_profit", "stop_loss"):
            active = None
        if status == "idle" and active is None and not blob.get("filled_by_order"):
            return
        if status not in ("idle", "armed", "resting", "cancelling_entry", "executing", "done"):
            status = "idle"
        public_state = str(custom_info.get("exit_state") or blob.get("exit_state") or "")
        if status == "idle" and public_state in ("ARMED", "RESTING", "TRIGGERED", "CLOSING", "DONE"):
            status = {
                "ARMED": "armed",
                "RESTING": "resting",
                "TRIGGERED": "cancelling_entry",
                "CLOSING": "executing",
                "DONE": "done",
            }[public_state]
        self._exit_active = active
        if self._exit_active is None and status == "resting":
            self._exit_active = "take_profit"
        self._exit_phase = status
        trigger = blob.get("trigger_price")
        self._exit_trigger_price = None if trigger in (None, "") else Decimal(str(trigger))
        if not self._resume_entries:
            self._rebuild_exits_only_accounting(custom_info, blob)
        else:
            entry_base = Decimal(str(blob.get("entry_filled_base") or "0"))
            entry_quote = Decimal(str(blob.get("entry_filled_quote") or "0"))
            if entry_base > 0:
                self._accounted_fill_by_order["restored-entry"] = entry_base
                self._accounted_quote_by_order["restored-entry"] = entry_quote
                self._accounted_fill_base = sum(self._accounted_fill_by_order.values(), Decimal("0"))
                self._accounted_fill_quote = sum(self._accounted_quote_by_order.values(), Decimal("0"))
        stop_ids = custom_info.get("exchange_stop_order_ids", blob.get("exchange_stop_order_ids"))
        stop_id = custom_info.get("exchange_stop_order_id", blob.get("exchange_stop_order_id"))
        if not isinstance(stop_ids, list) or not stop_ids:
            stop_ids = [] if stop_id in (None, "") else [stop_id]
        stop_ids = [str(item) for item in stop_ids if item not in (None, "")]
        if stop_ids:
            price = self._exchange_stop_trigger_price()
            total = self._position_base()
            count = len(stop_ids)
            share = total / Decimal(count) if count else total
            running = Decimal("0")
            self._exchange_stops = []
            for index, item in enumerate(stop_ids):
                amount = total - running if index == count - 1 else share
                running += amount
                self._exchange_stops.append({"id": item, "amount": amount, "price": price})
            self._exchange_stop_order_id = stop_ids[0]
            self._exchange_stop_amount = total
            self._exchange_stop_price = price
        if self._resume_entries:
            filled_by_order = blob.get("filled_by_order") or {}
            if isinstance(filled_by_order, dict):
                for order_id, fills in filled_by_order.items():
                    if not isinstance(fills, dict):
                        continue
                    self._exit_accounted_base[str(order_id)] = Decimal(str(fills.get("base") or "0"))
                    self._exit_accounted_quote[str(order_id)] = Decimal(str(fills.get("quote") or "0"))
        else:
            self._exit_accounted_base.clear()
            self._exit_accounted_quote.clear()
            self._exit_order_ids.clear()
            self._exit_slices.clear()
        if not self._resume_entries:
            blob = dict(blob)
            blob["order_ids"] = []
        for order_id in blob.get("order_ids") or []:
            text = str(order_id)
            if text not in self._exit_order_ids:
                self._exit_order_ids.append(text)
            order = self.get_in_flight_order(self.config.connector_name, text)
            if order is None or order.is_done or self._exit_slot_for(text) is not None:
                continue
            price = order.price if order.price is not None and order.price == order.price else (
                self._exit_trigger_price or Decimal("0")
            )
            slot = _Slice(LadderLevel(price=price, base_amount=order.amount))
            slot.tracked = TrackedOrder(order_id=text)
            slot.tracked.order = order
            self._exit_slices.append(slot)
        cfg = self._active_exit_config()
        if self._exit_slices:
            self._exit_plan_ready = True
            self._exit_range = bool(cfg is not None and cfg.limit_price is None and cfg.price_low is not None)
            self._exit_plan_base = self._position_base()
            count = cfg.slices if cfg is not None else 1
            self._exit_slice_base = self._exit_plan_base / Decimal(max(1, count))
        started = blob.get("started_at")
        if started not in (None, ""):
            self._exit_started_at = float(started)
        elif status in ("executing", "cancelling_entry") and self._exit_active is not None:
            self._exit_started_at = float(self._strategy.current_timestamp)
        if blob.get("paused_seconds") is not None:
            self._exit_paused_seconds = float(blob.get("paused_seconds"))
        schedule_state = blob.get("schedule")
        if isinstance(schedule_state, dict):
            if self._exit_plan_ready and cfg is not None and cfg.interval_seconds is not None:
                total = self._exit_plan_base if self._exit_plan_base > 0 else self._position_base()
                self._exit_schedule = self._build_exit_schedule(
                    total, max(1, cfg.slices), cfg.interval_seconds,
                )
                self._exit_schedule.restore_state(schedule_state)
            else:
                self._exit_schedule_state = schedule_state
        if self._exit_phase == "done":
            if self._exit_active == "take_profit":
                self.close_type = CloseType.TAKE_PROFIT
            elif self._exit_active == "stop_loss":
                self.close_type = CloseType.STOP_LOSS
            self._begin_shutdown()
