import json
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import MagicMock, PropertyMock

from pydantic import ValidationError

from hummingbot.connector.derivative.position import Position
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionSide, PriceType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.core.data_type.trade_fee import AddedToCostTradeFee
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import (
    CapBehavior,
    CatchUpPolicy,
    ExecutionMode,
    RangeIcebergExecutorConfig,
    ScheduleStart,
    SizeUnit,
)
from hummingbot.strategy_v2.executors.range_iceberg_executor.range_iceberg_executor import RangeIcebergExecutor
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType


class RangeIcebergExecutorTests(IsolatedAsyncioWrapperTestCase):
    def setUp(self):
        super().setUp()
        self.now = 1_000.0
        self.orders = {}
        self.buy_ids = []
        self.strategy = MagicMock()
        type(self.strategy).current_timestamp = PropertyMock(side_effect=lambda: self.now)
        self.connector = MagicMock()
        self.connector.trading_rules = {
            "AIOT-USDT": TradingRule(
                trading_pair="AIOT-USDT",
                min_order_size=Decimal("10"),
                min_price_increment=Decimal("0.00001"),
                min_base_amount_increment=Decimal("10"),
            )
        }
        self.connector.get_contract_size.return_value = Decimal("10")
        self.connector.get_price_by_type.return_value = Decimal("0.085")
        self.connector._order_tracker.fetch_order.side_effect = lambda client_order_id: self.orders.get(client_order_id)
        self.strategy.connectors = {"mexc_perpetual": self.connector}

        def buy(connector_name, trading_pair, amount, order_type, price, position_action):
            order_id = f"buy-{len(self.buy_ids) + 1}"
            self.buy_ids.append((order_id, amount, price, order_type, position_action))
            self.orders[order_id] = InFlightOrder(
                client_order_id=order_id,
                trading_pair=trading_pair,
                order_type=order_type,
                trade_type=TradeType.BUY,
                amount=amount,
                price=price,
                initial_state=OrderState.OPEN,
                creation_timestamp=self.now,
            )
            return order_id

        self.strategy.buy.side_effect = buy
        self.strategy.sell.side_effect = buy
        self.strategy.cancel.return_value = None

    def _config(self, **overrides) -> RangeIcebergExecutorConfig:
        values = dict(
            timestamp=self.now,
            connector_name="mexc_perpetual",
            trading_pair="AIOT-USDT",
            side=TradeType.BUY,
            total_amount=Decimal("10000"),
            size_unit=SizeUnit.QUOTE,
            price_low=Decimal("0.08"),
            price_high=Decimal("0.09"),
            levels=3,
            iceberg=True,
            visible_count=1,
            min_order_interval=0,
            random_seed=1,
        )
        values.update(overrides)
        return RangeIcebergExecutorConfig(**values)

    def _executor(self, **overrides) -> RangeIcebergExecutor:
        executor = RangeIcebergExecutor(self.strategy, self._config(**overrides), update_interval=0.1)
        executor._status = RunnableStatus.RUNNING
        return executor

    def _run(self, coro):
        return self.run_async_with_timeout(coro, 1)

    def test_iceberg_buy_shows_one_slice_then_replaces_it_inside_the_range(self):
        executor = self._executor()
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        order_id, amount, price, order_type, position_action = self.buy_ids[0]
        self.assertEqual(OrderType.LIMIT, order_type)
        self.assertEqual(PositionAction.OPEN, position_action)
        self.assertGreaterEqual(price, Decimal("0.08"))
        self.assertLessEqual(price, Decimal("0.09"))
        self.assertEqual(0, amount % Decimal("10"))

        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))

        self.orders[order_id].current_state = OrderState.FILLED
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.assertGreater(self.buy_ids[1][2], self.buy_ids[0][2])
        info = executor.get_custom_info()
        self.assertEqual(["buy-1", "buy-2"], info["child_order_ids"])
        self.assertEqual("BUY", info["side"])

    def test_pause_and_resume(self):
        executor = self._executor(iceberg=False, visible_count=3)
        executor.pause()
        self._run(executor.control_task())
        self.assertEqual(0, self.strategy.buy.call_count)
        self.assertTrue(executor.get_custom_info()["paused"])
        self._run(executor.resume())
        self._run(executor.control_task())
        self.assertEqual(1, self.strategy.buy.call_count)

    def test_cancel_parent_cancels_children(self):
        executor = self._executor()
        self._run(executor.control_task())
        executor.early_stop()
        self.strategy.cancel.assert_called_once_with("mexc_perpetual", "AIOT-USDT", "buy-1")
        self.assertEqual(CloseType.EARLY_STOP, executor.close_type)
        self.assertEqual(RunnableStatus.TERMINATED, executor.status)

    def test_stop_reports_a_cancel_that_is_still_open(self):
        executor = self._executor()
        self._run(executor.control_task())
        confirmed = {"ok": False}

        def checker(order_id):
            return confirmed["ok"]

        self.connector.is_cancel_confirmed = checker
        executor.early_stop()
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", "buy-1")
        self.assertEqual(RunnableStatus.RUNNING, executor.status)
        reason = executor.get_custom_info()["status_reason"]
        self.assertIn("buy-1", reason)
        self.assertIn("may still be open", reason)
        self.assertEqual(reason, executor.get_custom_info()["state"])
        self._run(executor.control_task())
        self.assertEqual(RunnableStatus.RUNNING, executor.status)
        self.assertEqual(1, len(self.buy_ids))
        self.assertIn("buy-1", executor.get_custom_info()["status_reason"])
        confirmed["ok"] = True
        self._run(executor.control_task())
        self.assertEqual(RunnableStatus.TERMINATED, executor.status)
        self.assertEqual("", executor.get_custom_info()["status_reason"])

    def test_expiry_cancels_the_working_slice(self):
        executor = self._executor(expiry_seconds=5)
        self._run(executor.control_task())
        self.now = 1_006
        self._run(executor.control_task())
        self.strategy.cancel.assert_called_once()
        self.assertEqual(CloseType.EXPIRED, executor.close_type)

    def test_slippage_guard_does_not_chase_above_the_range(self):
        self.connector.get_price_by_type.side_effect = lambda pair, price_type: (
            Decimal("0.2") if price_type is PriceType.BestAsk else Decimal("0.085")
        )
        executor = self._executor(max_slippage=Decimal("0"))
        self._run(executor.control_task())
        self.assertEqual(0, self.strategy.buy.call_count)

        self.connector.get_price_by_type.side_effect = lambda pair, price_type: Decimal("0.089")
        self._run(executor.control_task())
        self.assertEqual(1, self.strategy.buy.call_count)

    def test_reduce_only_sell_uses_close(self):
        executor = self._executor(side="SELL", reduce_only=True, size_unit=SizeUnit.BASE, total_amount=Decimal("30"))
        self._run(executor.control_task())
        self.assertEqual(PositionAction.CLOSE, self.strategy.sell.call_args.args[5])

    def test_order_spacing_respects_the_place_interval(self):
        executor = self._executor(iceberg=False, visible_count=3, min_order_interval=2)
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(1, self.strategy.buy.call_count)
        self.now += 2
        self._run(executor.control_task())
        self.assertEqual(2, self.strategy.buy.call_count)

    def test_post_only_uses_limit_maker(self):
        executor = self._executor(post_only=True)
        self._run(executor.control_task())
        self.assertEqual(OrderType.LIMIT_MAKER, self.buy_ids[0][3])

    def test_average_fill_uses_child_fills(self):
        executor = self._executor(levels=1, iceberg=False, size_unit=SizeUnit.BASE, total_amount=Decimal("20"))
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        order.executed_amount_base = Decimal("20")
        order.executed_amount_quote = Decimal("1.6")
        order.current_state = OrderState.FILLED
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.08"), executor.average_fill_price)
        self.assertEqual(Decimal("20"), executor.filled_base)
        self.assertEqual(CloseType.COMPLETED, executor.close_type)

    def _book(self, bid: Decimal, ask: Decimal):
        self.connector.get_price_by_type.side_effect = lambda pair, price_type: {
            PriceType.BestBid: bid,
            PriceType.BestAsk: ask,
            PriceType.MidPrice: (bid + ask) / Decimal("2"),
        }.get(price_type, ask)

    def _up_to(self, **overrides) -> RangeIcebergExecutor:
        values = dict(
            timestamp=self.now,
            connector_name="mexc_perpetual",
            trading_pair="AIOT-USDT",
            side=TradeType.BUY,
            total_amount=Decimal("10000"),
            size_unit=SizeUnit.BASE,
            mode=ExecutionMode.UP_TO,
            limit_price=Decimal("0.09"),
            iceberg=True,
            visible_count=1,
            levels=1,
            slice_amount=Decimal("10000"),
            min_order_interval=0,
            random_seed=1,
        )
        values.update(overrides)
        executor = RangeIcebergExecutor(self.strategy, RangeIcebergExecutorConfig(**values), update_interval=0.1)
        executor._status = RunnableStatus.RUNNING
        return executor

    def _fill(self, order_id, base, quote=None, state=OrderState.FILLED):
        order = self.orders[order_id]
        order.executed_amount_base = base
        order.executed_amount_quote = base * order.price if quote is None else quote
        order.current_state = state

    def test_up_to_buy_joins_the_ask_and_rounds_down(self):
        self._book(Decimal("0.08500"), Decimal("0.085019"))
        executor = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"))
        self._run(executor.control_task())
        price = self.buy_ids[0][2]
        self.assertEqual(Decimal("0.08501"), price)
        self.assertLessEqual(price, Decimal("0.09"))
        self.assertEqual(Decimal("100"), self.buy_ids[0][1])

    def test_up_to_rests_at_the_cap_when_the_ask_is_above_it(self):
        self._book(Decimal("0.094"), Decimal("0.095"))
        executor = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"))
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.09"), self.buy_ids[0][2])
        self.assertLessEqual(self.buy_ids[0][2], Decimal("0.09"))

    def test_up_to_waits_above_the_cap_and_buys_when_price_returns(self):
        self._book(Decimal("0.094"), Decimal("0.095"))
        executor = self._up_to(
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
            cap_behavior=CapBehavior.WAIT,
        )
        self._run(executor.control_task())
        self.assertEqual(0, len(self.buy_ids))
        self._book(Decimal("0.087"), Decimal("0.088"))
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.088"), self.buy_ids[0][2])
        self.assertLessEqual(self.buy_ids[0][2], Decimal("0.09"))

    def test_up_to_sell_rests_on_the_floor_and_rounds_up(self):
        self._book(Decimal("0.07"), Decimal("0.071"))
        executor = self._up_to(
            side="SELL",
            limit_price=Decimal("0.08"),
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
        )
        self._run(executor.control_task())
        self.assertGreaterEqual(self.buy_ids[0][2], Decimal("0.08"))
        self.assertEqual(Decimal("0.08"), self.buy_ids[0][2])

        self.buy_ids.clear()
        self._book(Decimal("0.085001"), Decimal("0.086"))
        executor = self._up_to(
            side="SELL",
            limit_price=Decimal("0.08"),
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
            cap_behavior=CapBehavior.WAIT,
        )
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.08501"), self.buy_ids[0][2])
        self.assertGreaterEqual(self.buy_ids[0][2], Decimal("0.08"))

    def test_up_to_post_only_buy_sits_one_tick_above_the_bid(self):
        self._book(Decimal("0.08"), Decimal("0.081"))
        executor = self._up_to(post_only=True, total_amount=Decimal("100"), slice_amount=Decimal("100"))
        self._run(executor.control_task())
        self.assertEqual(OrderType.LIMIT_MAKER, self.buy_ids[0][3])
        self.assertEqual(Decimal("0.08001"), self.buy_ids[0][2])
        self.assertLess(self.buy_ids[0][2], Decimal("0.081"))
        self.assertLessEqual(self.buy_ids[0][2], Decimal("0.09"))

    def test_up_to_reprices_toward_the_ask_and_never_above_the_cap(self):
        self._book(Decimal("0.079"), Decimal("0.08"))
        executor = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"))
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.08"), self.buy_ids[0][2])

        self._book(Decimal("0.084"), Decimal("0.085"))
        self._run(executor.control_task())
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", "buy-1")
        self.orders["buy-1"].current_state = OrderState.CANCELED
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.085"), self.buy_ids[1][2])

        self._book(Decimal("0.19"), Decimal("0.20"))
        self._run(executor.control_task())
        self.orders["buy-2"].current_state = OrderState.CANCELED
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.09"), self.buy_ids[2][2])
        self.assertLessEqual(self.buy_ids[2][2], Decimal("0.09"))

    def test_up_to_partial_fill_keeps_the_fill_and_replaces_the_rest(self):
        self._book(Decimal("0.079"), Decimal("0.08"))
        executor = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"))
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        order.executed_amount_base = Decimal("40")
        order.executed_amount_quote = Decimal("3.2")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._book(Decimal("0.080"), Decimal("0.081"))
        self._run(executor.control_task())
        order.current_state = OrderState.CANCELED
        self._run(executor.control_task())
        self.assertEqual(Decimal("40"), executor.filled_base)
        self.assertEqual(Decimal("0.08"), executor.average_fill_price)
        self.assertEqual(Decimal("60"), self.buy_ids[1][1])
        self.assertEqual(Decimal("0.081"), self.buy_ids[1][2])
        self.assertEqual("60", executor.get_custom_info()["unfilled_amount"])

    def test_up_to_stops_when_the_total_is_filled(self):
        self._book(Decimal("0.079"), Decimal("0.08"))
        executor = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"))
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("8")
        order.current_state = OrderState.FILLED
        self._run(executor.control_task())
        self.assertEqual(CloseType.COMPLETED, executor.close_type)
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual("0", executor.get_custom_info()["unfilled_amount"])
        self.assertEqual("100", executor.get_custom_info()["filled_amount"])

    def test_schedule_releases_each_slice_on_its_interval(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.assertEqual(Decimal("2500"), self.buy_ids[0][1])
        self.orders["buy-1"].executed_amount_base = Decimal("2500")
        self.orders["buy-1"].executed_amount_quote = Decimal("212.5")
        self.orders["buy-1"].current_state = OrderState.FILLED
        self.now = 1_024
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.now = 1_025
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.assertEqual(Decimal("2500"), self.buy_ids[1][1])
        self.assertLessEqual(self.buy_ids[1][2], Decimal("0.09"))

    def test_duration_cancels_working_orders_and_reports_the_remainder(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.now = 1_100
        self._run(executor.control_task())
        self.strategy.cancel.assert_called_once_with("mexc_perpetual", "AIOT-USDT", "buy-1")
        self.assertEqual(CloseType.TIME_LIMIT, executor.close_type)
        info = executor.get_custom_info()
        self.assertEqual("0", info["filled_amount"])
        self.assertEqual("10000", info["unfilled_amount"])
        self.assertEqual("10000", info["target_amount"])

    def test_skip_drops_a_window_that_was_blocked_by_the_cap(self):
        self._book(Decimal("0.10"), Decimal("0.11"))
        executor = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            cap_behavior=CapBehavior.WAIT,
            catch_up=CatchUpPolicy.SKIP,
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.assertEqual(0, len(self.buy_ids))
        self.now = 1_025
        self._run(executor.control_task())
        self.assertEqual(Decimal("2500"), executor._schedule.skipped)
        self._book(Decimal("0.084"), Decimal("0.085"))
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual(Decimal("2500"), self.buy_ids[0][1])
        self.orders["buy-1"].executed_amount_base = Decimal("2500")
        self.orders["buy-1"].executed_amount_quote = Decimal("212.5")
        self.orders["buy-1"].current_state = OrderState.FILLED
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual("2500", executor.get_custom_info()["skipped_amount"])
        self.assertEqual("7500", executor.get_custom_info()["unfilled_amount"])

    def test_carry_adds_a_blocked_window_onto_the_next_slice(self):
        self._book(Decimal("0.10"), Decimal("0.11"))
        executor = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            cap_behavior=CapBehavior.WAIT,
            catch_up=CatchUpPolicy.CARRY,
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.now = 1_025
        self._run(executor.control_task())
        self.assertEqual(Decimal("0"), executor._schedule.skipped)
        self._book(Decimal("0.084"), Decimal("0.085"))
        self._run(executor.control_task())
        self.orders["buy-1"].executed_amount_base = Decimal("2500")
        self.orders["buy-1"].executed_amount_quote = Decimal("212.5")
        self.orders["buy-1"].current_state = OrderState.FILLED
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.assertEqual(Decimal("2500"), self.buy_ids[0][1])
        self.assertEqual(Decimal("2500"), self.buy_ids[1][1])

    def test_pause_freezes_the_schedule_clock(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.orders["buy-1"].executed_amount_base = Decimal("2500")
        self.orders["buy-1"].executed_amount_quote = Decimal("212.5")
        self.orders["buy-1"].current_state = OrderState.FILLED
        executor.pause()
        self.now = 1_030
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertTrue(executor.get_custom_info()["paused"])
        self._run(executor.resume())
        self.now = 1_055
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))

    def test_up_to_sell_waits_below_the_floor(self):
        self._book(Decimal("0.07"), Decimal("0.071"))
        executor = self._up_to(
            side="SELL",
            limit_price=Decimal("0.08"),
            cap_behavior=CapBehavior.WAIT,
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
        )
        self._run(executor.control_task())
        self.assertEqual(0, len(self.buy_ids))
        self._book(Decimal("0.085"), Decimal("0.086"))
        self._run(executor.control_task())
        self.assertGreaterEqual(self.buy_ids[0][2], Decimal("0.08"))

    def test_range_schedule_holds_later_rungs_until_the_clock_releases_them(self):
        executor = self._executor(
            size_unit=SizeUnit.BASE,
            total_amount=Decimal("100"),
            levels=4,
            iceberg=False,
            visible_count=4,
            duration=100,
            schedule_slices=4,
            min_order_interval=0,
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.now = 1_025
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        for _order_id, _amount, price, _order_type, _action in self.buy_ids:
            self.assertGreaterEqual(price, Decimal("0.08"))
            self.assertLessEqual(price, Decimal("0.09"))

    def test_rounding_dust_is_added_to_a_later_slice(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(
            total_amount=Decimal("40"),
            slice_amount=None,
            levels=3,
            duration=30,
            schedule_slices=3,
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.assertEqual(Decimal("10"), self.buy_ids[0][1])
        self.orders["buy-1"].executed_amount_base = Decimal("10")
        self.orders["buy-1"].executed_amount_quote = Decimal("0.85")
        self.orders["buy-1"].current_state = OrderState.FILLED
        self.now = 1_010
        self._run(executor.control_task())
        self.orders["buy-2"].executed_amount_base = Decimal("10")
        self.orders["buy-2"].executed_amount_quote = Decimal("0.85")
        self.orders["buy-2"].current_state = OrderState.FILLED
        self.now = 1_020
        self._run(executor.control_task())
        self.assertEqual(Decimal("20"), self.buy_ids[2][1])
        self.assertEqual(Decimal("40"), sum(amount for _id, amount, _price, _type, _action in self.buy_ids))

    def test_twenty_four_hour_up_to_config(self):
        config = RangeIcebergExecutorConfig(
            timestamp=self.now,
            connector_name="mexc_perpetual",
            trading_pair="AIOT-USDT",
            side="BUY",
            total_amount=Decimal("10000"),
            size_unit=SizeUnit.BASE,
            mode="UP_TO",
            limit_price=Decimal("0.09"),
            iceberg=True,
            visible_count=1,
            duration="24h",
            schedule_slices=24,
            catch_up="CARRY",
            cap_behavior="REST_AT_CAP",
            min_order_interval=0.6,
        )
        self.assertEqual(86400.0, config.duration_seconds)
        count, interval = config.resolved_schedule()
        self.assertEqual(24, count)
        self.assertEqual(3600.0, interval)
        executor = RangeIcebergExecutor(self.strategy, config, update_interval=0.1)
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor._status = RunnableStatus.RUNNING
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertLessEqual(self.buy_ids[0][2], Decimal("0.09"))
        self.assertEqual(0, self.buy_ids[0][1] % Decimal("10"))

    def test_reprice_does_not_replace_a_child_until_the_fill_is_counted(self):
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        self._run(executor.control_task())
        self.assertEqual(Decimal("1000"), self.buy_ids[0][1])
        order = self.orders["buy-1"]
        order.current_state = OrderState.CANCELED
        order.executed_amount_base = Decimal("0")
        order.reported_deal_base = Decimal("340")
        self._book(Decimal("0.07829"), Decimal("0.07830"))
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        order.executed_amount_base = Decimal("340")
        order.executed_amount_quote = Decimal("340") * Decimal("0.07816")
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.assertEqual(Decimal("660"), self.buy_ids[1][1])
        self.assertEqual("660", executor.get_custom_info()["unfilled_amount"])

    def test_filled_child_is_not_replaced_while_its_fill_is_still_missing(self):
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("410"), slice_amount=Decimal("410"))
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        order.current_state = OrderState.FILLED
        order.executed_amount_base = Decimal("0")
        order.reported_deal_base = Decimal("410")
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        order.executed_amount_base = Decimal("410")
        order.executed_amount_quote = Decimal("410") * Decimal("0.07816")
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual(CloseType.COMPLETED, executor.close_type)

    def test_carry_does_not_buy_a_delayed_fill_again(self):
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(
            total_amount=Decimal("1000"),
            slice_amount=Decimal("1000"),
            duration=100,
            schedule_slices=2,
            catch_up=CatchUpPolicy.CARRY,
        )
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        order.current_state = OrderState.CANCELED
        order.executed_amount_base = Decimal("0")
        order.reported_deal_base = Decimal("340")
        self.now = 1_050
        self._book(Decimal("0.07829"), Decimal("0.07830"))
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual(Decimal("0"), executor._schedule.skipped)

    def test_skip_drops_blocked_windows_while_an_older_slice_rests_at_the_cap(self):
        self._book(Decimal("0.094"), Decimal("0.095"))
        executor = self._up_to(
            total_amount=Decimal("10000"),
            duration=86400,
            schedule_slices=24,
            cap_behavior=CapBehavior.REST_AT_CAP,
            catch_up=CatchUpPolicy.SKIP,
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual(Decimal("0.09"), self.buy_ids[0][2])
        self.now = 1_000 + 23 * 3600
        self._run(executor.control_task())
        self.assertGreater(executor._schedule.skipped, Decimal("9000"))
        self.assertLess(executor._placeable(), Decimal("1000"))
        self.assertEqual(1, len(self.buy_ids))

    def test_size_jitter_cannot_push_the_child_over_the_parent_total(self):
        executor = self._executor(
            size_unit=SizeUnit.BASE,
            total_amount=Decimal("100"),
            levels=2,
            iceberg=False,
            visible_count=2,
            size_jitter_pct=Decimal("0.5"),
        )
        executor._rng.random = lambda: 1.0
        self._run(executor.control_task())
        self._run(executor.control_task())
        placed = sum(amount for _order_id, amount, _price, _order_type, _action in self.buy_ids)
        self.assertGreaterEqual(len(self.buy_ids), 1)
        self.assertLessEqual(placed, Decimal("100"))
        for _order_id, amount, _price, _order_type, _action in self.buy_ids:
            self.assertLessEqual(amount, Decimal("100"))

    def test_off_grid_range_fails_and_does_not_trade(self):
        self.connector.trading_rules["AIOT-USDT"] = TradingRule(
            trading_pair="AIOT-USDT",
            min_order_size=Decimal("10"),
            min_price_increment=Decimal("0.0001"),
            min_base_amount_increment=Decimal("10"),
        )
        executor = self._executor(
            size_unit=SizeUnit.BASE,
            total_amount=Decimal("100"),
            price_low=Decimal("0.08001"),
            price_high=Decimal("0.08005"),
            levels=3,
        )
        self.assertEqual(CloseType.FAILED, executor.close_type)
        self._run(executor.control_task())
        self.assertEqual(0, len(self.buy_ids))
        for slot in executor._slices:
            self.assertGreaterEqual(slot.target_price, Decimal("0.08001"))
            self.assertLessEqual(slot.target_price, Decimal("0.08005"))

    def test_exchange_must_confirm_leverage_and_isolated_margin_before_the_first_child(self):
        calls = []

        async def confirm(pair, leverage, mode, position_side=None):
            calls.append((pair, int(leverage), str(getattr(mode, "value", mode)), getattr(position_side, "name", None)))
            return False

        self.connector.confirm_leverage_and_margin = confirm
        executor = self._up_to(
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
            leverage=2,
            margin_mode="ISOLATED",
        )
        self._run(executor.control_task())
        self.assertEqual([("AIOT-USDT", 2, "ISOLATED", "LONG")], calls)
        self.assertEqual(0, len(self.buy_ids))
        self.assertEqual(CloseType.FAILED, executor.close_type)

        calls.clear()
        self.buy_ids.clear()

        async def accept(pair, leverage, mode, position_side=None):
            calls.append((pair, int(leverage), str(getattr(mode, "value", mode)), getattr(position_side, "name", None)))
            return True

        self.connector.confirm_leverage_and_margin = accept
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
            leverage=2,
            margin_mode="ISOLATED",
        )
        self._run(executor.control_task())
        self.assertEqual([("AIOT-USDT", 2, "ISOLATED", "LONG")], calls)
        self.assertEqual(1, len(self.buy_ids))

    def test_executor_waits_while_position_mode_is_still_unread(self):
        ready = {"ok": False}
        calls = []

        def is_ready():
            return ready["ok"]

        async def confirm(pair, leverage, mode, position_side=None):
            calls.append(
                (
                    pair,
                    int(leverage),
                    None if mode is None else str(getattr(mode, "value", mode)),
                    getattr(position_side, "name", None),
                )
            )
            return True

        self.connector.is_position_mode_ready = is_ready
        self.connector.confirm_leverage_and_margin = confirm
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"), leverage=2)
        self._run(executor.control_task())
        self.assertEqual([], calls)
        self.assertEqual(0, len(self.buy_ids))
        self.assertIsNone(executor.close_type)
        self.assertEqual(RunnableStatus.RUNNING, executor.status)
        ready["ok"] = True
        self._run(executor.control_task())
        self.assertEqual([("AIOT-USDT", 2, None, "LONG")], calls)
        self.assertEqual(1, len(self.buy_ids))

    def _held_position(self, side: PositionSide, leverage: int, amount: str = "60770") -> Position:
        return Position(
            trading_pair="AIOT-USDT",
            position_side=side,
            unrealized_pnl=Decimal("0"),
            entry_price=Decimal("0.08"),
            amount=Decimal(amount) if side is PositionSide.LONG else Decimal(amount) * Decimal("-1"),
            leverage=Decimal(leverage),
        )

    def test_margin_mode_defaults_to_none_and_rejects_unknown_values(self):
        self.assertIsNone(self._config().margin_mode)
        with self.assertRaises(Exception):
            self._config(margin_mode="BOTH")

    def test_matching_hedge_long_skips_leverage_and_does_not_touch_the_short(self):
        calls = []

        def set_leverage(pair, leverage, position_side=None):
            calls.append((pair, int(leverage), getattr(position_side, "name", None)))

        self.connector.account_positions = {
            "AIOT-USDTLONG": self._held_position(PositionSide.LONG, 2),
            "AIOT-USDTSHORT": self._held_position(PositionSide.SHORT, 5, "10"),
        }
        self.connector.set_leverage = set_leverage
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"), leverage=2)
        self._run(executor.control_task())
        self.assertEqual([], calls)
        self.assertEqual(1, len(self.buy_ids))
        self.assertIsNone(executor.close_type)

    def test_hedge_short_leverage_is_set_without_a_margin_mode(self):
        calls = []

        def set_leverage(pair, leverage, position_side=None):
            calls.append((pair, int(leverage), getattr(position_side, "name", None)))

        self.connector.account_positions = {
            "AIOT-USDTLONG": self._held_position(PositionSide.LONG, 2),
            "AIOT-USDTSHORT": self._held_position(PositionSide.SHORT, 5, "10"),
        }
        self.connector.set_leverage = set_leverage
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
            leverage=2,
            side=TradeType.SELL,
        )
        self._run(executor.control_task())
        self.assertEqual([("AIOT-USDT", 2, "SHORT")], calls)
        self.assertEqual(1, len(self.buy_ids))

    def test_explicit_margin_switch_rejection_halts_with_the_exchange_reason(self):
        reason = (
            "MEXC rejected switching AIOT-USDT LONG margin from CROSS to ISOLATED "
            "while the position is open: Error: 2027 - There is only one cross or isolated "
            "in the same direction. Refusing to trade."
        )

        async def confirm(pair, leverage, mode, position_side=None):
            return False

        self.connector.confirm_leverage_and_margin = confirm
        self.connector.leverage_confirmation_error = lambda: reason
        self.connector.account_positions = {
            "AIOT-USDTLONG": self._held_position(PositionSide.LONG, 2),
        }
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
            leverage=2,
            margin_mode="ISOLATED",
        )
        self._run(executor.control_task())
        self.assertEqual([], self.buy_ids)
        self.assertEqual(CloseType.FAILED, executor.close_type)
        self.assertEqual(reason, executor.to_format_status())
        self.assertIn("CROSS", executor.get_custom_info()["status_reason"])
        self.assertIn("ISOLATED", executor.get_custom_info()["status_reason"])

    def test_hedge_position_read_uses_the_long_for_a_buy_and_the_short_for_a_sell(self):
        seen = []

        async def exposure(trading_pair, side, position_side=None):
            seen.append((trading_pair, side, getattr(position_side, "name", None)))
            return (Decimal("0"), Decimal("0"))

        self.connector.ground_truth_exposure = exposure
        self.connector.account_positions = {
            "AIOT-USDTLONG": self._held_position(PositionSide.LONG, 2),
            "AIOT-USDTSHORT": self._held_position(PositionSide.SHORT, 2, "10"),
        }
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        buy = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"), leverage=2)
        self._run(buy.control_task())
        sell = self._up_to(
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
            leverage=2,
            side=TradeType.SELL,
        )
        self._run(sell.control_task())
        close_long = self._up_to(
            total_amount=Decimal("100"),
            slice_amount=Decimal("100"),
            leverage=2,
            side=TradeType.SELL,
            reduce_only=True,
        )
        self._run(close_long.control_task())
        self.assertEqual(
            [
                ("AIOT-USDT", TradeType.BUY, "LONG"),
                ("AIOT-USDT", TradeType.SELL, "SHORT"),
                ("AIOT-USDT", TradeType.SELL, "LONG"),
            ],
            seen,
        )

    def test_failed_create_cancels_that_client_id_before_a_replacement(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"))
        self._run(executor.control_task())
        self.cancel_confirmed = False
        self.connector.is_cancel_confirmed = lambda order_id: self.cancel_confirmed
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", "buy-1")
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.cancel_confirmed = True
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))

    def test_failed_create_that_later_fills_is_kept_in_the_tally(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        self.cancel_confirmed = False
        self.connector.is_cancel_confirmed = lambda order_id: self.cancel_confirmed
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        order.executed_amount_base = Decimal("340")
        order.executed_amount_quote = Decimal("26.5744")
        order.current_state = OrderState.FILLED
        self.cancel_confirmed = True
        self._run(executor.control_task())
        self.assertEqual(Decimal("340"), executor.filled_base)
        self.assertEqual(Decimal("26.5744"), executor.filled_amount_quote)
        self.assertEqual(2, len(self.buy_ids))
        self.assertEqual(Decimal("660"), self.buy_ids[1][1])
        self.assertLessEqual(executor.filled_base + self.buy_ids[1][1], Decimal("1000"))

    def test_unresolved_cancel_pauses_and_does_not_buy_again(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 3
        self._run(executor.control_task())
        lookups = []

        async def resolve(trading_pair, client_order_id, exchange_order_id=None):
            lookups.append((trading_pair, client_order_id, exchange_order_id))
            return None

        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = resolve
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual([], lookups)
        self.assertEqual(1, len(self.buy_ids))
        self.assertFalse(executor.get_custom_info()["paused"])
        self._run(executor.control_task())
        self.assertEqual(1, len(lookups))
        self.assertEqual("buy-1", lookups[0][1])
        self.assertTrue(executor.get_custom_info()["paused"])
        self.assertIn("buy-1", executor.get_custom_info()["unknown_order_error"])
        self.now += 10
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))

    def test_cancel_lookup_runs_when_the_time_limit_is_reached(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 100
        executor._cancel_resolve_after_seconds = 30
        self._run(executor.control_task())
        lookups = []

        async def resolve(trading_pair, client_order_id, exchange_order_id=None):
            lookups.append(client_order_id)
            return None

        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = resolve
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertEqual([], lookups)
        self.now += 30
        self._run(executor.control_task())
        self.assertEqual(["buy-1"], lookups)
        self.assertTrue(executor.get_custom_info()["paused"])
        self.assertEqual(1, len(self.buy_ids))

    def test_lookup_that_shows_a_fill_is_kept_and_not_bought_again(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 1
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        order.exchange_order_id = "ex-9"

        async def resolve(trading_pair, client_order_id, exchange_order_id=None):
            self.assertEqual("ex-9", exchange_order_id)
            order.executed_amount_base = Decimal("340")
            order.executed_amount_quote = Decimal("26.5744")
            order.current_state = OrderState.FILLED
            return OrderState.FILLED

        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = resolve
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertEqual(Decimal("340"), executor.filled_base)
        self.assertEqual(Decimal("26.5744"), executor.filled_amount_quote)
        self.assertEqual(2, len(self.buy_ids))
        self.assertEqual(Decimal("660"), self.buy_ids[1][1])
        self.assertFalse(executor.get_custom_info()["paused"])

    def test_open_lookup_retries_then_pauses_with_a_status(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 1
        executor._open_lookup_cap = 3
        executor._open_lookup_backoff_seconds = 10
        executor._open_lookup_backoff_cap_seconds = 10
        executor._open_lookup_max_seconds = 120
        self._run(executor.control_task())
        lookups = []

        async def resolve(trading_pair, client_order_id, exchange_order_id=None):
            lookups.append(self.now)
            return OrderState.OPEN

        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = resolve
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertEqual([self.now], lookups)
        self.assertFalse(executor.get_custom_info()["paused"])
        self.assertEqual(1, len(self.buy_ids))
        self._run(executor.control_task())
        self.assertEqual(1, len(lookups))
        self.now += 10
        self._run(executor.control_task())
        self.assertEqual(2, len(lookups))
        self.assertFalse(executor.get_custom_info()["paused"])
        self.now += 10
        self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertEqual(3, len(lookups))
        self.assertTrue(info["paused"])
        self.assertTrue(info["unknown_order_halt"])
        self.assertEqual(["buy-1"], info["unknown_order_ids"])
        self.assertIn("unknown_order_halt", info["status_reason"])
        self.assertIn("buy-1", info["status_reason"])
        self.assertIn("unknown_order_halt", executor.to_format_status())
        self.now += 100
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))

    def test_open_lookup_pauses_when_the_time_cap_is_reached(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 1
        executor._open_lookup_cap = 10
        executor._open_lookup_backoff_seconds = 1000
        executor._open_lookup_backoff_cap_seconds = 1000
        executor._open_lookup_max_seconds = 120
        self._run(executor.control_task())
        lookups = []

        async def resolve(trading_pair, client_order_id, exchange_order_id=None):
            lookups.append(self.now)
            return OrderState.OPEN

        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = resolve
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertEqual(1, len(lookups))
        self.assertFalse(executor.get_custom_info()["paused"])
        self.now += 120
        self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertEqual(2, len(lookups))
        self.assertTrue(info["paused"])
        self.assertTrue(info["unknown_order_halt"])
        self.assertIn("buy-1", info["status_reason"])
        self.assertEqual(1, len(self.buy_ids))

    def test_resume_looks_up_again_and_stays_paused_while_the_order_is_open(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 1
        self._run(executor.control_task())
        lookups = []
        state = {"value": None}

        async def resolve(trading_pair, client_order_id, exchange_order_id=None):
            lookups.append(client_order_id)
            return state["value"]

        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = resolve
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertTrue(executor.get_custom_info()["paused"])
        state["value"] = OrderState.OPEN
        before = len(lookups)
        self._run(executor.resume())
        info = executor.get_custom_info()
        self.assertGreater(len(lookups), before)
        self.assertTrue(info["paused"])
        self.assertTrue(info["unknown_order_halt"])
        self.assertIn("unknown_order_halt", info["status_reason"])
        self.assertIn("buy-1", info["status_reason"])
        self.now += 30
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))

    def test_resume_clears_the_halt_when_the_order_is_filled_or_canceled(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 1
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        state = {"value": None}

        async def resolve(trading_pair, client_order_id, exchange_order_id=None):
            if state["value"] == OrderState.FILLED:
                order.executed_amount_base = Decimal("340")
                order.executed_amount_quote = Decimal("26.5744")
                order.current_state = OrderState.FILLED
            return state["value"]

        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = resolve
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertTrue(executor.get_custom_info()["paused"])
        state["value"] = OrderState.FILLED
        self._run(executor.resume())
        info = executor.get_custom_info()
        self.assertFalse(info["paused"])
        self.assertFalse(info["unknown_order_halt"])
        self._run(executor.control_task())
        self.assertEqual(Decimal("340"), executor.filled_base)
        self.assertEqual(Decimal("26.5744"), executor.filled_amount_quote)
        self.assertEqual(2, len(self.buy_ids))
        self.assertEqual(Decimal("660"), self.buy_ids[1][1])

    def test_resume_clears_the_halt_when_the_order_is_canceled_or_absent(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 1
        self._run(executor.control_task())
        state = {"value": None}

        async def resolve(trading_pair, client_order_id, exchange_order_id=None):
            return state["value"]

        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = resolve
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        state["value"] = OrderState.CANCELED
        self._run(executor.resume())
        self.assertFalse(executor.get_custom_info()["unknown_order_halt"])
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.assertEqual(Decimal("1000"), self.buy_ids[1][1])

        from bidict import bidict

        from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative

        exchange = MexcPerpetualDerivative(
            mexc_perpetual_api_key="key",
            mexc_perpetual_api_secret="secret",
            trading_pairs=["AIOT-USDT"],
            trading_required=True,
        )
        exchange._set_trading_pair_symbol_map(bidict({"AIOT_USDT": "AIOT-USDT"}))
        exchange._set_current_timestamp(self.now)
        read = {"kind": "absent"}

        async def api_get(path_url, **kwargs):
            if read["kind"] == "absent":
                return {"success": False, "code": 2040, "message": "Order does not exist", "data": None}
            raise IOError(f"Error executing request GET https://api.mexc.com{path_url}. HTTP status is 500.")

        exchange._api_get = api_get
        self.connector.resolve_unknown_order = exchange.resolve_unknown_order
        executor._unknown_order_ids.add("buy-1")
        executor._unconfirmed_cancels.add("buy-1")
        executor.pause()
        self._run(executor.resume())
        self.assertTrue(executor.get_custom_info()["paused"])
        self.assertTrue(executor.get_custom_info()["unknown_order_halt"])
        self.assertEqual(2, len(self.buy_ids))

        read["kind"] = "http500"
        executor._unknown_order_ids.add("buy-1")
        executor._unconfirmed_cancels.add("buy-1")
        executor.pause()
        self._run(executor.resume())
        info = executor.get_custom_info()
        self.assertTrue(info["paused"])
        self.assertTrue(info["unknown_order_halt"])
        self.assertIn("buy-1", info["status_reason"])
        self.assertEqual(2, len(self.buy_ids))

    def test_conflicting_canceled_and_open_reads_do_not_buy_the_slice_again(self):
        from bidict import bidict

        from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 1
        executor._open_lookup_backoff_seconds = 10
        executor._open_lookup_backoff_cap_seconds = 10
        executor._open_lookup_cap = 3
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        order.exchange_order_id = "ex-1"
        exchange = MexcPerpetualDerivative(
            mexc_perpetual_api_key="key",
            mexc_perpetual_api_secret="secret",
            trading_pairs=["AIOT-USDT"],
            trading_required=True,
        )
        exchange._set_trading_pair_symbol_map(bidict({"AIOT_USDT": "AIOT-USDT"}))
        exchange._set_current_timestamp(self.now)
        calls = {"n": 0}

        async def api_get(path_url, **kwargs):
            if "external" in path_url:
                payload = {
                    "orderId": "ex-1",
                    "externalOid": "buy-1",
                    "state": 4,
                    "dealVol": 0,
                    "price": "0.07816",
                    "vol": "100",
                }
            else:
                payload = {
                    "orderId": "ex-1",
                    "externalOid": "buy-1",
                    "state": 2,
                    "dealVol": 0,
                    "price": "0.07816",
                    "vol": "100",
                }
            return {"success": True, "code": 0, "data": payload}

        exchange._api_get = api_get

        async def resolve(trading_pair, client_order_id, exchange_order_id=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return OrderState.OPEN
            return await exchange.resolve_unknown_order(
                trading_pair,
                client_order_id,
                exchange_order_id or "ex-1",
            )

        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = resolve
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual(Decimal("1000"), self.buy_ids[0][1])
        self.assertFalse(executor.get_custom_info()["paused"])
        self.now += 10
        self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertEqual(1, len(self.buy_ids))
        self.assertTrue(info["paused"])
        self.assertTrue(info["unknown_order_halt"])
        self.assertIn("buy-1", info["status_reason"])
        self.assertEqual(OrderState.OPEN, order.current_state)
        self.assertEqual(Decimal("0"), order.executed_amount_base)

    def test_status_poll_2040_halts_instead_of_buying_the_slice_again(self):
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        self._run(executor.control_task())
        self.connector.orders_awaiting_resolution = lambda: {"buy-1"}
        self.connector.is_cancel_confirmed = lambda order_id: True
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertTrue(executor.get_custom_info()["paused"])
        self.assertTrue(executor.get_custom_info()["unknown_order_halt"])
        self.assertEqual(Decimal("0"), executor.filled_base)

    def test_ground_truth_caps_the_child_and_stops_when_the_query_fails(self):
        self._book(Decimal("0.07815"), Decimal("0.07816"))

        async def exposure(trading_pair, side):
            return (Decimal("0"), Decimal("200"))

        self.connector.ground_truth_exposure = exposure
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        self._run(executor.control_task())
        self.assertEqual(Decimal("800"), self.buy_ids[0][1])

        async def grown(trading_pair, side):
            return (Decimal("800"), Decimal("0"))

        self.connector.ground_truth_exposure = grown
        self.orders["buy-1"].current_state = OrderState.CANCELED
        self.orders["buy-1"].executed_amount_base = Decimal("0")
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.assertEqual(Decimal("200"), self.buy_ids[1][1])

        async def broken(trading_pair, side):
            raise IOError("position read failed")

        self.orders["buy-2"].current_state = OrderState.CANCELED
        self.connector.ground_truth_exposure = broken
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))

    def test_open_order_page_failure_sends_no_child(self):
        from bidict import bidict

        from hummingbot.connector.derivative.mexc_perpetual import mexc_perpetual_constants as CONSTANTS
        from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        exchange = MexcPerpetualDerivative(
            mexc_perpetual_api_key="key",
            mexc_perpetual_api_secret="secret",
            trading_pairs=["AIOT-USDT"],
            trading_required=True,
        )
        exchange._set_trading_pair_symbol_map(bidict({"AIOT_USDT": "AIOT-USDT"}))
        exchange._contract_sizes["AIOT-USDT"] = Decimal("10")
        previous_size = CONSTANTS.OPEN_ORDERS_PAGE_SIZE
        CONSTANTS.OPEN_ORDERS_PAGE_SIZE = 1

        async def get(path_url, params=None, **kwargs):
            if path_url == CONSTANTS.OPEN_POSITIONS_PATH:
                return {"success": True, "code": 0, "data": []}
            page = int((params or {}).get("page_num") or 0)
            if path_url == CONSTANTS.OPEN_ORDERS_PATH and page == 1:
                return {
                    "success": True,
                    "code": 0,
                    "data": [{"symbol": "AIOT_USDT", "side": 1, "vol": 1, "dealVol": 0}],
                }
            if path_url == CONSTANTS.OPEN_ORDERS_PATH and page >= 2:
                raise IOError("open order page failed")
            return {"success": True, "code": 0, "data": []}

        exchange._api_get = get
        self.connector.ground_truth_exposure = exchange.ground_truth_exposure
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        try:
            self._run(executor.control_task())
        finally:
            CONSTANTS.OPEN_ORDERS_PAGE_SIZE = previous_size
        self.assertEqual([], self.buy_ids)

    def test_websocket_open_then_empty_lookup_does_not_buy_again(self):
        import json

        from aioresponses import aioresponses
        from bidict import bidict

        from hummingbot.connector.derivative.mexc_perpetual import mexc_perpetual_constants as CONSTANTS
        from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative
        from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_web_utils import private_rest_url
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 1
        self._run(executor.control_task())
        exchange = MexcPerpetualDerivative(
            mexc_perpetual_api_key="key",
            mexc_perpetual_api_secret="secret",
            trading_pairs=["AIOT-USDT"],
            trading_required=True,
        )
        exchange._set_trading_pair_symbol_map(bidict({"AIOT_USDT": "AIOT-USDT"}))
        exchange._set_current_timestamp(self.now)
        exchange._contract_sizes["AIOT-USDT"] = Decimal("10")
        exchange.start_tracking_order(
            order_id="buy-1",
            exchange_order_id="ex-1",
            trading_pair="AIOT-USDT",
            trade_type=TradeType.BUY,
            price=Decimal("0.07816"),
            amount=Decimal("1000"),
            order_type=OrderType.LIMIT,
            position_action=PositionAction.OPEN,
        )
        exchange._process_order_event_message(
            {
                "orderId": "ex-1",
                "externalOid": "buy-1",
                "state": 2,
                "dealVol": 0,
                "price": "0.07816",
                "vol": "100",
            }
        )
        client_url = private_rest_url(CONSTANTS.ORDER_BY_EXTERNAL_PATH.format(symbol="AIOT_USDT", external_oid="buy-1"))
        order_url = private_rest_url(CONSTANTS.ORDER_GET_PATH.format(order_id="ex-1"))
        body = json.dumps({"success": True, "code": 0, "data": None})
        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = exchange.resolve_unknown_order
        self.connector.counted_fill = exchange.counted_fill
        self.connector.order_was_seen_live = exchange.order_was_seen_live
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        with aioresponses() as mocked:
            mocked.get(client_url, body=body, repeat=True)
            mocked.get(order_url, body=body, repeat=True)
            self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertTrue(executor.get_custom_info()["paused"] or executor.get_custom_info()["unknown_order_halt"])
        self.assertLessEqual(sum(amount for _oid, amount, _price, _otype, _action in self.buy_ids), Decimal("1000"))

    def test_conflicting_fill_is_kept_and_a_later_2040_does_not_buy_again(self):
        from bidict import bidict

        from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative
        from hummingbot.core.event.events import MarketOrderFailureEvent

        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("1000"), slice_amount=Decimal("1000"))
        executor._cancel_resolve_after_attempts = 1
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        order.exchange_order_id = "ex-1"
        exchange = MexcPerpetualDerivative(
            mexc_perpetual_api_key="key",
            mexc_perpetual_api_secret="secret",
            trading_pairs=["AIOT-USDT"],
            trading_required=True,
        )
        exchange._set_trading_pair_symbol_map(bidict({"AIOT_USDT": "AIOT-USDT"}))
        exchange._set_current_timestamp(self.now)
        exchange._contract_sizes["AIOT-USDT"] = Decimal("10")
        exchange._order_tracker.TRADE_FILLS_WAIT_TIMEOUT = 0.01
        exchange.start_tracking_order(
            order_id="buy-1",
            exchange_order_id="ex-1",
            trading_pair="AIOT-USDT",
            trade_type=TradeType.BUY,
            price=order.price,
            amount=Decimal("1000"),
            order_type=OrderType.LIMIT,
            position_action=PositionAction.OPEN,
        )
        phase = {"step": "conflict"}

        async def api_get(path_url, **kwargs):
            if phase["step"] == "absent":
                return {"success": False, "code": 2040, "message": "Order does not exist", "data": None}
            if "external" in path_url:
                payload = {
                    "orderId": "ex-1",
                    "externalOid": "buy-1",
                    "state": 3,
                    "dealVol": "100",
                    "dealAvgPrice": "0.07816",
                    "price": "0.07816",
                    "vol": "100",
                }
            else:
                payload = {
                    "orderId": "ex-1",
                    "externalOid": "buy-1",
                    "state": 2,
                    "dealVol": 0,
                    "price": "0.07816",
                    "vol": "100",
                }
            return {"success": True, "code": 0, "data": payload}

        exchange._api_get = api_get
        self.connector.is_cancel_confirmed = lambda order_id: False
        self.connector.resolve_unknown_order = exchange.resolve_unknown_order
        self.connector.counted_fill = exchange.counted_fill
        self.connector.order_was_seen_live = exchange.order_was_seen_live
        executor.process_order_failed_event(
            0,
            None,
            MarketOrderFailureEvent(self.now, "buy-1", OrderType.LIMIT),
        )
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual(Decimal("1000"), executor.filled_base)
        self.assertEqual(Decimal("78.16"), executor.filled_amount_quote)
        self.assertTrue(executor.get_custom_info()["paused"])
        phase["step"] = "absent"
        self._run(executor.resume())
        self._run(executor.control_task())
        self.assertTrue(executor.get_custom_info()["paused"])
        self.assertTrue(executor.get_custom_info()["unknown_order_halt"])
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual(Decimal("1000"), executor.filled_base)
        self.assertLessEqual(sum(amount for _oid, amount, _price, _otype, _action in self.buy_ids), Decimal("1000"))

    def test_stop_cancels_a_child_that_is_not_tracked_yet(self):
        self._book(Decimal("0.07815"), Decimal("0.07816"))
        executor = self._up_to(total_amount=Decimal("100"), slice_amount=Decimal("100"))
        self._run(executor.control_task())
        self.orders.clear()
        executor._slices[0].tracked.order = None
        executor.early_stop()
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", "buy-1")

    def test_rounding_remainder_is_tracked_and_folded_without_exceeding_the_total(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(
            total_amount=Decimal("10000"),
            slice_amount=None,
            levels=24,
            duration=86400,
            schedule_slices=24,
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.assertEqual(Decimal("410"), self.buy_ids[0][1])
        info = executor.get_custom_info()
        self.assertIn("rounding_remainder", info)
        self.assertEqual(Decimal("10000") / Decimal("24") - Decimal("410"), Decimal(info["rounding_remainder"]))
        placed = Decimal("0")
        for index in range(23):
            order_id = self.buy_ids[-1][0]
            self.orders[order_id].executed_amount_base = self.buy_ids[-1][1]
            self.orders[order_id].executed_amount_quote = self.buy_ids[-1][1] * Decimal("0.085")
            self.orders[order_id].current_state = OrderState.FILLED
            self.now = 1_000 + (index + 1) * 3600
            self._run(executor.control_task())
            placed = sum(amount for _order_id, amount, _price, _order_type, _action in self.buy_ids)
            self.assertLessEqual(placed, Decimal("10000"))
        remainder = Decimal(executor.get_custom_info()["rounding_remainder"])
        self.assertLess(remainder, Decimal("10"))
        self.assertEqual(Decimal("10000"), placed + remainder)

    def test_schedule_start_defaults_to_first_fill(self):
        config = self._config(duration=100, schedule_slices=2)
        self.assertEqual(ScheduleStart.FIRST_FILL, config.schedule_start)
        with self.assertRaises(ValidationError) as caught:
            self._config(duration=100, schedule_slices=2, schedule_start="LATER")
        self.assertIn("FIRST_FILL", str(caught.exception))

    def test_no_schedule_time_passes_before_the_first_fill(self):
        self._book(Decimal("0.094"), Decimal("0.095"))
        executor = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            iceberg=True,
            visible_count=2,
        )
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertEqual(Decimal("2500"), self.buy_ids[0][1])
        self.assertEqual(Decimal("0.09"), self.buy_ids[0][2])
        self.now = 1_000 + 10_000
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertIsNone(executor.close_type)
        info = executor.get_custom_info()
        self.assertIsNone(info["schedule_started_at"])
        self.assertEqual(100.0, info["schedule_time_remaining"])
        self.assertEqual(Decimal("0"), executor._schedule.skipped)
        self.assertEqual(0, executor._schedule._closed)

    def test_partial_fill_anchors_the_schedule_at_the_observation_time(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
        self._run(executor.control_task())
        self.now = 2_500
        self._run(executor.control_task())
        self.assertIsNone(executor.get_custom_info()["schedule_started_at"])
        self._fill("buy-1", Decimal("40"), Decimal("3.4"), OrderState.PARTIALLY_FILLED)
        self._run(executor.control_task())
        self.assertEqual(2_500, executor.get_custom_info()["schedule_started_at"])
        self.assertEqual(100.0, executor.get_custom_info()["schedule_time_remaining"])
        self.assertIsNone(executor.close_type)
        self.now = 2_510
        self._fill("buy-1", Decimal("80"), Decimal("6.8"), OrderState.PARTIALLY_FILLED)
        self._run(executor.control_task())
        self.assertEqual(2_500, executor.get_custom_info()["schedule_started_at"])
        self.assertEqual(1, len(self.buy_ids))

    def test_fill_event_anchors_on_the_event_timestamp(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
        self._run(executor.control_task())
        self.now = 4_000
        executor.process_order_filled_event(
            0,
            None,
            OrderFilledEvent(
                timestamp=4_000,
                order_id="buy-1",
                trading_pair="AIOT-USDT",
                trade_type=TradeType.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("0.085"),
                amount=Decimal("0"),
                trade_fee=AddedToCostTradeFee(percent=Decimal("0")),
            ),
        )
        self.assertIsNone(executor.get_custom_info()["schedule_started_at"])
        self._fill("buy-1", Decimal("40"), Decimal("3.4"), OrderState.PARTIALLY_FILLED)
        executor.process_order_filled_event(
            0,
            None,
            OrderFilledEvent(
                timestamp=3_980,
                order_id="buy-1",
                trading_pair="AIOT-USDT",
                trade_type=TradeType.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("0.085"),
                amount=Decimal("40"),
                trade_fee=AddedToCostTradeFee(percent=Decimal("0")),
            ),
        )
        self.assertEqual(3_980, executor.get_custom_info()["schedule_started_at"])
        self._run(executor.control_task())
        self.assertEqual(3_980, executor.get_custom_info()["schedule_started_at"])
        self.assertAlmostEqual(80.0, executor.get_custom_info()["schedule_time_remaining"])
        self.assertIsNone(executor.close_type)

    def test_later_slices_release_from_the_anchor(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
        self._run(executor.control_task())
        self.now = 5_000
        self._fill("buy-1", Decimal("2500"), Decimal("212.5"))
        self._run(executor.control_task())
        self.assertEqual(5_000, executor.get_custom_info()["schedule_started_at"])
        self.assertEqual(1, len(self.buy_ids))
        self.now = 5_024
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.now = 5_025
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.assertEqual(Decimal("2500"), self.buy_ids[1][1])
        self.assertLessEqual(self.buy_ids[1][2], Decimal("0.09"))

    def test_range_rungs_release_from_the_anchor(self):
        executor = self._executor(
            size_unit=SizeUnit.BASE,
            total_amount=Decimal("100"),
            levels=4,
            iceberg=False,
            visible_count=4,
            duration=100,
            schedule_slices=4,
            min_order_interval=0,
        )
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.now = 3_000
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        for _order_id, _amount, price, _order_type, _action in self.buy_ids:
            self.assertGreaterEqual(price, Decimal("0.08"))
            self.assertLessEqual(price, Decimal("0.09"))
        self._fill("buy-1", self.buy_ids[0][1])
        self._run(executor.control_task())
        anchor = self.now
        self.assertEqual(anchor, executor.get_custom_info()["schedule_started_at"])
        self.now = anchor + 24
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.now = anchor + 25
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.assertGreaterEqual(self.buy_ids[1][2], Decimal("0.08"))
        self.assertLessEqual(self.buy_ids[1][2], Decimal("0.09"))

    def test_time_limit_is_measured_from_the_anchor(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
        self._run(executor.control_task())
        self.now = 5_000
        self._run(executor.control_task())
        self.assertIsNone(executor.close_type)
        self._fill("buy-1", Decimal("40"), Decimal("3.4"), OrderState.PARTIALLY_FILLED)
        self._run(executor.control_task())
        self.assertEqual(5_000, executor.get_custom_info()["schedule_started_at"])
        self.assertIsNone(executor.close_type)
        self.now = 5_099
        self._run(executor.control_task())
        self.assertIsNone(executor.close_type)
        self.now = 5_100
        self._run(executor.control_task())
        self.assertEqual(CloseType.TIME_LIMIT, executor.close_type)
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", "buy-1")

    def test_skip_and_carry_are_measured_from_the_anchor(self):
        self._book(Decimal("0.10"), Decimal("0.11"))
        skip = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            cap_behavior=CapBehavior.WAIT,
            catch_up=CatchUpPolicy.SKIP,
        )
        self._run(skip.control_task())
        self.now = 5_000
        self._run(skip.control_task())
        self.assertEqual(0, len(self.buy_ids))
        self.assertEqual(Decimal("0"), skip._schedule.skipped)
        self.assertIsNone(skip.get_custom_info()["schedule_started_at"])
        self._book(Decimal("0.084"), Decimal("0.085"))
        self._run(skip.control_task())
        self._fill("buy-1", Decimal("2500"), Decimal("212.5"))
        self._run(skip.control_task())
        self.assertEqual(5_000, skip.get_custom_info()["schedule_started_at"])
        self.assertEqual(Decimal("0"), skip._schedule.skipped)
        self._book(Decimal("0.10"), Decimal("0.11"))
        self.now = 5_050
        self._run(skip.control_task())
        self.assertEqual(Decimal("2500"), skip._schedule.skipped)
        self.assertEqual(Decimal("2500"), skip._placeable())
        self.assertIsNone(skip.close_type)

        self.buy_ids.clear()
        self.now = 1_000
        carry = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            cap_behavior=CapBehavior.WAIT,
            catch_up=CatchUpPolicy.CARRY,
        )
        self._book(Decimal("0.084"), Decimal("0.085"))
        self._run(carry.control_task())
        self.now = 5_000
        self._fill("buy-1", Decimal("2500"), Decimal("212.5"))
        self._run(carry.control_task())
        self._book(Decimal("0.10"), Decimal("0.11"))
        self.now = 5_050
        self._run(carry.control_task())
        self.assertEqual(Decimal("0"), carry._schedule.skipped)
        self.assertEqual(Decimal("5000"), carry._placeable())
        self.assertEqual(5_000, carry.get_custom_info()["schedule_started_at"])

    def test_expiry_stays_on_the_executor_start(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(
            duration=1_000,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            expiry_seconds=5,
        )
        self._run(executor.control_task())
        self.now = 1_005
        self._run(executor.control_task())
        self.assertEqual(CloseType.EXPIRED, executor.close_type)
        self.assertIsNone(executor.get_custom_info()["schedule_started_at"])
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", "buy-1")

    def test_submit_starts_the_clock_at_executor_start(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            schedule_start=ScheduleStart.SUBMIT,
        )
        self._run(executor.control_task())
        self.assertEqual(1_000, executor.get_custom_info()["schedule_started_at"])
        self.now = 1_100
        self._run(executor.control_task())
        self.assertEqual(CloseType.TIME_LIMIT, executor.close_type)

    def test_unscheduled_execution_does_not_wait_for_a_fill(self):
        executor = self._executor(
            size_unit=SizeUnit.BASE,
            total_amount=Decimal("100"),
            levels=3,
            iceberg=False,
            visible_count=3,
            min_order_interval=0,
        )
        self._run(executor.control_task())
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(3, len(self.buy_ids))
        self.now = 10_000
        self._run(executor.control_task())
        self.assertIsNone(executor.close_type)
        info = executor.get_custom_info()
        self.assertIsNone(info["schedule_started_at"])
        self.assertIsNone(info["schedule_time_remaining"])
        self.assertIsNone(info["schedule_clock"])

    def test_pause_after_the_anchor_freezes_the_schedule(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
        self._run(executor.control_task())
        self._fill("buy-1", Decimal("2500"), Decimal("212.5"))
        self._run(executor.control_task())
        self.assertEqual(1_000, executor.get_custom_info()["schedule_started_at"])
        self.now = 1_010
        executor.pause()
        self.now = 1_040
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertTrue(executor.get_custom_info()["paused"])
        self._run(executor.resume())
        self.now = 1_054
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.now = 1_055
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))

    def test_pause_before_the_anchor_does_not_consume_the_schedule(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
        self._run(executor.control_task())
        executor.pause()
        self.now = 1_100
        self._run(executor.control_task())
        self.assertIsNone(executor.get_custom_info()["schedule_started_at"])
        self.assertEqual(1, len(self.buy_ids))
        self._run(executor.resume())
        self._fill("buy-1", Decimal("2500"), Decimal("212.5"))
        self._run(executor.control_task())
        self.assertEqual(1_100, executor.get_custom_info()["schedule_started_at"])
        self.now = 1_124
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.now = 1_125
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))

    def test_first_slice_still_reprices_before_the_fill(self):
        self._book(Decimal("0.079"), Decimal("0.08"))
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.08"), self.buy_ids[0][2])
        self.now = 1_500
        self._book(Decimal("0.084"), Decimal("0.085"))
        self._run(executor.control_task())
        self.orders["buy-1"].current_state = OrderState.CANCELED
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.085"), self.buy_ids[1][2])
        self.assertEqual(Decimal("2500"), self.buy_ids[1][1])
        self.assertLessEqual(self.buy_ids[1][2], Decimal("0.09"))
        self.assertIsNone(executor.get_custom_info()["schedule_started_at"])
        self.assertEqual(2, len(self.buy_ids))

    def test_schedule_anchor_survives_serialization_and_resume(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            expiry_seconds=4_000,
        )
        self._run(executor.control_task())
        self.now = 5_000
        self._fill("buy-1", Decimal("2500"), Decimal("212.5"))
        self._run(executor.control_task())
        payload = json.loads(executor.executor_info.model_dump_json())
        self.assertEqual(5_000, payload["custom_info"]["schedule_started_at"])
        self.now = 5_024
        resumed = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            expiry_seconds=4_000,
        )
        self.assertIsNone(resumed.get_custom_info()["schedule_started_at"])
        self.assertFalse(resumed._expired())
        self.assertFalse(resumed._time_limit_reached())
        resumed.restore_from_custom_info(payload["custom_info"])
        self.assertEqual(5_000, resumed.get_custom_info()["schedule_started_at"])
        self.assertAlmostEqual(24.0, resumed._schedule_elapsed())
        self.assertEqual(Decimal("2500"), resumed._schedule.opened(resumed._schedule_elapsed()))
        self.assertFalse(resumed._time_limit_reached())
        self.assertTrue(resumed._expired())
        self.now = 5_025
        self.assertAlmostEqual(25.0, resumed._schedule_elapsed())
        self.assertEqual(Decimal("5000"), resumed._schedule.opened(resumed._schedule_elapsed()))
        self.assertFalse(resumed._time_limit_reached())

    def test_restored_schedule_keeps_windows_already_closed(self):
        self._book(Decimal("0.084"), Decimal("0.085"))
        executor = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            cap_behavior=CapBehavior.WAIT,
            catch_up=CatchUpPolicy.SKIP,
        )
        self._run(executor.control_task())
        self.now = 5_000
        self._fill("buy-1", Decimal("2500"), Decimal("212.5"))
        self._run(executor.control_task())
        self._book(Decimal("0.10"), Decimal("0.11"))
        self.now = 5_050
        self._run(executor.control_task())
        self.assertEqual(Decimal("2500"), executor._schedule.skipped)
        payload = json.loads(executor.executor_info.model_dump_json())
        self._book(Decimal("0.084"), Decimal("0.085"))
        resumed = self._up_to(
            duration=100,
            schedule_slices=4,
            slice_amount=Decimal("2500"),
            cap_behavior=CapBehavior.WAIT,
            catch_up=CatchUpPolicy.SKIP,
        )
        resumed.restore_from_custom_info(payload["custom_info"])
        self._run(resumed.control_task())
        self.assertEqual(Decimal("2500"), resumed._schedule.skipped)
        self.assertEqual(5_000, resumed.get_custom_info()["schedule_started_at"])
