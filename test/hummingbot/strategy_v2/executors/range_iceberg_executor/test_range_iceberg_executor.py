from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import MagicMock, PropertyMock

from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import (
    CapBehavior,
    CatchUpPolicy,
    ExecutionMode,
    RangeIcebergExecutorConfig,
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
        executor.resume()
        self._run(executor.control_task())
        self.assertEqual(1, self.strategy.buy.call_count)

    def test_cancel_parent_cancels_children(self):
        executor = self._executor()
        self._run(executor.control_task())
        executor.early_stop()
        self.strategy.cancel.assert_called_once_with("mexc_perpetual", "AIOT-USDT", "buy-1")
        self.assertEqual(CloseType.EARLY_STOP, executor.close_type)
        self.assertEqual(RunnableStatus.TERMINATED, executor.status)

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
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
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
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
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
        executor = self._up_to(duration=100, schedule_slices=4, slice_amount=Decimal("2500"))
        self._run(executor.control_task())
        self.orders["buy-1"].executed_amount_base = Decimal("2500")
        self.orders["buy-1"].executed_amount_quote = Decimal("212.5")
        self.orders["buy-1"].current_state = OrderState.FILLED
        executor.pause()
        self.now = 1_030
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        self.assertTrue(executor.get_custom_info()["paused"])
        executor.resume()
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
