from decimal import Decimal
from unittest import TestCase

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import LadderDistribution, SizeUnit
from hummingbot.strategy_v2.executors.range_iceberg_executor.ladder import build_range_ladder


class RangeLadderTests(TestCase):
    def test_linear_quote_buy_stays_inside_the_range_and_on_the_contract_step(self):
        levels = build_range_ladder(
            total_amount=Decimal("10000"),
            size_unit=SizeUnit.QUOTE,
            price_low=Decimal("0.08"),
            price_high=Decimal("0.09"),
            levels=5,
            distribution=LadderDistribution.LINEAR,
            weight=Decimal("0"),
            side=TradeType.BUY,
            price_step=Decimal("0.00001"),
            amount_step=Decimal("10"),
            min_amount=Decimal("10"),
            contract_size=Decimal("10"),
        )

        self.assertEqual(Decimal("0.08"), levels[0].price)
        self.assertEqual(Decimal("0.09"), levels[-1].price)
        quote = sum(level.quote_amount for level in levels)
        self.assertGreater(quote, Decimal("9900"))
        self.assertLessEqual(quote, Decimal("10000"))
        for level in levels:
            self.assertGreaterEqual(level.price, Decimal("0.08"))
            self.assertLessEqual(level.price, Decimal("0.09"))
            self.assertEqual(0, level.base_amount % Decimal("10"))

    def test_contracts_are_converted_with_contract_size(self):
        levels = build_range_ladder(
            total_amount=Decimal("30"),
            size_unit=SizeUnit.CONTRACTS,
            price_low=Decimal("0.08"),
            price_high=Decimal("0.08"),
            levels=1,
            distribution=LadderDistribution.LINEAR,
            weight=Decimal("0"),
            side=TradeType.BUY,
            price_step=Decimal("0.00001"),
            amount_step=Decimal("10"),
            min_amount=Decimal("10"),
            contract_size=Decimal("10"),
        )
        self.assertEqual(Decimal("300"), levels[0].base_amount)

    def test_geometric_prices_grow_by_a_constant_ratio(self):
        levels = build_range_ladder(
            total_amount=Decimal("3"),
            size_unit=SizeUnit.BASE,
            price_low=Decimal("1"),
            price_high=Decimal("8"),
            levels=4,
            distribution=LadderDistribution.GEOMETRIC,
            weight=Decimal("0"),
            side=TradeType.BUY,
            price_step=Decimal("0.0001"),
            amount_step=Decimal("0.1"),
            min_amount=Decimal("0.1"),
        )
        ratios = [levels[index + 1].price / levels[index].price for index in range(len(levels) - 1)]
        self.assertAlmostEqual(float(ratios[0]), float(ratios[1]), places=4)
        self.assertAlmostEqual(float(ratios[1]), float(ratios[2]), places=4)

    def test_negative_weight_puts_more_size_at_the_low_price(self):
        levels = build_range_ladder(
            total_amount=Decimal("100"),
            size_unit=SizeUnit.BASE,
            price_low=Decimal("0.08"),
            price_high=Decimal("0.09"),
            levels=3,
            distribution=LadderDistribution.WEIGHTED,
            weight=Decimal("-1"),
            side=TradeType.BUY,
            price_step=Decimal("0.00001"),
            amount_step=Decimal("1"),
            min_amount=Decimal("1"),
        )
        self.assertGreater(levels[0].base_amount, levels[-1].base_amount)
        self.assertEqual(Decimal("100"), sum(level.base_amount for level in levels))
