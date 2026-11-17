from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from typing import List

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import LadderDistribution, SizeUnit


@dataclass
class LadderLevel:
    price: Decimal
    base_amount: Decimal

    @property
    def quote_amount(self) -> Decimal:
        return self.price * self.base_amount


def build_range_ladder(
    total_amount: Decimal,
    size_unit: SizeUnit,
    price_low: Decimal,
    price_high: Decimal,
    levels: int,
    distribution: LadderDistribution,
    weight: Decimal,
    side: TradeType,
    price_step: Decimal,
    amount_step: Decimal,
    min_amount: Decimal,
    contract_size: Decimal = Decimal("1"),
) -> List[LadderLevel]:
    """
    Split a total size into limit-order levels inside [price_low, price_high].

    Prices are linear or geometric. Sizes are equal unless weight pulls them toward
    one end of the range. Amounts and prices are rounded onto the contract steps.
    """
    if levels < 1:
        raise ValueError("levels must be at least 1")
    if price_low <= 0 or price_high <= 0 or price_low > price_high:
        raise ValueError("price range must be positive and low must not exceed high")
    if contract_size <= 0:
        contract_size = Decimal("1")
    prices = _prices(price_low, price_high, levels, distribution, side, price_step)
    weights = _weights(len(prices), weight)
    if size_unit is SizeUnit.QUOTE:
        return _levels_from_quote(total_amount, prices, weights, amount_step, min_amount)
    total_base = total_amount * contract_size if size_unit is SizeUnit.CONTRACTS else total_amount
    return _levels_from_base(total_base, prices, weights, amount_step, min_amount)


def quantize_down(value: Decimal, step: Decimal) -> Decimal:
    if step is None or step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def quantize_up(value: Decimal, step: Decimal) -> Decimal:
    if step is None or step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def cap_buy_price(price: Decimal, step: Decimal, cap: Decimal) -> Decimal:
    """
    Round a buy limit down onto the price step and never above the cap.
    Rounding down is the safe direction: the order cannot pay more than price or the cap.
    """
    capped = price if price <= cap else cap
    quantized = quantize_down(capped, step)
    if quantized > cap:
        quantized = quantize_down(cap, step)
    return quantized


def floor_sell_price(price: Decimal, step: Decimal, floor: Decimal) -> Decimal:
    """
    Round a sell limit up onto the price step and never below the floor.
    Rounding up is the safe direction: the order cannot sell for less than price or the floor.
    """
    floored = price if price >= floor else floor
    quantized = quantize_up(floored, step)
    if quantized < floor:
        quantized = quantize_up(floor, step)
    return quantized


def clamp_price(price: Decimal, price_low: Decimal, price_high: Decimal, price_step: Decimal) -> Decimal:
    if price_step is None or price_step <= 0:
        quantized = price
    else:
        quantized = quantize_down(price, price_step)
        if quantized < price_low:
            quantized = (price_low / price_step).to_integral_value(rounding=ROUND_CEILING) * price_step
    if quantized < price_low:
        quantized = price_low
    if quantized > price_high:
        quantized = quantize_down(price_high, price_step) if price_step and price_step > 0 else price_high
    return quantized


def _prices(
    price_low: Decimal,
    price_high: Decimal,
    levels: int,
    distribution: LadderDistribution,
    side: TradeType,
    price_step: Decimal,
) -> List[Decimal]:
    if levels == 1:
        anchor = price_low if side is TradeType.BUY else price_high
        return [clamp_price(anchor, price_low, price_high, price_step)]
    prices = []
    for index in range(levels):
        fraction = Decimal(index) / Decimal(levels - 1)
        if distribution is LadderDistribution.GEOMETRIC:
            raw = price_low * ((price_high / price_low) ** fraction)
        else:
            raw = price_low + (price_high - price_low) * fraction
        prices.append(clamp_price(raw, price_low, price_high, price_step))
    return prices


def _weights(levels: int, weight: Decimal) -> List[Decimal]:
    if levels == 1:
        return [Decimal("1")]
    raw = []
    for index in range(levels):
        position = (Decimal(2 * index) / Decimal(levels - 1)) - Decimal("1")
        raw.append(max(Decimal("1") + weight * position, Decimal("0.0000001")))
    total = sum(raw)
    return [item / total for item in raw]


def _levels_from_base(
    total_base: Decimal,
    prices: List[Decimal],
    weights: List[Decimal],
    amount_step: Decimal,
    min_amount: Decimal,
) -> List[LadderLevel]:
    remaining = total_base
    levels = []
    for index, (price, weight) in enumerate(zip(prices, weights)):
        if index == len(prices) - 1:
            piece = remaining
        else:
            piece = total_base * weight
        amount = quantize_down(min(piece, remaining), amount_step)
        if amount >= min_amount and amount > 0:
            levels.append(LadderLevel(price=price, base_amount=amount))
            remaining -= amount
        if remaining <= 0:
            break
    if not levels:
        raise ValueError("The range ladder is below the minimum order size after rounding.")
    return levels


def _levels_from_quote(
    total_quote: Decimal,
    prices: List[Decimal],
    weights: List[Decimal],
    amount_step: Decimal,
    min_amount: Decimal,
) -> List[LadderLevel]:
    remaining_quote = total_quote
    levels = []
    for index, (price, weight) in enumerate(zip(prices, weights)):
        if price <= 0 or remaining_quote <= 0:
            break
        quote_piece = remaining_quote if index == len(prices) - 1 else total_quote * weight
        amount = quantize_down(min(quote_piece, remaining_quote) / price, amount_step)
        if amount >= min_amount and amount > 0:
            levels.append(LadderLevel(price=price, base_amount=amount))
            remaining_quote -= amount * price
    if not levels:
        raise ValueError("The range ladder is below the minimum order size after rounding.")
    return levels
