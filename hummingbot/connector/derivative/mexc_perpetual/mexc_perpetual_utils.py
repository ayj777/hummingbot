from decimal import Decimal
from typing import Any, Dict

from pydantic import ConfigDict, Field, SecretStr

from hummingbot.client.config.config_data_types import BaseConnectorConfigMap
from hummingbot.connector.derivative.mexc_perpetual import mexc_perpetual_constants as CONSTANTS
from hummingbot.core.data_type.trade_fee import TradeFeeSchema

CENTRALIZED = True
EXAMPLE_PAIR = "BTC-USDT"

# Most enabled USDT-M contracts on the public contract list currently publish maker 0 and taker 0.0002.
# Pair-specific rates from contract info replace this fallback after the trading-fee update.
DEFAULT_FEES = TradeFeeSchema(
    maker_percent_fee_decimal=Decimal("0"),
    taker_percent_fee_decimal=Decimal("0.0002"),
    percent_fee_token="USDT",
)


def is_exchange_information_valid(exchange_info: Dict[str, Any]) -> bool:
    """
    True for an enabled USDT-margined perpetual that API trading is allowed to use.
    """
    if not exchange_info.get("symbol"):
        return False
    if exchange_info.get("futureType") != CONSTANTS.FUTURE_TYPE_PERPETUAL:
        return False
    if exchange_info.get("state") != CONSTANTS.CONTRACT_STATE_ENABLED:
        return False
    if exchange_info.get("quoteCoin") != "USDT" or exchange_info.get("settleCoin") != "USDT":
        return False
    if exchange_info.get("apiAllowed") is False:
        return False
    return True


class MexcPerpetualConfigMap(BaseConnectorConfigMap):
    connector: str = "mexc_perpetual"
    mexc_perpetual_api_key: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": "Enter your MEXC Perpetual API key",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    mexc_perpetual_api_secret: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": "Enter your MEXC Perpetual API secret",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    model_config = ConfigDict(title="mexc_perpetual")


KEYS = MexcPerpetualConfigMap.model_construct()
