from decimal import Decimal
from unittest import TestCase

from pydantic import SecretStr

import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_utils as utils
from hummingbot.client.settings import AllConnectorSettings, ConnectorType


class MexcPerpetualUtilsTests(TestCase):
    def test_default_fees_match_the_common_usdt_contract_tier(self):
        self.assertEqual(Decimal("0"), utils.DEFAULT_FEES.maker_percent_fee_decimal)
        self.assertEqual(Decimal("0.0002"), utils.DEFAULT_FEES.taker_percent_fee_decimal)
        self.assertEqual("USDT", utils.DEFAULT_FEES.percent_fee_token)

    def test_config_map_credential_fields_are_secure_connect_keys(self):
        self.assertEqual("mexc_perpetual", utils.KEYS.connector)
        for field_name in ("mexc_perpetual_api_key", "mexc_perpetual_api_secret"):
            extra = utils.MexcPerpetualConfigMap.model_fields[field_name].json_schema_extra
            self.assertTrue(extra["is_secure"], field_name)
            self.assertTrue(extra["is_connect_key"], field_name)
            self.assertTrue(extra["prompt_on_new"], field_name)

    def test_config_map_holds_credentials_as_secrets(self):
        config = utils.MexcPerpetualConfigMap(
            mexc_perpetual_api_key="key-id",
            mexc_perpetual_api_secret="secret",
        )
        self.assertIsInstance(config.mexc_perpetual_api_key, SecretStr)
        self.assertIsInstance(config.mexc_perpetual_api_secret, SecretStr)
        self.assertEqual("key-id", config.mexc_perpetual_api_key.get_secret_value())
        self.assertEqual("secret", config.mexc_perpetual_api_secret.get_secret_value())

    def test_exchange_information_filter_keeps_enabled_usdt_perpetuals(self):
        valid = {
            "symbol": "BTC_USDT",
            "futureType": 1,
            "state": 0,
            "quoteCoin": "USDT",
            "settleCoin": "USDT",
            "apiAllowed": True,
        }
        self.assertTrue(utils.is_exchange_information_valid(valid))
        self.assertFalse(utils.is_exchange_information_valid({**valid, "futureType": 2}))
        self.assertFalse(utils.is_exchange_information_valid({**valid, "state": 4}))
        self.assertFalse(utils.is_exchange_information_valid({**valid, "quoteCoin": "USD", "settleCoin": "BTC"}))
        self.assertFalse(utils.is_exchange_information_valid({**valid, "apiAllowed": False}))

    def test_connector_is_registered_in_client_settings(self):
        AllConnectorSettings.create_connector_settings()
        settings = AllConnectorSettings.get_connector_settings()["mexc_perpetual"]

        self.assertEqual(ConnectorType.Derivative, settings.type)
        self.assertEqual("BTC-USDT", settings.example_pair)
        self.assertTrue(settings.centralised)
        self.assertIs(utils.DEFAULT_FEES, settings.trade_fee_schema)
        self.assertIs(utils.KEYS, settings.config_keys)
        self.assertEqual(
            "hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative",
            settings.module_path(),
        )
        self.assertEqual("MexcPerpetualDerivative", settings.class_name())
