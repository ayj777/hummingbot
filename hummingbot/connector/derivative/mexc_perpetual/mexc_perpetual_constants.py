from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.data_type.common import OrderType, PositionMode
from hummingbot.core.data_type.in_flight_order import OrderState

EXCHANGE_NAME = "mexc_perpetual"

REST_URL = "https://api.mexc.com"
WSS_URL = "wss://contract.mexc.com/edge"

# Seconds. The server default window is 10 seconds and the maximum is 60.
RECV_WINDOW = "10"

# External order ids are rejected above 32 characters (error 2030).
ORDER_ID_MAX_LEN = 32
HBOT_ORDER_ID_PREFIX = "mxp"

# Documented market-order taker depth. Contract info also returns marketOrderMaxLevel per symbol.
MARKET_ORDER_DEFAULT_LEVELS = 20

# Feature is under maintenance / the futures system is under maintenance.
MAINTENANCE_ERROR_CODES = {604, 801}
ORDER_NOT_FOUND_ERROR_CODES = {2040}
CONFIRMED_ABSENT = "confirmed_absent"
# The order can no longer be canceled. That is what a fill looks like on the cancel endpoint.
ORDER_CANNOT_CANCEL_ERROR_CODE = 2041

# Order side: 1 open long, 2 close short, 3 open short, 4 close long.
SIDE_OPEN_LONG = 1
SIDE_CLOSE_SHORT = 2
SIDE_OPEN_SHORT = 3
SIDE_CLOSE_LONG = 4

# Order type: 1 limit, 2 post only, 3 IOC, 4 FOK, 5 market.
ORDER_TYPE_LIMIT = 1
ORDER_TYPE_POST_ONLY = 2
ORDER_TYPE_IOC = 3
ORDER_TYPE_FOK = 4
ORDER_TYPE_MARKET = 5

# openType: 1 isolated, 2 cross.
OPEN_TYPE_ISOLATED = 1
OPEN_TYPE_CROSS = 2

# positionMode: 1 hedge, 2 one-way.
POSITION_MODE_HEDGE = 1
POSITION_MODE_ONEWAY = 2

# positionType: 1 long, 2 short.
POSITION_TYPE_LONG = 1
POSITION_TYPE_SHORT = 2

# Contract state: 0 enabled, 1 delivery, 2 delivered, 3 offline, 4 paused.
CONTRACT_STATE_ENABLED = 0
FUTURE_TYPE_PERPETUAL = 1

# Order state: 1 pending, 2 unfilled, 3 filled, 4 canceled, 5 invalid.
ORDER_STATE_PENDING = 1
ORDER_STATE_OPEN = 2
ORDER_STATE_FILLED = 3
ORDER_STATE_CANCELED = 4
ORDER_STATE_INVALID = 5

SECONDS_TO_WAIT_TO_RECEIVE_MESSAGE = 60
# The server closes the socket if it receives no ping within 60 seconds.
WS_HEARTBEAT_TIME_INTERVAL = 15

PING_PATH = "/api/v1/contract/ping"
# Documented contract-info endpoint. The older /api/v1/contract/detail path still
# responds, but /country is the current reference and applies country filters.
CONTRACT_INFO_PATH = "/api/v1/contract/detail/country"
TICKER_PATH = "/api/v1/contract/ticker"
DEPTH_PATH = "/api/v1/contract/depth/{symbol}"
DEPTH_LIMIT_ID = "/api/v1/contract/depth"
DEPTH_COMMITS_PATH = "/api/v1/contract/depth_commits/{symbol}/{limit}"
DEPTH_COMMITS_LIMIT_ID = "/api/v1/contract/depth_commits"
DEALS_PATH = "/api/v1/contract/deals/{symbol}"
DEALS_LIMIT_ID = "/api/v1/contract/deals"
FUNDING_RATE_PATH = "/api/v1/contract/funding_rate/{symbol}"
FUNDING_RATE_LIMIT_ID = "/api/v1/contract/funding_rate"

PLACE_ORDER_PATH = "/api/v1/private/order/create"
BATCH_ORDER_PATH = "/api/v1/private/order/submit_batch"
CANCEL_ORDER_PATH = "/api/v1/private/order/cancel"
CANCEL_BY_EXTERNAL_PATH = "/api/v1/private/order/cancel_with_external"
ORDER_BY_EXTERNAL_PATH = "/api/v1/private/order/external/{symbol}/{external_oid}"
ORDER_BY_EXTERNAL_LIMIT_ID = "/api/v1/private/order/external"
ORDER_GET_PATH = "/api/v1/private/order/get/{order_id}"
ORDER_GET_LIMIT_ID = "/api/v1/private/order/get"
ORDER_DEALS_PATH = "/api/v1/private/order/deal_details/{order_id}"
ORDER_DEALS_LIMIT_ID = "/api/v1/private/order/deal_details"
ASSETS_PATH = "/api/v1/private/account/assets"
OPEN_POSITIONS_PATH = "/api/v1/private/position/open_positions"
CHANGE_LEVERAGE_PATH = "/api/v1/private/position/change_leverage"
POSITION_MODE_PATH = "/api/v1/private/position/position_mode"
CHANGE_POSITION_MODE_PATH = "/api/v1/private/position/change_position_mode"
FUNDING_RECORDS_PATH = "/api/v1/private/position/funding_records"

PLAN_ORDER_PLACE_PATH = "/api/v1/private/planorder/place/v2"
PLAN_ORDER_CANCEL_PATH = "/api/v1/private/planorder/cancel"
PLAN_ORDER_CANCEL_ALL_PATH = "/api/v1/private/planorder/cancel_all"
PLAN_ORDER_CHANGE_PRICE_PATH = "/api/v1/private/planorder/change_price"
PLAN_ORDER_CHANGE_STOP_PATH = "/api/v1/private/planorder/change_stop_order"
PLAN_ORDER_LIST_PATH = "/api/v1/private/planorder/list/orders"

STOP_ORDER_PLACE_PATH = "/api/v1/private/stoporder/place"
STOP_ORDER_CANCEL_PATH = "/api/v1/private/stoporder/cancel"
STOP_ORDER_CANCEL_ALL_PATH = "/api/v1/private/stoporder/cancel_all"
STOP_ORDER_CHANGE_PRICE_PATH = "/api/v1/private/stoporder/change_price"
STOP_ORDER_CHANGE_PLAN_PRICE_PATH = "/api/v1/private/stoporder/change_plan_price"
STOP_ORDER_OPEN_PATH = "/api/v1/private/stoporder/open_orders"
STOP_ORDER_LIST_PATH = "/api/v1/private/stoporder/list/orders"

# Trigger price type: 1 latest, 2 fair, 3 index.
TRIGGER_PRICE_LATEST = 1
TRIGGER_PRICE_FAIR = 2
TRIGGER_PRICE_INDEX = 3
# Plan trigger: 1 price >= trigger, 2 price <= trigger.
TRIGGER_TYPE_GTE = 1
TRIGGER_TYPE_LTE = 2
# Plan execute cycle: 1 is 24 hours, 2 is 7 days.
EXECUTE_CYCLE_24H = 1
EXECUTE_CYCLE_7D = 2
BATCH_ORDER_MAX = 50
# Self-trade prevention. 0 leaves both orders working.
STP_MODE_NONE = 0

WS_LOGIN_METHOD = "login"
WS_PING_METHOD = "ping"
WS_PONG_CHANNEL = "pong"
WS_LOGIN_CHANNEL = "rs.login"
WS_ERROR_CHANNEL = "rs.error"
WS_DEPTH_CHANNEL = "push.depth"
WS_DEAL_CHANNEL = "push.deal"
WS_FUNDING_CHANNEL = "push.funding.rate"
WS_ORDER_CHANNEL = "push.personal.order"
WS_ORDER_DEAL_CHANNEL = "push.personal.order.deal"
WS_POSITION_CHANNEL = "push.personal.position"
WS_ASSET_CHANNEL = "push.personal.asset"

DEPTH_SNAPSHOT_LIMIT = 1000

ORDER_TYPE_MAP = {
    OrderType.LIMIT: ORDER_TYPE_LIMIT,
    OrderType.LIMIT_MAKER: ORDER_TYPE_POST_ONLY,
    OrderType.MARKET: ORDER_TYPE_MARKET,
}

POSITION_MODE_MAP = {
    PositionMode.HEDGE: POSITION_MODE_HEDGE,
    PositionMode.ONEWAY: POSITION_MODE_ONEWAY,
}

POSITION_MODE_FROM_EXCHANGE = {
    POSITION_MODE_HEDGE: PositionMode.HEDGE,
    POSITION_MODE_ONEWAY: PositionMode.ONEWAY,
}

ORDER_STATE_MAP = {
    ORDER_STATE_PENDING: OrderState.OPEN,
    ORDER_STATE_OPEN: OrderState.OPEN,
    ORDER_STATE_FILLED: OrderState.FILLED,
    ORDER_STATE_CANCELED: OrderState.CANCELED,
    ORDER_STATE_INVALID: OrderState.FAILED,
}

# Limits below are the per-endpoint values published in the futures docs (counts per 2 seconds,
# except change leverage which is 10 per 10 seconds).
RATE_LIMITS = [
    RateLimit(limit_id=PING_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=CONTRACT_INFO_PATH, limit=10, time_interval=2),
    RateLimit(limit_id=TICKER_PATH, limit=10, time_interval=2),
    RateLimit(limit_id=DEPTH_LIMIT_ID, limit=10, time_interval=2),
    RateLimit(limit_id=DEPTH_COMMITS_LIMIT_ID, limit=20, time_interval=2),
    RateLimit(limit_id=DEALS_LIMIT_ID, limit=20, time_interval=2),
    RateLimit(limit_id=FUNDING_RATE_LIMIT_ID, limit=20, time_interval=2),
    RateLimit(limit_id=PLACE_ORDER_PATH, limit=4, time_interval=2),
    RateLimit(limit_id=BATCH_ORDER_PATH, limit=4, time_interval=2),
    RateLimit(limit_id=CANCEL_ORDER_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=PLAN_ORDER_PLACE_PATH, limit=4, time_interval=2),
    RateLimit(limit_id=PLAN_ORDER_CANCEL_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=PLAN_ORDER_CANCEL_ALL_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=PLAN_ORDER_CHANGE_PRICE_PATH, limit=4, time_interval=2),
    RateLimit(limit_id=PLAN_ORDER_CHANGE_STOP_PATH, limit=4, time_interval=2),
    RateLimit(limit_id=PLAN_ORDER_LIST_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=STOP_ORDER_PLACE_PATH, limit=5, time_interval=2),
    RateLimit(limit_id=STOP_ORDER_CANCEL_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=STOP_ORDER_CANCEL_ALL_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=STOP_ORDER_CHANGE_PRICE_PATH, limit=4, time_interval=2),
    RateLimit(limit_id=STOP_ORDER_CHANGE_PLAN_PRICE_PATH, limit=4, time_interval=2),
    RateLimit(limit_id=STOP_ORDER_OPEN_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=STOP_ORDER_LIST_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=CANCEL_BY_EXTERNAL_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=ORDER_BY_EXTERNAL_LIMIT_ID, limit=20, time_interval=2),
    RateLimit(limit_id=ORDER_GET_LIMIT_ID, limit=20, time_interval=2),
    RateLimit(limit_id=ORDER_DEALS_LIMIT_ID, limit=20, time_interval=2),
    RateLimit(limit_id=ASSETS_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=OPEN_POSITIONS_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=CHANGE_LEVERAGE_PATH, limit=10, time_interval=10),
    RateLimit(limit_id=POSITION_MODE_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=CHANGE_POSITION_MODE_PATH, limit=20, time_interval=2),
    RateLimit(limit_id=FUNDING_RECORDS_PATH, limit=20, time_interval=2),
]
