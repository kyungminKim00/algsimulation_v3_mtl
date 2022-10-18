# from envs.sp500_trading import Sp500TradingEnv
# from envs.fund_selection_v0 import FundSelectionEnv  # MultiDiscrete Env. n to n reward calculation
# from envs.fund_selection_v1 import FundSelectionEnvCov  # FundSelectionEnv_Cov

import header.index_forecasting.RUNHEADER as RUNHEADER1
import header.market_timing.RUNHEADER as RUNHEADER2

assert (
    RUNHEADER1.release == RUNHEADER2.release
), "index_forecasting.RUNHEADER.release and market_timing.RUNHEADER.release are should be the same"


if RUNHEADER1.release:
    from libs.envs.market_timing_v0 import MarketTimingEnv
    from libs.envs.index_forecasting_v0 import (
        IndexForecastingEnv,
    )  # IndexFForecastingEnv
else:
    from envs.market_timing_v0 import MarketTimingEnv
    from envs.index_forecasting_v0 import IndexForecastingEnv  # IndexForecastingEnv
