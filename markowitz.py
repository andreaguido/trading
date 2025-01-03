import scipy.optimize as sco
import scipy.interpolate as sci
from import_data import datosYahoo
import MA_function as ma
import numpy as np
import matplotlib.pyplot as plt
import MR_function as mo

# 1. Define assets
assets = ["AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "INTC", "NVDA", "TSM", "AVGO",   # tech
          "TSLA","NFLX",                                                                    # consumer (electronics)
          "V",                                                                              # consumer (business services)
          "PFE",
          "XOM", "CVX",                                                                      # energy (oil)
          "JPM", "AXP"
          ]
noa = len(assets)

# 2. Import data
df = datosYahoo(asset_list=assets)
#df=pd.read_csv('data.csv', index_col=0)

# 3. Moving average strategy

# Initialize SMA object
sma_instance = ma.SMA(
    #symbol=assets,
    db=df,
    SMA1=range(10, 50, 10),
    SMA2=range(50, 200, 20),
    series=False
)

backtesting=sma_instance.ma_backtesting(short_allowed=True, plot=True, percent_training=0.7)

# Portfolio returns based on strategy and comparison with benchmark
portfolio_results=sma_instance.portfolio_simulation(weights=opts['x'], benchmark="even")
### NEED TO MAKE IT REUSABLE XXX

# 4. Strategy momentum

# Initialize MO object
mo_instance = mo.MO(db=df, momentum=range(1,3,1), series=False)
mo_strategy=mo_instance.mo_backtesting()

# 5. Mean reversion
import MR as meanrev
mr_instance = meanrev.mr(db=df, SMA=range(1, 40, 1), threshold=3)
mr_strategy=mr_instance.mr_backtesting()