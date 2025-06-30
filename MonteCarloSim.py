
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

# Data Retrieval

def get_data(tickers, start, end):
    data = pd.DataFrame()
    for t in tickers:
        data[t] = yf.download(t, start= start, end = end)['Close']
    return data

stock_list = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'PLTR', 'META']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

stock_data = get_data(stock_list, startDate, endDate)

log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
mean = log_returns.mean()
std = log_returns.std()

# Monte Carlo Simulation

T = 212 # future days to simulate prices to
sims = 1000 # number of monte carlo sims
# np.random.seed(55)

weights = np.array([1/len(stock_list)] * len(stock_list))  # equal weights

last_prices = stock_data.iloc[-1].values
simulated_paths = np.zeros((T, sims, len(stock_list)))

for i, ticker in enumerate(stock_list):
    drift = mean[ticker]
    vol = std[ticker]
    random_returns = np.random.normal(loc = drift, scale = vol, size = (T, sims))
    cumulative_returns = np.cumsum(random_returns, axis = 0)
    simulated_paths[:, :, i] = last_prices[i] * np.exp(cumulative_returns)

portfolio_paths = np.sum(simulated_paths * weights, axis = 2)

# Plotting

# plt.figure(figsize=(12, 8))
# plt.plot(portfolio_paths)
# plt.xlabel('Days')
# plt.ylabel('Price')
# plt.show()

# Analyze to see likely returns

start_values = portfolio_paths[0, :]
end_values = portfolio_paths[-1, :]
returns = (end_values - start_values) / start_values

baseline = 0.10
beat_returns = np.mean(returns > baseline)
print(beat_returns)

avg_returns = np.mean(returns)
print(avg_returns)