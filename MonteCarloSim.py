
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

# Data Retrieval

def get_data(ticker, start, end):
    data = yf.download(ticker, start= start, end = end)['Close']
    return data

stock = 'NVDA'
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

stock_data = get_data(stock, startDate, endDate)

log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
mean = log_returns.mean()
std = log_returns.std()

# Monte Carlo Simulation

T = 212 # future days to simulate prices to
sims = 1000 # number of monte carlo sims
np.random.seed(55)

simulated_returns = np.random.normal(loc = mean, scale = std, size = (T, sims))

last_price = stock_data.iloc[-1].item()
cumulative_returns = np.cumsum(simulated_returns, axis = 0)
price_path = last_price * np.exp(cumulative_returns)

# Plotting

plt.figure(figsize=(12, 8))
plt.plot(price_path)
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()