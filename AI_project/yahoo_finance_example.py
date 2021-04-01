import yfinance as yf
import pandas as pd

stock_dtf = yf.download('SPY',
                        start='2020-12-31',
                        end='2021-01-31',
                        progress=False)
df = pd.DataFrame(stock_dtf)
vec = []
for index, row in df.iterrows():
    print(index, row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume'])
    vec.append([row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume']])
print("vec", vec)
