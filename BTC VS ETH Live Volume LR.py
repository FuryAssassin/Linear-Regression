import matplotlib.pyplot as plt
import pandas as pd
import ssl
from sklearn.linear_model import LinearRegression
root1 = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv"
root2=  "https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_1h.csv"
ssl._create_default_https_context = ssl._create_unverified_context
df1 = pd.read_csv(root1, skiprows=1)
df2 = pd.read_csv(root2, skiprows=1)
data1 = df1.head(200)
data2 = df2.head(200)
x = data1[['Volume USDT']].iloc[:, 0].values.reshape(-1, 1)
y = data2[['Volume USDT']].iloc[:, 0].values.reshape(-1, 1)
lr = LinearRegression()
lr.fit(x, y)
y_pred = lr.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

###### Done by Rethvick (RA1911042020012) ########



###### Fury Assassin ######