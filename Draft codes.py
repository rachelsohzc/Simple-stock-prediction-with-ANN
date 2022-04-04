import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'FB.csv'
meta = pd.read_csv(url)
nvidia = meta.dropna(how='any',axis=0) #If there is missing data
meta['Date'].apply(pd.to_datetime) #For ease of adjusting dates later on
len(meta.index)

meta = meta.sort_values('Date')
meta.head()

plt.figure(figsize = (20,9))
plt.plot(meta.Date, meta.Close)
plt.xticks(range(0,meta.shape[0],100),meta['Date'].loc[::100],rotation=45)
plt.xlabel('Date',fontsize=12)
plt.ylabel('Mid price',fontsize=12)
plt.title('Meta stock price changes over time',fontsize=20)
plt.show()