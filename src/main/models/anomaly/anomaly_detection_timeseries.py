from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from matplotlib import  pyplot as plt
from stldecompose import decompose, forecast

if __name__ == '__main__':
   data_path = "/home/qburst/Downloads/Temperature-33994.csv"
   orig_dataset= pd.read_csv(data_path)
   dataset = pd.read_csv(data_path)
   dataset = dataset.drop_duplicates(subset="Date")
   sliced_dataset = dataset[['Date','Value']]
   sliced_dataset.index = pd.to_datetime(dataset['Date'])
   result = decompose(sliced_dataset['Value'],period=140)
   result.plot()
   #plt.show()
   residuals = result.resid
   dataset['residuals']= residuals.values
   sd = dataset['residuals'].std()
   print("SD:"+str(sd))
   print(residuals)
   anomalies = dataset[dataset['residuals']< - 3 * sd]
   print(anomalies)
   print("size:"+str(len(anomalies)))
   anomalies.to_csv("result_anomalies_temperature.csv")





