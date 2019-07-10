from __future__ import division
from __future__ import print_function

import os
import sys

from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest


from collections import Counter
import numpy as np





from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from datetime import datetime as dt
from matplotlib import  pyplot as plt


import pandas as pd

if __name__ == '__main__':
    date_format = ""
    data_path = "/home/qburst/Projects/StockModels/src/main/models/anomaly/data/PD data -2017-2019.csv"
    dataset = pd.read_csv(data_path)
    dataset['date'] = pd.to_datetime(dataset['Inspection Date'],dayfirst=True)
    sorted_dataset= dataset.sort_values(by = ['date'])
    print(sorted_dataset.groupby('Inspection Date').count())
    #print(sorted_dataset.to_string())

    sliced_data = sorted_dataset[['PD Average','PD Count','Temperature','Humidity','Loading']]

    print(sorted_dataset.loc[sorted_dataset['Confirm action']=='2'].to_string())


    clfs = [ABOD(contamination=.01),COF(contamination=.01),CBLOF(contamination=.01),IForest(contamination=.01)]

    anomalies = []

    
    for clf in clfs:
        clf.fit(sliced_data)
        y_train_pred = clf.labels_
        sorted_dataset['Anomaly_status'] = y_train_pred
        anomalies.extend(sorted_dataset.loc[sorted_dataset['Anomaly_status']==1].index.values.tolist())
        print("Completed:"+clf.__class__.__name__)



    anomaly_counter = Counter(anomalies)
    print(anomaly_counter)

    print("Anomalies")
    print(dataset.iloc[list(anomaly_counter.keys())].to_csv("result_anomalies_pd.csv"))
    

    

    






