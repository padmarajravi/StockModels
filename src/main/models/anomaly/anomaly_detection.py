import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import  pyplot as plt
import math


def multivariateGaussian(X, mu, sigma):
    m, n = X.shape # number of training examples, number of features

    X = X.values - mu.values.reshape(1,n) # (X - mu)

    # vectorized implementation of calculating p(x) for each m examples: p is m length array
    p = (1.0 / (math.pow((2 * math.pi), n / 2.0) * math.pow(np.linalg.det(sigma),0.5))) * np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(sigma)) * X, axis=1))

    return p

if __name__ == '__main__':
    data_path = "/home/qburst/Projects/StockModels/src/main/models/anomaly/data/PD data -2017-2019.csv"
    dataset = pd.read_csv(data_path)
    sliced_data = dataset[['PD Average','PD Count']]
    print("Dataset rows:"+str(dataset.shape))
    print(dataset.columns.values)
    mean = np.mean(dataset[['PD Average','PD Count']],axis=0)
    sigma = np.cov(dataset[['PD Average','PD Count']].T)
    y = multivariateGaussian(sliced_data,mean,sigma)
    print("Max p value:"+str(max(y)))
    print("Max p value:"+str(min(y)))
    print(y)

