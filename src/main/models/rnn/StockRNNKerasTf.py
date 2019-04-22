
import tensorflow as tf
import StockRNN as sr
import numpy as np


def runTraining(trainX,trainY):
    priceList = sr.readData()
    totalList = sr.preparePredictorList(priceList)
    model = tf.keras.models.Sequential()
    dense = tf.keras.layers.Dense(1)
    lstm  = tf.keras.layers.LSTM(4, input_shape=(1, sr.nInput))
    model.add(lstm)
    model.add(dense)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)



if __name__ == '__main__':
    priceList = sr.readData()
    inputOutputList = sr.preparePredictorList(priceList)
    trainX    = []
    trainY    = []
    for i in range(0,len(inputOutputList)):
        if(i + sr.nInput < len(inputOutputList)):
            trainX.append(np.array([inputOutputList[j][0] for j in range(i,i+sr.nInput)]))
            trainY.append(sr.getOneHot(inputOutputList[i+sr.nInput][1]))
    trainX = np.array(trainX).T
    runTraining(trainX,trainY)




