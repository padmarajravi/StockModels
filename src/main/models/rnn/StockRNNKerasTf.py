
import tensorflow as tf
import StockRNN as sr
import numpy as np


modelSavePath = "models/keras/model"
testPercentage = .8
batchSize = 1


"""
Train the network using LTSM and a dense layer with softmax  
"""

def trainNetworkLSTM(trainX, trainY):
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    model = tf.keras.models.Sequential()
    lstm  = tf.keras.layers.LSTM(4,batch_input_shape=(batchSize,sr.nInput, 1),stateful= True , return_sequences=True)
    lstm2 = tf.keras.layers.LSTM(10)
    dense = tf.keras.layers.Dense(sr.outputSize,activation="softmax")
    model.add(lstm)
    model.add(lstm2)
    model.add(dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    for i in range(1,500):
        model.fit(trainX, trainY, epochs=1, batch_size=batchSize, verbose=2,shuffle=False)
        model.reset_states()
        print("Completed iteration:"+str(i))
    model.save(modelSavePath)


"""
Train network using only a dense layer and no LTSM layers
"""
def trainNetworkDense(trainX,trainY):
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
    model = tf.keras.models.Sequential()
    dense1 = tf.keras.layers.Dense(10,input_shape=(sr.nInput,))
    dense2 = tf.keras.layers.Dense(3,activation="softmax")
    model.add(dense1)
    model.add(dense2)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(trainX, trainY, epochs=25, batch_size=batchSize, verbose=2)
    model.save(modelSavePath)


def testNetworkDense(testX,testY):
    print("testing for :"+str(len(testX)))
    model = tf.keras.models.load_model(modelSavePath)
    testX = np.reshape(testX, (testX.shape[0],testX.shape[1]))
    predictedY = np.argmax(model.predict(testX),axis=1)
    originalY  = np.argmax(testY,axis=1)
    print(predictedY)
    print(originalY)
    results = sr.getClassPrecisionAndRecall(zip(predictedY,originalY))
    sr.displayResults(results)

def testNetworkLSTM(testX,testY):
    print("testing for :"+str(len(testX)))
    model = tf.keras.models.load_model(modelSavePath)
    testX = np.reshape(testX, (testX.shape[0],testX.shape[1],1))
    predictedY = np.argmax(model.predict(testX),axis=1)
    originalY  = np.argmax(testY,axis=1)
    print(predictedY)
    print(originalY)
    results = sr.getClassPrecisionAndRecall(zip(predictedY,originalY))
    sr.displayResults(results)


def prepareData():
    global trainX, testX, trainY, testY
    priceList = sr.readData()
    inputOutputList = sr.preparePredictorList(priceList)
    totalX = []
    totalY = []
    for i in range(0, len(inputOutputList)):
        if (i + sr.nInput < len(inputOutputList)):
            totalX.append(np.array([inputOutputList[j][0] for j in range(i, i + sr.nInput)]))
            totalY.append(sr.getOneHot(inputOutputList[i + sr.nInput][1]))
    totalX = np.array(totalX)
    totalY = np.array(totalY)
    print("total data set length:"+str(len(totalX)))
    trainSlice = int(testPercentage * len(totalX))
    testSlice = int(testPercentage * len(totalY))
    trainX = totalX[:trainSlice]
    testX = totalX[testSlice:]
    trainY = totalY[:trainSlice]
    testY = totalY[testSlice:]
    return (trainX,testX,trainY,testY)


if __name__ == '__main__':
    trainX,testX,trainY,testY = prepareData()
    print(trainX.shape)
    print(trainY.shape)
    #trainNetworkDense(trainX, trainY)
    #testNetworkDense(trainX,trainY)
    trainNetworkLSTM(trainX,trainY)
    testNetworkLSTM(trainX,trainY)




