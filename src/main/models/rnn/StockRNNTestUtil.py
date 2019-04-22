import csv
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
import StockRNN as sr


if __name__ == '__main__':

    priceList = sr.readData()
    print("PriceList:"+str(priceList))
    totalList  = sr.preparePredictorList(priceList)[:1000]
    print("Predictor list:"+str(totalList))
    x = tf.placeholder("float", shape = ( sr.nInput, 1))
    y = tf.placeholder("float", shape = (sr.outputSize))
    weights = {
       'out': tf.Variable(tf.random_normal([sr.nHidden, sr.outputSize]))
    }
    biases = {
       'out': tf.Variable(tf.random_normal([sr.outputSize]))
    }

    pred = sr.RNN(x,weights,biases)
    saver = tf.train.Saver()
    print("Restoring model")
    with tf.Session() as session:
         saver.restore(session, "model/model.ckpt")
         print("Restored model")
         testNumber = 0
         testPredictions = []
         nInput = sr.nInput
         outputSize = sr.outputSize
         for i in range(len(totalList)):
             if(testNumber+nInput < len(totalList)):
                 testSequeunce   = [totalList[i][0] for i in range(testNumber,testNumber+nInput)]
                 testSequeunce   = np.array(testSequeunce).reshape((nInput,1))
                 correctOutput   = totalList[testNumber+nInput][1]
                 predictedOutputOnehot =  session.run(pred, feed_dict={x: testSequeunce})
                 predictedOutput        = int(tf.argmax(predictedOutputOnehot, 1).eval())+1
                 testPredictions.append((int(predictedOutput),int(correctOutput)))
                 testNumber += 1
                 if i % 1000 == 0 :
                     print("Completed prediction for :"+str(i))
         print("Prediction list size:"+str(len(testPredictions)))
         countDict = sr.getClassPrecisionAndRecall(testPredictions)
         for key in countDict.keys():
             entry     = countDict[key]
             if entry[0] != 0:
                 precision = float(entry[1]/entry[0])
             else:
                 precision = 0
             if entry[2] != 0 :
                 recall = float(entry[1]/entry[2])
             else:
                 recall = 0
             print("For class:"+str(key)+" Precision = "+ str(precision)+" Recall "+str(recall))

