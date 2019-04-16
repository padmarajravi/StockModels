import csv
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time


sourcefile = "BSE-BOM532500.csv"
High = 5
Med  = 3
Low  = 0
postiveLow   = 1
positiveMed  = 2
positiveHigh = 3
negativelow  = 4
negativeMed  = 5
negativeHigh = 6

nInput = 3
nHidden = 512

## Window for  which price change is to be predicted.
predictorInterval = 1
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


"""
Read data into a list of close prices
Prepare predictor list from price list.


"""
def readData():
    prices = []
    with open(sourcefile) as csvFile:
        csvReader = csv.reader(csvFile)
        lineCount=0
        for line in csvReader:
            if lineCount > 0:
               prices.append(float(line[4]))
            lineCount += 1
    prices.reverse()
    return prices


def getClassPrecisionAndRecall(predictions):

    # Will store class id , total number of classes and correct prediction
    countDict = {}
    recallDict = {}

    # Function to add values to thekey if it already exists , else create values
    def getOrCreate(key,origCount,correctCount,predictedCount):
        print("Trying to insert key "+str(key)+" in Count dict:"+str(countDict))
        if key in countDict.keys():
            currentEntry = countDict[key]
            countDict[key] = (currentEntry[0]+origCount,currentEntry[1]+correctCount,currentEntry[2]+predictedOutput)
        else :
            countDict[key] = (origCount,correctCount,predictedCount)


    for pair in predictions:
        originalOutput  = pair[1]
        predictedOutput = pair[0]
        # If predicted and actual are same , add one to orignal count , correct count and predicted class count
        if originalOutput == predictedOutput:
            getOrCreate(originalOutput,1,1,1)
        else :
            # Else add 1 to original class count , and zero to correct count and predicted class count. Also add 1 to [redicted class count of relavant class
            getOrCreate(originalOutput,1,0,0)
            getOrCreate(predictedOutput,0,0,1)

    return countDict



def preparePredictorList(priceList):
    predictorList = []
    predictorPriceList = []
    count = 0
    prevPrice = float(0)
    for price in priceList:
        if count + predictorInterval < len(priceList) :
            predictorDayPrice = priceList[count+predictorInterval]
            predictorPriceList.append(predictorDayPrice)
            predictorPriceChange = (predictorDayPrice- price ) * 100 / price
            if predictorPriceChange > 5:
                predictorList.append(positiveHigh)
            elif predictorPriceChange > 3:
                predictorList.append(positiveMed)
            elif predictorPriceChange > 0:
                predictorList.append(postiveLow)
            elif predictorPriceChange < -5:
                predictorList.append(negativeHigh)
            elif predictorPriceChange < -3:
                predictorList.append(negativeMed)
            else:
                predictorList.append(negativelow)
        else:
            predictorList.append(negativelow)

        count += 1
    return predictorList



def RNN(x,weights,biases):

    # reshaping into rows with 3 columns.
    x = tf.reshape(x,[-1,nInput])
    print("x shape inside RNN cell:"+str(tf.shape(x)))
    x = tf.split(x,nInput,1)
    print("x shape after split  inside RNN cell:"+str(tf.shape(x)))
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(nHidden),rnn.BasicLSTMCell(nHidden)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



def runNetwork():
    priceList = readData()
    trainingPercentage = .8
    testPercentage = 1 - trainingPercentage
    totalList  = preparePredictorList(priceList)
    trainingList = totalList[:int(trainingPercentage*len(totalList))]
    testList     = totalList[int(trainingPercentage*len(totalList)):]
    print(trainingList)
    learning_rate  = 0.001
    training_iters = 1000
    display_step   = 1000
    x = tf.placeholder("float", shape = ( nInput, 1))
    y = tf.placeholder("float", shape = ( 6))
    weights = {
        'out': tf.Variable(tf.random_normal([nHidden, 6]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([6]))
    }

    pred = RNN(x,weights,biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,0), tf.argmax(y,0))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    init = tf.global_variables_initializer()

    print("Initialized")

    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0,nInput+1)
        end_offset = nInput+1
        acc_total = 0
        loss_total = 0
        writer.add_graph(session.graph)
        print("Optimization started")
        while(step < training_iters):
            if(offset> len(trainingList) - end_offset):
                offset = random.randint(0, nInput+1)
            offset = random.randint(0, nInput+1)
            inputSequence = [trainingList[i] for i in range(offset,offset+nInput)]
            inputSequence = np.reshape(np.array(inputSequence),(nInput,1))
            outputElement = np.zeros(6,dtype=float)
            outputElement[trainingList[offset+nInput] - 1] = 1.0
            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                    feed_dict={x: inputSequence, y: outputElement})
            loss_total += loss
            acc_total += acc
            if (step+1) % display_step == 0:
                print("Iter= " + str(step+1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100*acc_total/display_step))
                acc_total = 0
                loss_total = 0
                symbols_in = [trainingList[i] for i in range(offset, offset + nInput)]
                symbols_out = trainingList[offset + nInput]
                symbols_out_pred = int(tf.argmax(onehot_pred, 1).eval()) + 1
                print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
            step = step + 1
            offset = offset + nInput + 1
        print("Optimization Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % (logs_path))
        print("Point your web browser to: http://localhost:6006/")


        print("Testing for "+str(len(testList)))
        testNumber = 0
        testPredictions = []
        for i in range(len(testList)):
            if(testNumber+nInput < len(testList)):
                testSequeunce   = [testList[i] for i in range(testNumber,testNumber+nInput)]
                testSequeunce   = np.array(testSequeunce).reshape((nInput,1))
                correctOutput   = testList[testNumber+nInput]
                predictedOutputOnehot =  session.run(pred, feed_dict={x: testSequeunce})
                predictedOutput        = int(tf.argmax(predictedOutputOnehot, 1).eval())+1
                print("Predicted:"+str(predictedOutput)+" | Correct :"+str(correctOutput))
                testPredictions.append((int(predictedOutput),int(correctOutput)))
                testNumber += 1


        countDict = getClassPrecisionAndRecall(testPredictions)
        for key in countDict.keys():
            entry     = countDict[key]
            precision = float(entry[1]/entry[0])
            recall    = float(entry[1]/entry[2])
            print("For class:"+str(key)+" Precision = "+ str(precision)+" Recall"+str(recall))










if __name__ == '__main__':
    runNetwork()




