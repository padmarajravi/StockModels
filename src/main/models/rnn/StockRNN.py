import csv
import numpy as np


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

## Window for  which price change is to be predicted.
predictorInterval = 3



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
    return zip(priceList,predictorPriceList, predictorList)




if __name__ == '__main__':
    priceList = readData()
    predictorbatches = preparePredictorList(priceList)
    for tuple in preparePredictorList(priceList):
        print(tuple)




