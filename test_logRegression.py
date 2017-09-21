#################################################
# logRegression: Logistic Regression
# Author : zouxy
# Date   : 2014-03-02
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import matplotlib.pyplot as plt
import time
import logRegression as lR

def loadData(fileName):
    fileName = './' + fileName;
    train_x = []
    train_y = []
#    fileIn = open('./trainSet.txt')
    fileIn = open(fileName)
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])]) #为什么第一列有个1.0
        #train_x.append([float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()

startTime = time.time()

## step 1: load data
print
"step 1: load data..."
#train_x, train_y = loadData()
train_x, train_y = loadData('trainSet.txt')
test_x, test_y =   loadData('testSet.txt')
print (test_x);

## step 2: training...
print
"step 2: training..."
opts = {'alpha': 0.01, 'maxIter': 3000, 'optimizeType': 'myGradDescent'}# gradDescent  myGradDescent neuralNetwork
opts_neural = {'alpha': 0.01, 'maxIter': 500}
optimalWeights = lR.trainLogRegres(train_x, train_y, opts)
#weights_1,weights_2 =  lR.trainNeuralNetwork(train_x, train_y, opts_neural)

endTime = time.time()
print('trainging time is', endTime-startTime)

## step 3: testing
print
"step 3: testing..."
accuracy = lR.testLogRegres(optimalWeights, test_x, test_y)
print('accuracy = ',accuracy)

#accuracy = lR.testNeuralNetwork(weights_1, weights_2, test_x, test_y)
#print('accuracy = ',accuracy)

# step 4: show the result
print
"step 4: show the result..."
print
'The classify accuracy is: %.3f%%' % (accuracy * 100)
lR.showLogRegres(optimalWeights, train_x, train_y)


