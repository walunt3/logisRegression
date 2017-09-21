#################################################
# logRegression: Logistic Regression
# Author : zouxy
# Date   : 2014-03-02
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

import numpy as np
import matplotlib.pyplot as plt
import time


# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# train a logistic regression model using some optional optimize algorithm
# input: train_x is a mat datatype, each row stands for one sample
#        train_y is mat datatype too, each row is the corresponding label
#        opts is optimize option include step and maximum number of iterations
def trainNeuralNetwork(train_x, train_y, opts):
    # calculate training timeopts
    startTime = time.time()
    alpha = opts['alpha'];
    maxIter = opts['maxIter']

    numSamples, numFeatures = np.shape(train_x)
    weights_1 = np.random.randn(numFeatures, 1) * 0.01 #一层：w_1
    weights_2 = np.random.randn(1, 1) * 0.01  #二层：w_2

    for k in range(maxIter):
        output_1 = sigmoid(train_x * weights_1)  # + b
        error_1 = train_y - output_1  # dz=A-Y  error = -dz
        weights_1 = weights_1 + alpha * train_x.transpose() * error_1 / numSamples  # w = w - a * x * dz/m
        # 二层
        output_2 = sigmoid(output_1 * weights_2)
        error_2 = train_y - output_2
        weights_2 = weights_2 + alpha * output_1.transpose() * error_2 / numSamples

    print
    'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return weights_1,weights_2

def trainLogRegres(train_x, train_y, opts):
    # calculate training timeopts
    startTime = time.time()

    numSamples, numFeatures = np.shape(train_x)
    alpha = opts['alpha'];
    maxIter = opts['maxIter']
    #weights = np.ones((numFeatures, 1))
    #weights = np.random.randn(numFeatures, numSamples) * 0.01  # 只有一层 或 最后一层
    weights = np.random.randn(numFeatures, 1)*0.01 #只有一层 或 最后一层
    #weights_1 = np.random.randn(numFeatures, 1) * 0.01  #一层：w_1
    b = np.zeros((numSamples,1))

    # optimize through gradient descent algorilthm
    for k in range(maxIter):
        if opts['optimizeType'] == 'gradDescent':  # gradient descent algorilthm
            output = sigmoid(train_x * weights)
            error = train_y - output #dz=A-Y  error = -dz
            weights = weights + alpha * train_x.transpose() * error #w = w - a * x * dz/m
        elif opts['optimizeType'] == 'stocGradDescent':  # stochastic gradient descent
            for i in range(numSamples):
                output = sigmoid(train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose() * error
        elif opts['optimizeType'] == 'smoothStocGradDescent':  # smooth stochastic gradient descent
            # randomly select samples to optimize for reducing cycle fluctuations
            dataIndex = list(range(numSamples))
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del (dataIndex[randIndex])  # during one interation, delete the optimized sample
        elif opts['optimizeType'] == 'myGradDescent':
            output = sigmoid(train_x * weights ) #+ b
            error = train_y - output  # dz=A-Y  error = -dz
            weights = weights + alpha * train_x.transpose() * error / numSamples  #w = w - a * x * dz/m
            #b = b + alpha * np.sum(error) / numSamples
            #alpha = alpha - 1.0*k/(100*maxIter)
        else:
            raise NameError('Not support optimize method type!')
    print
    'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    #print (weights)
    return weights


# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy

#二层网络
def testNeuralNetwork(weights_1, weights_2, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        a1 = sigmoid(test_x[i, :] * weights_1)
        a2 = sigmoid(a1 * weights_2)
        predict = a2[0,0] > 0.5
        #predict = sigmoid(test_x[i, :] * weights_1)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# show your trained logistic regression model only available with 2-D data
def showLogRegres(weights, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = np.shape(train_x)
    if numFeatures != 3:
        print
        "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

        # draw all samples
    for i in range(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

            # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()