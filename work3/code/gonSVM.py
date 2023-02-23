# @author   : gonzalez
# @time     : 2022.5.19
# @function : 实现SVM分类器

import tools
import numpy as np
import matplotlib.pyplot as plt


# @function : SVM分类器实现
class gonSVM:
    # @function : 初始化
    def __init__(self):
        self.weight = None # 权重

    # @function : 计算SVM损失函数
    # @detail   : 使用SVM的loss函数进行计算
    # @formula  : 1/N*Sigma_i^N{L_i(f_i(x_i,weight),y_i)} + alpha*Sigma_k Sigma_l wight_kl^2
    def svmLossGradient(self, batchData, batchLabels, alpha, delta):

        # 初始化
        batchSize = len(batchLabels)

        # 计算损失
        loss = 0.0                                                                # 存储损失函数值
        scores = batchData.dot(self.weight)                                       # 矩阵点乘运算(x_i * weight)-->产生batchSite*10的二维矩阵 每一行表示每个测试数据的判定结果
        correctScore = scores[range(batchSize), list(batchLabels)].reshape(-1, 1) # 提取每行正确评判结果-->转换成batchSize行1列的正确评估值数据矩阵
        hingeValue = np.maximum(0, scores - correctScore + delta)                 # (s_j-s_i+delta)折页损失计算结果中小于0的结果用0代替
        hingeValue[range(batchSize), list(batchLabels)] = 0                       # 纠正评估为正确的结果值为0
        loss += np.sum(hingeValue)/batchSize                                      # 折页损失
        loss += alpha * np.sum(self.weight * self.weight)                         # 添加L2正则化

        # 计算梯度
        gradient = 0.0                                                            # 存储梯度结果矩阵 --> 3072*10
        maskArr = np.zeros(hingeValue.shape)                                      # hingeValue-->batchSize*10 的结果数组
        maskArr[hingeValue > 0] = 1                                               # 所有不为0的位置变为1-->即所有得分与标准得分差大于delta的评估结果
        maskArr[range(batchSize), list(batchLabels)] = -np.sum(maskArr, axis=1)   # 求出所有需要计算的非零评估位个数-->num(-x_i^Tw_y_i)
        gradient += batchData.T.dot(maskArr)/batchSize                            # 产生3073*10的求导结果矩阵
        gradient += 2 * alpha * self.weight                                       # 添加正则化求导结果

        return loss, gradient

    # @function : svm开始训练
    # @detail   : 采用==>SVM损失函数+L2正则化+微分分析计算梯度+随机梯度下降
    #           : 采用==>采用随机梯度下降方法进行实现
    def svmTrain(self, trainDataIn, trainLabels, learningRate=1e-3, delta=1.0, alpha=1.0, epoch=2000, batchSize=200):
        # 数据初始化
        trainData = np.hstack((trainDataIn, np.ones((trainDataIn.shape[0], 1))))  # 将W和b融合 --> n*1全1矩阵 --> 每个数据添加一个1
        trainNum, featureNum = trainData.shape
        classesNum = np.max(trainLabels) + 1
        # 权值初始化
        if self.weight == None:
            self.weight = 0.001 * np.random.randn(featureNum, classesNum)     # 生成3073*10的二维数组
        # 开始训练
        lossHistory = []                                                      # 用于存储迭代情况-->以展示随时函数迭代情况
        for i in range(epoch):                                                # 显示进度条
            indexBatch = np.random.choice(trainNum, batchSize, replace=False) # 在训练集中随机抽取batchSize个数据的索引值(互不相同)
            trainBatch = trainData[indexBatch]
            labelsBatch = trainLabels[indexBatch]
            loss, gradient = self.svmLossGradient(trainBatch, labelsBatch, alpha, delta) # 调用SVM类中函数 获取损失值及梯度值
            self.weight -= learningRate * gradient                                       # 沿梯度下降方向改变权值weight
            lossHistory.append(loss)                                                     # 存储迭代信息情况
            # 展示迭代情况
            if (i+1) % 100 == 0:                                                         # 每迭代100次 显示迭代情况
                print('迭代次数为: %d / %d  损失函数值为: %f' % (i+1, epoch, loss))          # 展示迭代情况
        return lossHistory                                                               # 返回迭代过程损失函数值变化情况 用于可视化

    # @function : svm开始预测
    def svmPredict(self, testDataIn):
        testData = np.hstack((testDataIn, np.ones((testDataIn.shape[0], 1))))  # 将W和b融合 --> n*1全1矩阵 --> 每个数据添加一个1
        scores = testData.dot(self.weight)                                     # 根据训练权重计算各类别得分
        res = np.argmax(scores, axis=1)                                        # 返回max值的索引即得分最高类的类标签
        return res


# @function : main主函数
if __name__ == '__main__' :

    # 数据处理
    trainData, trainLabels, testData, testLabels, verifyData, verifyLabels = tools.datasetGet()  # 数据集
    train_data = tools.preProcessing(trainData)                                                  # 数据预处理
    test_data = tools.preProcessing(testData)                                                    # 数据预处理

    # 开始执行
    svm = gonSVM()                                                                               # 实例化SVM对象
    lossHistory = svm.svmTrain(train_data, trainLabels, learningRate=1e-7, alpha=5e3, epoch=3000)# 记录损失情况
    res = svm.svmPredict(test_data)                                                              # 记录预测结果
    print('测试集上准确率为:%.2f %%' % (np.mean(res == testLabels)*100))                            # 计算最终结果

    # 可视化
    tools.graphShow(np.linspace(1, 3000, 3000), lossHistory, 'times', 'loss', 'LossValueGraph')  # 可视化结果

