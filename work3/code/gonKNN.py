# @author   : gonzalez
# @time     : 2022.5.19
# @function : 实现KNN分类器

import numpy as np
from collections import Counter
from tqdm import *
import matplotlib.pyplot as plt
import tools

# @function : KNN分类器实现
class gonKNN:
    # @function : 初始化
    def __init__(self, k):
        assert k > 0
        self.k = k                                                                         # 超参数

    # @function : 计算欧拉距离
    def eularDis(self, instance1, instance2):
        return np.sqrt(sum((instance1-instance2)**2))                                      # 返回欧拉距离

    # @function : 实现KNN分类器预测
    def predict(self, trainData, trainLables, testData):
        # 开始分类识别
        preLabels = []                                                                     # 存储KNN分类结果
        for eachTest in tqdm(testData):                                                    # 进度条显示预测成果
            # 分类计算
            distancesAll = [self.eularDis(eachTrain,eachTest) for eachTrain in trainData]  # 求得每个训练数据像对测试数据的距离
            kMinIndex = np.argsort(distancesAll)[:self.k]                                  # 求得距离最小的前k个训练数据的索引
            countRes = Counter(trainLables[kMinIndex])                                     # 对前k个训练数据的标签计数
            preLabels.append(countRes.most_common(1)[0][0])                                # 将出现频率最多的标签返回
        return preLabels                                                                   # 返回预测结果标签


if __name__ == '__main__':

    # 数据处理
    trainData, trainLabels, testData, testLabels, verifyData, verifyLabels = tools.datasetGet(ratioTrain=50, ratioTest=50, ratioVerify=50, verifyNum=2)          # 获得数据集

    # 开始预测
    print('开始执行KNN预测~~')
    train_data = tools.preProcessing(trainData)                                                                  # 数据预处理
    test_data = tools.preProcessing(testData)                                                                    # 数据预处理
    preLabels = gonKNN(7).predict(train_data, trainLabels, test_data)                                            # 执行预测

    # 结果展示
    print('预测准确度为：%.2f%%' % (np.mean(preLabels == testLabels) * 100))                                        # 显示结果


