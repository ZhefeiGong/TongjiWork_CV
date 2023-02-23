# @author   : gonzalez
# @time     : 2022.5.22
# @function : 基于HOG方向梯度直方图优化SVM检测结果
# @notice   : 只需要调用HOG类对原始数据集进行处理即可，直接喂入gonSVM/gonKNN预测即可

import math
import cv2
import numpy as np
from tqdm import *
import os
import tools
import gonKNN
import gonSVM


# @function : 提取图像HOG特征
class HOG:
    # @function : 初始化函数
    def __init__(self):
        print('hello HOG')

    # @function : 对一个灰度图像image提取hog信息
    # @graph:
    #    Y--
    # X  *************************
    # |  *                       *
    #    *                       *
    #    *                       *
    #    *                       *
    #    *                       *
    #    *************************
    def hogImage(self, image, pixelsPerCell=(8, 8), cellsPerClock=(2,2), stride=8):

        # 数据初始化
        cellX, cellY = pixelsPerCell   # hog直方图提取过程中每个cell的规模
        blockX,blockY = cellsPerClock  # hog直方图提取过程中每个block的规模
        imageX,imageY = image.shape    # 提取特征的图像情况--> X表示行 Y表示列
        cellsNumX = int(np.floor(imageX // cellX))   # cell在x方向的数量
        cellsNumY = int(np.floor(imageY // cellY))   # cell在y方向的数量
        blocksNumX = cellsNumX - blockX + 1          # 滑窗取block，x方向的block数量
        blocksNumY = cellsNumY - blockY + 1          # 滑窗取block，y方向的block数量
        gradientX = np.zeros((imageX, imageY), dtype='float32') # 初始化X方向梯度存储数组
        gradientY = np.zeros((imageX, imageY), dtype='float32') # 初始化Y方向梯度存储数组
        gradient = np.zeros((imageX,imageY,2), dtype='float32') # 存储各个方向的梯度方向和梯度幅值

        # 整体梯度计算
        eps = 1e-5                         # 避免出现分母为0的误差值
        for i in range(1, imageX-1):       # X方向遍历
            for j in range(1, imageY-1):   # Y方向遍历
                gradientX[i][j] = image[i][j-1] - image[i][j+1] # X方向水平梯度
                gradientY[i][j] = image[i+1][j] - image[i-1][j] # Y方向垂直梯度
                gradient[i][j][0] = np.arctan(gradientY[i][j]/(gradientX[i][j]+eps))*180/math.pi # 算出该点的角度值
                if gradientX[i][j] < 0:
                    gradient[i][j][0] += 180                                          # 角度转换至0-180度的区间
                gradient[i][j][0] = int(gradient[i][j][0]+360) % 360                  # 保证梯度方向为正
                gradient[i][j][1] = np.sqrt(gradientY[i][j]**2+gradientX[i][j]**2)    # 计算梯度幅值

        # 各个模块计算
        normalisedBlock = []                                                                       # 存储最终结果
        for y in range(blocksNumY):
            for x in range(blocksNumX):
                block = gradient[y*stride:y*stride+blockY*stride, x*stride: x*stride+blockX*stride]# 分离出一个block单独计算
                histogramBlock =[]                                                                 # 存储该block的直方图
                eps = 1e-5                                                                         # 偏差参数 避免出现分母为0的情况
                for n in range(blockY):
                    for m in range(blockX):
                        cell = block[n * stride:(n + 1) * stride, m * stride:(m + 1) * stride]     # 在原有的block中分离出一个cell
                        histogramCell = np.zeros(8, dtype='float32')                               # 初始化cell的梯度直方图结果
                        for p in range(cellY):
                            for q in range(cellX):
                                histogramCell[int(cell[q][p][0]/45)] += cell[q][p][1]              # 为所属方向上的梯度添加自己的值
                        if len(histogramBlock) == 0:
                            histogramBlock = histogramCell
                        else:
                            histogramBlock = np.hstack((histogramBlock, histogramCell))
                histogramBlock = np.array(histogramBlock)                                          # 划归为矩阵
                histogramBlock = histogramBlock / np.sqrt(histogramBlock.sum()**2+eps)             # 直方图数据归一化
                if len(normalisedBlock) == 0:
                    normalisedBlock = histogramBlock                                               # 若为空则初始赋值
                else:
                    normalisedBlock = np.hstack((normalisedBlock, histogramBlock))                 # 添加至最后的归一化数组数据中

        return normalisedBlock                                                                     # 返回每张图像hog之后的结果数组

    # @function : 对原始n*3073的数据进行处理
    def hogProcess(self, originalData, fileName='data'):
        processedData=[]                                            # 存储最终结果
        for data in tqdm(originalData):                             # 显示对应进度条
            image = np.reshape(data.T, (32, 32, 3))                 # 一维组转化为图片形式数据
            grayImage= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255. # 转化为灰度图
            hogEach = self.hogImage(grayImage)                      # 提取灰度图hog特征
            if len(processedData) == 0:                             # 若数组为空
                processedData = hogEach                             # 若为空则初始赋值
            else:                                                   # 若数组不为空
                processedData = np.hstack((processedData, hogEach)) # 添加至最后的归一化数组数据中
        processedData = np.array(processedData).astype('float32')   # 转化为float型矩阵
        processedData = np.reshape(processedData, (-1, 32*9))       # 转化为n*特征数的数据集
        np.save(fileName+'.npy', processedData)                     # 保存在npy文件，便于下次使用
        return processedData                                        # 返回hog数据

    # @function : 数据处理
    def hogGetData(self, dataSet, fileName='0'):
        if os.path.exists(fileName+'.npy'):
            # 已经存在HOG特征提取的结果，则直接使用即可。避免再次HOG提取占用大量时间
            # print('existed')                          # 显示存在
            return np.load(fileName+'.npy')             # 载入npy数据
        else:
            # 不存在HOG特征提取的结果，则直接需要首先加载原数据，再对原数据做HOG特征提取获得梯度特征，最终是使用每张图片的梯度特征训练SVM并进行预测
            # print('not existed')                      # 显示不存在
            return self.hogProcess(dataSet, fileName)   # 进行计算


# @function : main主函数
if __name__ == '__main__':

    # HOG 应用于 KNN分类器
    # 数据处理
    print('开始执行KNN预测~~')
    trainData, trainLabels, testData, testLabels, verifyData, verifyLabels = tools.datasetGet(ratioTrain=50, ratioTest=50, ratioVerify=50, verifyNum=2)  # 获得数据集
    hog = HOG()
    train_data = hog.hogGetData(trainData, '../npyFile/trainData_KNN_HOG')                               # 获取hog数据
    test_data = hog.hogGetData(testData, '../npyFile/testData_KNN_HOG')                                  # 获取hog数据
    # 开始预测
    preLabels = gonKNN.gonKNN(7).predict(train_data, trainLabels, test_data)  # 执行预测
    # 结果展示
    print('预测准确度为：%.2f%%' % (np.mean(preLabels == testLabels) * 100))    # 显示结果


    # # HOG 应用于 SVM分类器
    # print('开始执行SVM预测~~')
    # trainData, trainLabels, testData, testLabels, verifyData, verifyLabels = tools.datasetGet()         # 数据集
    # hog = HOG()                                                                                         # 实例化HOG()
    # train_data = hog.hogGetData(trainData, '../npyFile/trainData_SVM_HOG')                              # 获取hog数据
    # test_data = hog.hogGetData(testData, '../npyFile/testData_SVM_HOG')                                 # 获取hog数据
    # svm = gonSVM.gonSVM()                                                                               # 实例化gonSVM()
    # lossHistory = svm.svmTrain(train_data, trainLabels, learningRate=1e-7, alpha=5e3, epoch=3000)       # 获取损失值数据
    # res = svm.svmPredict(test_data)                                                                     # 进行预测
    # print('测试集上准确率为:%.2f %%' % (np.mean(res == testLabels) * 100))                                 # 显示准确率结果
    # tools.graphShow(np.linspace(1, 3000, 3000), lossHistory, 'times', 'loss', 'LossValueGraph')         # 可视化结果