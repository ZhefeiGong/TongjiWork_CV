# @author    : gonzalez
# @time      : 2022.4.25
# @function  : 运用聚类算法实现图像分割

import os
from gonKMeans import *
from gonHac import *
from skimage.util import img_as_float
from skimage import color
from skimage import transform

# 对RGB特征进行聚类
def featureColor(img):
    height,width,channel=img.shape                                                   # 获取图像的高度，宽度和通道数
    img = img_as_float                                                               # 将图像像素转换成浮点数表示
    features = np.reshape(img,(height*width,channel))                                # 得到图像大小(高度*宽度)和颜色通道的特征
    return features

# 对RGB+位置特征进行聚类
def featureColorPosition(img):
    H, W, C = img.shape                                                              # 获取图像的高度，宽度和通道数
    imgFloat = img_as_float(img)                                                     # 将图像像素转换成浮点数表示
    features = np.zeros((H * W, C + 2))                                              # 预留出两个通道的位置，用于存放像素的位置特征
    position = np.dstack(np.mgrid[0:H, 0:W]).reshape((H * W, 2))                     # 得到每个像素的位置信息
    features[:, 0:C] = np.reshape(imgFloat, (H * W, C))                              # 将图像的每个像素的所有颜色通道作为特征通道
    features[:, C:C + 2] = position                                                  # 特征通道再额外拼接上各个像素的位置信息这两个通道
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0))   # 对图像特征中各个元素每个通道的数值进行中心化
    return features

# 对RGB+位置+xy梯度特征进行聚类
def featureColorPositionGradient(img):
    H, W, C=img.shape                                                                # 获取图像的高度，宽度和通道数
    imgFloat = img_as_float(img)                                                     # 将图像像素转换成浮点数表示
    position = np.dstack(np.mgrid[0:H, 0:W]).reshape((H*W, 2))                       # 得到每个像素的位置信息
    grayImg = color.rgb2gray(img)                                                    # 将图像换成灰度图形式
    gradient = np.gradient(grayImg)                                                  # 计算图像每个像素在X和Y方向上的梯度值
    gradient = np.abs(gradient[0])+np.abs(gradient[1])                               # 将两个方向上的梯度值进行相加
    features = np.zeros((H*W, C+3))                                                  # 将图像RGB三通道颜色传入feature
    features[:, 0:C] = np.reshape(imgFloat, (H*W, C))                                # 将图像的每个像素的所有颜色通道作为特征通道
    features[:, C:C+2] = position                                                    # 特征通道再额外拼接上各个像素的位置信息这两个通道
    features[:, C+2] = gradient.reshape((H*W))                                       # 特征通道再额外拼接上各个像素的两个方向上的梯度值之和这个通道
    features = (features-np.mean(features,axis=0))/(np.std(features, axis=0))        # 对图像特征中各个元素每个通道的数值进行中心化
    return features

# @function：计算分割方案和gt的重合度，作为该种分割方案的准确度
def computeAccuracy(maskGT,mask):
    accuracy = np.mean(mask == maskGT)                                               # 计算结果分割准确率
    return accuracy

# @function：加载所有原图像和所有GroundTruth的掩模
def loadFileName(dataDir):
    imgsName=[]
    for fname in sorted(os.listdir(os.path.join(dataDir))):                          # 在对应的文件夹中寻找对应
        if fname.endswith('.jpg'):                                                   # 原图像为jpg格式文件
            imgsName.append(fname[:-4])
    return imgsName

#运用KMeans聚类执行图像分割
def segmentKmeans(originalImage,groundTruth,clusterNum =3,scale=0.5,featureMethod=featureColorPosition):

    # 图片预处理（缩小、RGB、像素点位置、XY梯度）
    height,width,channel = originalImage.shape                                       # 获取原始图像的高度 宽度 通道数
    if scale > 0:
        imgSmaller = transform.rescale(originalImage, scale, multichannel=True)      # 调用缩小函数 进行缩小
    featureGraph = featureMethod(imgSmaller)                                         # 调用提取特征序列函数
    featureGraph = featureGraph*25                                                   # 特征序列归一化中心化后扩大一定倍数，增加计算精度

    # 执行KMeans聚类
    myKmeans = KMeans(clusterNum)                                                    # 调用KMeans类
    myKmeans.operate(featureGraph)                                                   # 执行均值聚类算法

    # 图片复原，产生分类结果图
    resultArray = np.array(myKmeans.sortArr)                                         # 将结果划归为矩阵
    resultArray = resultArray.reshape((imgSmaller.shape[:2]))                        # 矩阵转化为二维高*宽类型
    if scale > 0:
        resultArray = transform.resize(resultArray, (height, width), preserve_range=True) # 图形放大 回到原有比例
        resultArray = np.rint(resultArray).astype(int)                                    # 图像像素点所属簇名转化为int

    # 寻找最优概率值
    maxAccuray = 0
    for i in range(clusterNum):
        maskGraph = (resultArray==i).astype(int)                      # 选择不同的类作为主要内容（白色）
        accuracy = computeAccuracy(maskGraph, groundTruth)            # 求出不同情况下的准确率
        maxAccuray = max(accuracy,maxAccuray)                         # 统计准确率的最高情况
    for i in range(clusterNum):                                       # 找到统计的最高情况下的二值化图像maskGraph
        maskGraph = (resultArray==i).astype(int)
        accuracy = computeAccuracy(maskGraph, groundTruth)
        if accuracy == maxAccuray:                                    # 找到对应的最高准确率情况
            break
    # 返回值
    return maskGraph, maxAccuray, myKmeans.count                      # 返回结果二值化图像 最高准确率 KMeans计算次数

#运用HAC聚类执行图像分割
def segmentHac(originalImage,groundTruth,clusterNum =3,scale=0.5,featureMethod=featureColorPosition):

    # 图片预处理（缩小、RGB、像素点位置、XY梯度）
    height,width,channel = originalImage.shape                                       # 获取原始图像的高度 宽度 通道数
    if scale > 0:
        imgSmaller = transform.rescale(originalImage, scale, multichannel=True)      # 调用缩小函数 进行缩小
    featureGraph = featureMethod(imgSmaller)                                         # 调用提取特征序列函数
    featureGraph = featureGraph*25                                                   # 特征序列归一化中心化后扩大一定倍数，增加计算精度

    # 执行HAC层次聚类
    myKmeans = HierArchCal(clusterNum)                                               # 调用HAC类
    myKmeans.operate(featureGraph)                                                   # 执行层次聚类算法

    # 图片复原，产生分类结果图
    resultArray = np.zeros(len(featureGraph))                                        # 构造高*宽结果图
    for i, node in enumerate(myKmeans.HacNodes):                                     # 将层次树中的剩余簇对应的像素值在结果图中找到对应点
        listTemp = node.children
        for j in listTemp:
            resultArray[j] = i                                                       # 对结果图中的类进行赋值
    resultArray = resultArray.reshape((imgSmaller.shape[:2]))                        # 构造二维高宽结果图


    if scale > 0:
        resultArray = transform.resize(resultArray, (height, width), preserve_range=True) # 图形放大 回到原有比例
        resultArray = np.rint(resultArray).astype(int)                                    # 图像像素点所属簇名转化为int

    # 寻找最优概率值
    maxAccuray = 0
    for i in range(clusterNum):
        maskGraph = (resultArray==i).astype(int)                      # 选择不同的类作为主要内容（白色）
        accuracy = computeAccuracy(maskGraph, groundTruth)            # 求出不同情况下的准确率
        maxAccuray = max(accuracy,maxAccuray)                         # 统计准确率的最高情况
    for i in range(clusterNum):                                       # 找到统计的最高情况下的二值化图像maskGraph
        maskGraph = (resultArray==i).astype(int)
        accuracy = computeAccuracy(maskGraph, groundTruth)
        if accuracy == maxAccuray:                                    # 找到对应的最高准确率情况
            break
    # 返回值
    return maskGraph, maxAccuray                                      # 返回结果二值化图像 最高准确率

# 根据不同的数据进行不同的比例缩小
# KMean（method=0） --》 250*250=62500个像素点
# HAC  （method!=0）--》 350个像素点
def chooseScale(img,method):
    height, width, channel = img.shape
    scale = 0.5                                                       # 保留小数点后三位
    if method == 0:
        scale=round(math.sqrt(250*250/height/width),3)                # K-Means均值聚类算法
    else:
        scale=round(math.sqrt(350/height/width), 3)                   # HAC层次聚类算法
    return scale