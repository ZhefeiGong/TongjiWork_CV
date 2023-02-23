# @author    : gonzalez
# @time      : 2022.4.24
# @function  : 运用KMeans聚类算法实现图像分割

from toolFunction import *
from PIL import Image
import time


originalImageDir = "../data/imgs/"                                                             # 原始图片所处路径
groundTruthImageDir = "../data/gt/"                                                            # gt图片所处路径
myKmeansDir1 = "../data/myKMeans/featureColorPosition/"                                        # 特征提取方法一保存路径
myKmeansDir2 = "../data/myKMeans/featureColorPositionGradient/"                                # 特征提取方法二保存路径

segmentNum = 3                                                                                 # 簇的个数
scale = 0.5                                                                                    # 初始化缩小比例
featureMethod = featureColorPosition                                                           # 采用的特征提取方式


if __name__ == "__main__":
    myKmeansDir = myKmeansDir1                                                                 # 选择不同的保存路径
    imgsName = loadFileName(originalImageDir)                                                  # 查找对应文件夹中需要处理的照片
    meanAccuracy = 0                                                                           # 存储最后的平均准确率
    for i in range(len(imgsName)):                                                             # 遍历所有照片文件
        # 打开照片
        originalImage = np.array(Image.open(originalImageDir+imgsName[i]+'.jpg'))              # 打开原始图片
        groundTruth = np.array(Image.open(groundTruthImageDir+imgsName[i]+'.png'))             # 打开gt图片
        scale=chooseScale(originalImage, 0)                                                    # 根据图片确定缩小比例
        # 执行聚类
        startTime = time.time()                                                                # 记录开始时间
        mask, accuracy, count = segmentKmeans(originalImage,groundTruth,segmentNum,scale,featureMethod) # 调用聚类分割
        endTime = time.time()                                                                  # 记录结束时间
        # 显示信息
        print('第%d张图像(%d * %d)--遍历次数：%d  准确率：%0.4f  耗时：%0.4fs  缩小比例：%0.3f'%(i+1, len(originalImage), len(originalImage[0]), count, accuracy, (endTime-startTime), scale))
        meanAccuracy += accuracy                                                               # 统计平均准确率情况
        # 保存图像
        mask = (mask == 0).astype(int)                                                         # 黑白区域反向
        plt.imshow(mask, cmap='binary')
        plt.axis('off')
        plt.savefig(myKmeansDir+'GONG-'+imgsName[i]+'.png')                                    # 保存对应图片
    print("平均识别准确率为：%0.4f" %(meanAccuracy/len(imgsName)))                                 # 显示平均准确率
