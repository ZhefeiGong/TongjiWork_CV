### 聚类算法实现图像分割

#### data:：存储图片数据

* gt：GroundTruth图片
* imgs：原始图片
* myHAC：存储使用HAC算法进行聚类的结果图
  * featureColorPosition：采用颜色+坐标位置特征提取方法结果图
  * featureColorPositionGradient：采用颜色+坐标位置+梯度特征提取方法结果图
* myKMeans：存储使用KMeans算法进行聚类的结果图
  * featureColorPosition：采用颜色+坐标位置特征提取方法结果图
  * featureColorPositionGradient：采用颜色+坐标位置+梯度特征提取方法结果图

#### code：存储代码

* gonHac.py：HAC聚类算法实现代码
* gonKMeans.py：K-Means聚类算法实现代码
* gonMeanShift.py：均值漂移聚类算法实现代码
* toolFunction.py：实现图像分割工具函数代码
* imageSegmentHAC.py：HAC实现图像分割实现代码
* imageSegmentKMeans.py：K-Means实现图像分割实现代码