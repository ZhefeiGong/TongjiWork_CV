# @author    : gonzalez
# @time      : 2022.4.15
# @function  : K-means聚类算法:1.确定聚类中心 2.确定每个数据所属类

import numpy as np
import matplotlib.pyplot as plt
import  math

# 计算两点距离-->直接使用欧式距离n维
def distance(node1, node2):
    distance = 0.0
    for a, b in zip(node1, node2):                   # 求出两个node间的n维欧拉距离
        temp = int(a)-int(b)
        distance += math.pow(temp, 2)
    return math.sqrt(distance)

# KMeans类
# 返回self.centerArray --> 类中心坐标
# 返回self.sortArr     --> 原始数据分类情况
# 返回self.count       --> 迭代次数
class KMeans:
    #初始化
    def __init__(self, num=1):
        assert num > 0
        self.clusterNum = num

    # 寻找array中距离最远的点-->用于初始化聚类中心结点
    def farthest(self, centerArray, array):
        targetSite = [0, 0]
        max_dis = 0
        for each in array:
            dis = 0
            for i in range(centerArray.__len__()):
                dis = dis + distance(centerArray[i], each)
            if dis > max_dis:
                max_dis = dis
                targetSite = each
        return targetSite

    # 寻找array中距离最近的点
    def closest(self,site, centerArray):
        targetSite = 0
        min_dis = distance(site, centerArray[0])
        i = 0
        for each in centerArray:
            dis = distance(site, each)
            if dis < min_dis:
                min_dis = dis
                targetSite = i
            i += 1
        return targetSite

    # 计算集合中心
    def means(self,sortArr,array,cluster):
        result = []
        num = 0
        for j in range(len(array[0])):
            result.append(0)
        for i in range(len(sortArr)):
            if(sortArr[i] == cluster):
                num += 1
                for j in range(len(array[i])):
                    result[j]+=array[i][j]
        for i in range(len(result)):
            result[i]=result[i]/num
        return result                                                            # 返回集合中心的坐标

    # 返回某一类有多少个
    def eachClusterNum(self,sortArr,cluster):
        num = 0
        for i in range(len(sortArr)):
            if(sortArr[i]==cluster):
                num += 1
        return num


    # 执行操作
    def operate(self,array):
        # 初始化
        center = np.random.randint(array.__len__() - 1)                           # 随机选择第一个簇
        self.centerArray = np.array([array[center]])                              # 聚类中心
        self.sortArr = []                                                         # 记录结果
        for i in range(self.clusterNum - 1):
            k = self.farthest(self.centerArray, array)                            # 找到离得最远的元素
            self.centerArray = np.concatenate([self.centerArray, np.array([k])])  # 对矩阵进行拼接

        # 迭代聚类
        self.count = 0
        self.centerArray = np.array(self.centerArray, dtype=float)                # 转换为浮点型矩阵
        while (True):
            self.count += 1                                                       # 记录迭代次数
            # 根据现有聚类中心进行分类
            for each in array:
                self.sortArr.append(self.closest(each, self.centerArray))         # 找到最近的点进行分类
            # 更新迭代聚类中心
            exact = 0
            for k in range(self.centerArray.__len__()):
                if self.eachClusterNum(self.sortArr, k) != 0.0:                   # 不为空才传入求值-->避免means函数RuntimeError
                    temp = self.means(self.sortArr, array, k)                     # means need to be changed
                    if (temp == self.centerArray[k]).all():                       # 判断两矩阵是否一致
                        exact += 1
                    self.centerArray[k] = temp
            # 聚类中心恒定 --> k-means结束
            if exact >= self.clusterNum:
                break
            else:
                self.sortArr = []


# 主函数
if __name__ == "__main__":


    # 生成50个随机数据（x,y）-->矩阵存储
    array = np.random.randint(100, size=(100, 1, 2))[:, 0, :]     # 涉及切片相关知识[ : : ]...

    # 调用KMeans类
    clusterNum = 4                                                # 簇个数人为规定(variable parameter)
    my = KMeans(clusterNum)
    my.operate(array)
    print("遍历次数：",my.count)

    # 可视化展示
    col = ['HotPink', 'Aqua', 'Chartreuse', 'yellow', 'LightSalmon', 'blue', 'black', 'red', 'peru', 'violet']
    for i in range(clusterNum):
        plt.scatter(my.centerArray[i][0], my.centerArray[i][1], linewidth=10, color=col[i])   # 聚类中心
    for i in range(len(array)):
        plt.scatter(array[i][0],array[i][1],color=col[my.sortArr[i]])                         # 该类数据
    plt.show()












