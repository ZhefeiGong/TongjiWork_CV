# @author    : gonzalez
# @time      : 2022.4.14
# @function  : HAC聚类算法

import math
import numpy as np
from sklearn.datasets import load_iris                         # 鸢尾花数据集
import matplotlib.pyplot as plt

# 计算欧拉距离（n维）
def EulerDistance(point1, point2):
    distance = 0
    for a, b in zip(point1, point2):                             # 求出两个簇节点的相似度
        temp = int(a)-int(b)
        distance += math.pow(temp, 2)
    return math.sqrt(distance)

# 层次聚类树结点
class ClusterNode:
    def __init__(self, vec, id, children):
        self.vec = vec                                           # 存储该节点的值 用于此后的相似度衡量
        self.id = id                                             # 存储该节点的独一无二的id
        self.children = children                                 # 存储该节点的孩子结点的index

# 层次聚类
# 人为指定cluster个数
# 返回一个ClusterNode结点列表，每个结点的children含该类结点的索引
# 复杂度较高，为O(n^3)
class HierArchCal:
    def __init__(self, num=1):
        assert num > 0
        self.clusterNum = num                                    # 初始化簇个数

    def operate(self,data):
        nodes = [ClusterNode(vec=v, id=i,children=[i]) for i, v in enumerate(data)]
        pointNum, featureNum = np.shape(data)                    # 统计输入数据的点宽和点数
        currentId = -1                                           # 目前起初的ID
        while len(nodes) > self.clusterNum:                      # 只需要得到clusterNum个数据即可
            minDistance = np.inf
            nodesCurNum = len(nodes)
            closestPart = None
            for i in range(nodesCurNum - 1):                     # 求出所有簇中相似度最优的两个簇
                for j in range(i+1, nodesCurNum):
                    dis = EulerDistance(nodes[i].vec, nodes[j].vec)
                    if dis < minDistance:
                        minDistance = dis
                        closestPart = (i, j)
            (part1, part2) = closestPart
            (node1, node2) = nodes[part1], nodes[part2]          # 求出簇的参数
            newVec = [(node1.vec[i]*len(node1.children)+node2.vec[i]*len(node2.children))/(len(node1.children)+len(node2.children))
                      for i in range(featureNum)]                # 求均值
            newChildren = node1.children+node2.children          # 两个簇合并
            newNode = ClusterNode(vec=newVec, id=currentId, children=newChildren) # 产生新的节点
            currentId -= 1                                       # 改变独一无二的ID值
            del nodes[part2], nodes[part1]                       # 删除现有结点
            nodes.append(newNode)                                # 加入新结点

        self.HacNodes = nodes                                    # 最后结果节点

if __name__ == "__main__":

    # 层次聚类
    cluster = 4
    iris = load_iris()
    my = HierArchCal(cluster)
    my.operate(iris.data)

    # 可视化展示
    col = ['HotPink', 'Aqua', 'Chartreuse', 'yellow', 'LightSalmon', 'blue', 'black', 'red', 'peru', 'violet']
    for i, node in enumerate(my.HacNodes):
        listTemp = node.children
        for j in listTemp:
            plt.scatter(iris.data[j][0], iris.data[j][1], color=col[i])
    print("hello HAC!!!")
    plt.show()