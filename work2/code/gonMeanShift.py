# @author    : gonzalez
# @time      : 2022.4.24
# @function  : 均值漂移算法：1.计算图像的中心（平均值） 2.将窗口移动到中心 3.迭代以上步骤直至收敛
import numpy as np
import matplotlib.pyplot as plt

"""
均值漂移聚类实现步骤：
1、在未被分类的数据点中随机选择一个点作为中心点；
2、找出离中心点距离在带宽之内的所有点，记做集合M，认为这些点属于簇c。
3、计算从中心点开始到集合M中每个元素的向量，将这些向量相加，得到偏移向量。
4、中心点沿着shift的方向移动，移动距离是偏移向量的模。
5、重复步骤2、3、4，直到偏移向量的大小满足设定的阈值要求，记住此时的中心点。
6、重复1、2、3、4、5直到所有的点都被归类。
7、分类：根据每个类，对每个点的访问频率，取访问频率最大的那个类，作为当前点集的所属类。
"""


class meanShift:

    # 构造函数
    def __init__(self, radius=1):
        assert radius > 0
        self.radius = radius                     # 初始化均值漂移聚类的半径值

    # 根据频率对当前类进行划分
    def clustering(self,data):
        t = []
        for cluster in self.clusters:
            cluster['data'] = []
            t.append(cluster['frequency'])
        t = np.array(t)
        for i in range(len(data)):               # 进行聚类
            column_frequency = t[:, i]
            cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
            self.clusters[cluster_index]['data'].append(data[i])

    # 显示聚类结果
    def show_clusters(self,showData):
        colors = 10 * ['r', 'g', 'b', 'k', 'y']  # 定义可视化的点颜色
        plt.figure(figsize=(5, 5))
        plt.xlim((-8, 8))                        # 定义X轴的坐标范围
        plt.ylim((-8, 8))                        # 定义Y轴的坐标范围
        plt.scatter(showData[:, 0], showData[:, 1], s=20)
        theta = np.linspace(0, 2 * np.pi, 800)
        for i in range(len(self.clusters)):      # plt可视化显示点
            cluster = self.clusters[i]
            data = np.array(cluster['data'])
            plt.scatter(data[:, 0], data[:, 1], color=colors[i], s=20)
            centroid = cluster['centroid']
            plt.scatter(centroid[0], centroid[1], color=colors[i], marker='x', s=30)
            x, y = np.cos(theta) * self.radius + centroid[0], np.sin(theta) * self.radius + centroid[1]
            plt.plot(x, y, linewidth=1, color=colors[i])
        plt.show()                               # 触发可视化

    # 执行均值漂移聚类
    def mean_shift(self, data):
        self.clusters = []
        for i in range(len(data)):
            cluster_centroid = data[i]
            cluster_frequency = np.zeros(len(data))
            # 在半径内寻找对应的点
            while True:
                temp_data = []
                for j in range(len(data)):
                    v = data[j]
                    if np.linalg.norm(v - cluster_centroid) <= self.radius: # 处理半径内的点
                        temp_data.append(v)
                        cluster_frequency[i] += 1
                # 更新质心
                old_centroid = cluster_centroid
                new_centroid = np.average(temp_data, axis=0)
                cluster_centroid = new_centroid
                # 找到对应的模式
                if np.array_equal(new_centroid, old_centroid):
                    break# 弹出while循环
            # 合并一样的簇
            has_same_cluster = False
            for cluster in self.clusters:
                if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= self.radius:
                    has_same_cluster = True
                    cluster['frequency'] = cluster['frequency'] + cluster_frequency
                    break
            if not has_same_cluster:
                self.clusters.append({
                    'centroid': cluster_centroid,
                    'frequency': cluster_frequency
                })
        # 调用聚类函数
        self.clustering(data)


if __name__ == "__main__":

    #生成50个数据 显示结果
    arrayExample = np.array([
        [-6.1, -1.0], [-3.4, -1.6], [2.5, 1], [-1.2, -1], [2.9, -0.8], [2, 2.2],
        [-5, -3.5], [-3.5, -7], [-2.7, -4.5],[-2, -4.5], [-2.9, -2.9], [-0.4, -4.5],
        [-1.4, -6.5], [-1.6, -2], [-1.5, -1.3],[-6.5, -6.1], [-0.6, -1], [6, -1.6],
        [-4.8, -1], [-2.4, -0.6], [-3.5, 0],[-0.2, 4], [0.9, 1.8], [1, 2.2],
        [3.1, 7.8], [1.1, 3.4], [1, 4.5],[6.8, 0.3], [2.2, 1.3], [2.9, 0],
        [7.7, 1.2], [3, 3], [3.4, 2.8],[3, 5], [5.4, 1.2], [6.3, 2],
        [1.1, 2.2], [4.0, 3.6], [2, 0.7], [2.8, 7.3], [-2.2, 0.3], [2.9, -2],
        [4.4, -1.5], [2.6, -2], [-3.2, -5.3], [-2.5, 3.1], [-2.6, -1.7], [0, -6.6],
        [-1.6, -5.8], [1, 3.6]
    ])

    #调用均值漂移算法类
    my = meanShift(2.5)
    my.mean_shift(arrayExample)
    my.show_clusters(arrayExample)
