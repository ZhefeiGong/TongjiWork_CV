### 基于TeachableMachine的分类识别模型控制鼠标实验

1. 本实验环境为
   * python:3.7
   * tensorflow-gpu:2.3.0
2. 由于TeachableMachine训练模型识别效果不佳，因此选择差异较大的四种物品分别代表控制鼠标的上下左右：
   * 胡萝卜：  上
   * 皮卡丘：  下
   * 鲨鱼包：  左
   * 抱枕：     右
   * 区域：     Nothing(即显示蜡笔小新图案)
3. picture文件夹中为需要显示的图片文件