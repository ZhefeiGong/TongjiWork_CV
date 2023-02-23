'''
@ author : zhefei.gong
@ time   : 2022.3.10
'''

#导入一些必要的包
from time import sleep
import numpy as np
import cv2
import tensorflow
import pyautogui

cap = cv2.VideoCapture(0)                                              #使用笔记本的摄像头获取数据
labels = ['Up', 'Down', 'Left', 'Right', 'Nothing']                    #几标志移动的标签
model = tensorflow.keras.models.load_model('keras_model.h5')           #打开teachable machine训练好的模型

while True:
    ret, image = cap.read()
    if ret:                                                            #若成功读取
        image = cv2.flip(image, 1)                                     #摄像头图像翻转
        cv2.imshow("Frame", image)                                     #视频显示图像

        cv2.namedWindow("sign", cv2.WINDOW_AUTOSIZE)                   #开一个新的桌面窗口
        cv2.moveWindow("sign", 20, 20)                                 #调整新窗口位置

        img = cv2.resize(image, (224, 224))                            #修改获取到的图片规格
        img = np.array(img, dtype=np.float32)                          #把图像转换为numpy数组
        img = np.expand_dims(img, axis=0)
        img = img / 255                                                #归一化数据
        prediction = model.predict(img)                                #调用已经训练好的模型进行预测
        predicted_class = np.argmax(prediction[0], axis=-1)            #返回预测结果最大值的索引
        predicted_class_name = labels[predicted_class]                 #获取结果名称

        current_pos = pyautogui.position()                             #调用鼠标
        current_x = current_pos.x
        current_y = current_pos.y

        print(predicted_class_name)                                    #命令行打印预测结果
        src = cv2.imread("picture/xiaoxin.jpg")                        #图片识别结果
        if predicted_class_name == 'Nothing':
            src = cv2.imread("picture/xiaoxin.jpg")
            sleep(1)
        elif predicted_class_name == 'Left':
            src = cv2.imread("picture/left.jpg")
            pyautogui.moveTo(current_x - 60, current_y, duration=1)
            sleep(1)
        elif predicted_class_name == 'Right':
            src = cv2.imread("picture/right.jpg")
            pyautogui.moveTo(current_x + 60, current_y, duration=1)
            sleep(1)
        elif predicted_class_name == 'Down':
            src = cv2.imread("picture/down.jpg")
            pyautogui.moveTo(current_x, current_y + 60, duration=1)
            sleep(1)
        elif predicted_class_name == 'Up':
            src = cv2.imread("picture/up.jpg")
            pyautogui.moveTo(current_x, current_y - 60, duration=1)
            sleep(1)
        cv2.imshow("sign", src)                                        #显示对应图片

    if cv2.waitKey(1) & 0xFF == ord('q'):                              #当按下‘q’时停止显示q
        break

cap.release()                                                          #关闭连接
cv2.destroyAllWindows()