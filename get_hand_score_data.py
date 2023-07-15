import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from func_score import get_real_time, get_pairs, get_triples
from hand_realtime_classify import cv2ImgAddText

"""
这个主要是用来获得手部地标的信息，用于评分，因为分类是直接用的模型，分类和评分的数据不一样，为了保证分类模型的简洁和评分准确，需要分开录取。
那个get_hand_data跟这个不一样，这个是获得评分的数据。
数据包含21个点的x,y,z坐标（63列），24个角度信息，和标识分类的的标签信息
我这里设定的是运行之后按’w‘键录一条数据，按一次录一条，不用录太多，一个动作几十条最多了
录完按esc退出
"""

def get_hand_score_data(label,database_path):

    # 获得手部关节对与三元组
    landmark_pairs = get_pairs()
    landmark_triples = get_triples()

    # 创建一个Hand对象，用于检测和识别手部
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, model_complexity=1,
                           min_tracking_confidence=0.5)
    # 创建一个Drawing对象，用于绘制手部地标
    mp_drawing = mp.solutions.drawing_utils
    # 创建一个VideoCapture对象，用于获取摄像头的视频流
    cap = cv2.VideoCapture(0)
    # 创建一个空的DataFrame，用于存储手部地标的信息
    df = pd.DataFrame()
    # 创建一个标志变量，用于控制是否保存数据
    save_data = False
    n = 0   # 计数器

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像转换为RGB格式，并翻转水平方向
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        # 将图像传入Hand对象，得到检测和识别的结果
        results = hands.process(image)

        # 如果有检测到手部
        if results.multi_hand_landmarks:
            # 遍历每一只手
            for hand_landmarks in results.multi_hand_landmarks:
                # 在图像上绘制三维关键点
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 获取手部地标的坐标列表，每个坐标是一个(x,y,z)元组
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # 获得手部地标的角度等信息
                landmarks_ang = hand_landmarks.landmark
                _, angles = get_real_time(landmarks_ang, landmark_pairs, landmark_triples)  # 这里舍弃掉距离信息

                # 将坐标列表转换为numpy数组，并添加一列类别标签
                landmarks = np.array(landmarks)
                landmarks = np.append(landmarks,angles)
                landmarks = np.append(landmarks, label)

                # 如果需要保存数据
                if save_data:
                    # 将numpy数组转换为Series，并添加到DataFrame中
                    series = pd.Series(landmarks)
                    df = df._append(series, ignore_index=True)

                    # 设置停止保存数据，并将DataFrame保存到一个文件中 mode='a'设置为追加模式
                    save_data = False

                    # 保存数据到本地
                    df.to_csv(database_path, index=False, mode='a', header=False)

                    # 计数器加一，并打印提示信息
                    n = n + 1
                    print('保存成功，当前第' + str(n) + '条')

                    # 使用df.drop方法来删除已经保存的数据，保持df为空
                    df = df.drop(df.index)
                    # series = series.drop

        # 如果按下w键，开始保存数据
        if cv2.waitKey(1) & 0xFF == ord('w'):
            save_data = True


        # 显示图像
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2ImgAddText(image, '按w记录一条数据，按e键退出', 10, 30, textSize=40, textColor=(0, 128, 255))
        cv2.imshow('Image', image)
        # 如果按下e键，开始保存数据
        if cv2.waitKey(1) & 0xFF==ord('e'):
            break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_hand_score_data(label=9, database='zero_nine')