'''
通过调用摄像头，利用mediapipe识别手部地标，识别出的关键点信息并展示。按w键之后计算当前帧21个关节点相邻两个间的距离，
五指尖端两两间距离，对数据做标准化。计算相邻两条变间相互的夹角，
将这些信息并保存到csv文件中，
按w键开始保存，按q键停止保存并将数据存入
# '''

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# 获取手部的类别标签
from func_score import get_pairs, get_triples, distance, angle
from hand_realtime_classify import cv2ImgAddText


def get_hand_data(label, database_path):
    # label = input('请输入录数据的类型：')  # 0: 捏指 1：握拳 2:五指张开 3：半握 4：五指并拢

    # 创建一个Hand对象，用于检测和识别手部
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, model_complexity=1,
                           min_tracking_confidence=0.5)

    # 创建一个Drawing对象，用于绘制手部地标
    mp_drawing = mp.solutions.drawing_utils

    # 创建一个VideoCapture对象，用于获取摄像头的视频流
    cap = cv2.VideoCapture(0)

    # 创建一个空的DataFrame，用于存储手部地标的信息
    df = pd.DataFrame()

    # 创建一个标志变量，用于控制是否保存数据
    save_data = False

    # 获得手部关节对与三元组
    landmark_pairs = get_pairs()
    landmark_triples = get_triples()
    num=0

    # 循环处理每一帧图像

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

                # 如果需要保存数据
                if save_data:
                    # 获取手部地标的三维坐标（归一化到[0,1]范围）
                    landmarks = hand_landmarks.landmark

                    # 计算手部地标对之间的欧氏距离，并将其作为特征向量的一部分
                    distances = []
                    for pair in landmark_pairs:
                        p1 = landmarks[pair[0]]
                        p2 = landmarks[pair[1]]
                        dist = distance(p1, p2)
                        distances.append(dist)

                    # 计算手部地标三元组之间的夹角，并将其作为特征向量的另一部分
                    angles = []
                    for triple in landmark_triples:
                        p1 = landmarks[triple[0]]
                        p2 = landmarks[triple[1]]
                        p3 = landmarks[triple[2]]
                        ang = angle(p1, p2, p3)
                        angles.append(ang)

                    # 将距离和角度合并为一个特征向量，并添加到DataFrame中
                    row = np.array(distances + angles)
                    df = df._append(pd.Series(row), ignore_index=True)

                    # 获取手部的左右信息（0表示左手，1表示右手）
                    handedness = results.multi_handedness[0].classification[0].label

                    # 添加左右信息到DataFrame中
                    df['handedness'] = handedness

                    # 添加类别标签到DataFrame中
                    df['label'] = label
                    num+=1
                    print(num)

                # 等待按键输入
                # 如果按下w键，开始保存数据
                if cv2.waitKey(1) & 0xFF == ord('w'):
                    save_data = True

                # 如果按下q键，停止保存数据，并将DataFrame保存到一个文件中（你需要自己指定文件名）
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    save_data = False

                    # # 添加列名到DataFrame中（你需要自己定义）
                    # columns = ['d' + str(i) for i in range(15)] + ['a' + str(i) for i in range(18)] + ['handedness', 'label']
                    # df.columns = columns

                    # 保存DataFrame到文件中，使用mode='a'表示追加模式，header=False表示不写入列名
                    df.to_csv(database_path, index=False, mode='a', header=False)

                    # 清空df中的数据，保留列名
                    df = df.drop(df.index)
                # 在图像上显示左右手的分类结果

        # 将图像转换为BGR格式，并显示在窗口中
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not save_data:
            image = cv2ImgAddText(image, '按w键记录数据，按e退出', 10, 30, textSize=40, textColor=(0, 128, 255))
        else:
            image = cv2ImgAddText(image, '按q键停止', 10, 30, textSize=40, textColor=(0, 128, 255))
        cv2.imshow('Hand Recognition', image)
        # 如果按下esc键，退出循环
        if cv2.waitKey(1) & 0xFF==ord('e'):
            break

    # 释放摄像头资源，并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_hand_data(9, database='zero_nine')