'''
这里录得是手部，包括手腕，小臂的角度信息，增添手腕动作的分类
按esc键退出
'''
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# 获取手部的类别标签
from func_score import get_pairs, get_triples, distance, angle, get_real_time

label = input('选择要录取的类别')  # 定义录取的类
classify_data_path = 'score_data/upper_score_data.csv'  # 这里修改数据保存的文件，一组动作存到一个文件里，注意录之前改标签

# 创建一个Hand对象，用于检测和识别手部
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, model_complexity=1,
                       min_tracking_confidence=0.5)

# 初始化pose对象，用于检测全身地标
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
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
n = 0
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
    pose_results = pose.process(image)
    # 如果有检测到手部
    if results.multi_hand_landmarks:
        # 遍历每一只手
        for hand_landmarks in results.multi_hand_landmarks:
            # 在图像上绘制三维关键点

            # 如果需要保存数据
            if save_data:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                upper_angles = []

                # 获取手部的左右信息（0表示左手，1表示右手）
                handedness = results.multi_handedness[0].classification[0].label
                # 获取手部地标的坐标列表，每个坐标是一个(x,y,z)元组
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                # 获得手部地标的角度等信息
                landmarks_ang = hand_landmarks.landmark
                _, angles = get_real_time(landmarks_ang, landmark_pairs, landmark_triples)  # 这里舍弃掉距离信息

                # 提取估计结果中的三维关键点
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=pose_results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS)
                    landmarks_3d = []
                    pose_triples = []

                    for landmark_3d in pose_results.pose_world_landmarks.landmark:
                        landmarks_3d.append(landmark_3d)

                    if handedness == 'Left':
                        joints = [[14, 16, 22], [14, 16, 20], [14, 16, 18]]
                        # 遍历每个手腕关节
                        for joint in joints:
                            # 添加一组三个地标，表示手腕的角度
                            pose_triples.append((joint[0], joint[1], joint[2]))
                    elif handedness == 'Right':
                        joints = [[13, 15, 21], [13, 15, 19], [13, 15, 17]]
                        # 遍历每个手腕关节
                        for joint in joints:
                            # 添加一组三个地标，表示手腕的角度
                            pose_triples.append((joint[0], joint[1], joint[2]))

                    # 计算手部地标三元组之间的夹角，并将其作为特征向量的另一部分

                    for triple in pose_triples:
                        p1 = landmarks_3d[triple[0]]
                        p2 = landmarks_3d[triple[1]]
                        p3 = landmarks_3d[triple[2]]
                        ang = angle(p1, p2, p3)
                        upper_angles.append(ang)

                    # mp_drawing.draw_landmarks(image, landmark_3d, mp_pose.POSE_CONNECTIONS)

                if len(upper_angles) != 0:
                    # 将坐标列表转换为numpy数组，并添加一列类别标签
                    landmarks = np.array(landmarks)
                    landmarks = np.append(landmarks, angles)
                    landmarks = np.append(landmarks, upper_angles)
                    landmarks = np.append(landmarks, label)
                    # 将距离和角度合并为一个特征向量，并添加到DataFrame中

                    # 将numpy数组转换为Series，并添加到DataFrame中
                    series = pd.Series(landmarks)
                    df = df.append(series, ignore_index=True)

                    # 保存数据到本地
                    df.to_csv('score_sample\hand_landmarks.csv', index=False, mode='a', header=False)

                    # 计数器加一，并打印提示信息
                    n = n + 1
                    print('保存成功，当前第' + str(n) + '条')
                    n = 0
                    # 使用df.drop方法来删除已经保存的数据，保持df为空
                    df = df.drop(df.index)
                    series = series.drop

                    print('记录成功')
                save_data = False
    # 等待按键输入
    key = cv2.waitKey(1) & 0xFF
    # 如果按下w键，开始保存数据
    if key == ord('w'):
        save_data = True

    # 如果按下q键，停止保存数据，并将DataFrame保存到一个文件中（你需要自己指定文件名）
    if key == ord('q'):
        save_data = False

    # 将图像转换为BGR格式，并显示在窗口中
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Recognition', image)
    # 如果按下esc键，退出循环
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 释放摄像头资源，并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
