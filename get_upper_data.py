'''
这里录得是手部，包括手腕，小臂的角度信息，增添手腕动作的分类
按esc键退出
'''
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# 获取手部的类别标签
from func_score import get_pairs, get_triples, distance, angle

label = input('选择要录取的类别')  # 定义录取的类
classify_data_path = 'pose_data/upper_data.csv'  # 这里修改数据保存的文件，一组动作存到一个文件里，注意录之前改标签

# 创建一个Hand对象，用于检测和识别手部
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, model_complexity=1,
                       min_tracking_confidence=0.5)

# 初始化pose对象，用于检测全身地标
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
                # 获取手部地标的三维坐标（归一化到[0,1]范围）
                landmarks = hand_landmarks.landmark

                # 获取手部的左右信息（0表示左手，1表示右手）
                handedness = results.multi_handedness[0].classification[0].label

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
                    # 将距离和角度合并为一个特征向量，并添加到DataFrame中
                    row = np.array(distances + angles + upper_angles)
                    df = df.append(pd.Series(row), ignore_index=True)

                    # 添加左右信息到DataFrame中
                    df['handedness'] = handedness

                    # 添加类别标签到DataFrame中
                    df['label'] = label
                    print('记录成功')

            # 等待按键输入
            key = cv2.waitKey(1) & 0xFF
            # 如果按下w键，开始保存数据
            if key == ord('w'):
                save_data = True

            # 如果按下q键，停止保存数据，并将DataFrame保存到一个文件中（你需要自己指定文件名）
            if key == ord('q'):
                save_data = False

                # 保存DataFrame到文件中，使用mode='a'表示追加模式，header=False表示不写入列名
                df.to_csv(classify_data_path, index=False, mode='a', header=False)

                # 清空df中的数据，保留列名
                df = df.drop(df.index)

    # 将图像转换为BGR格式，并显示在窗口中
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Recognition', image)
    # 如果按下esc键，退出循环
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 释放摄像头资源，并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
