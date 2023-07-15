import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

'''
这个是我用来测试评分性能的，可以不用管，修改前的评分代码我已经整合到hand_realtime_classify里了，等我把评分细节改完整合进去就行
这个文件基本逻辑就是读取本地标准数据，识别手部地标，获得当前手部的信息（包括空间坐标和角度信息，空间坐标用于作图，角度信息用于评分）
里面涉及到评分和生成建议的逻辑，这里我封装到func_score里了，具体见那个文件

'''

from func_score import evaluate_similarity, generate_feedback, update_data, get_tri_weights, get_weights, get_pairs, \
    get_triples, get_real_time

label = 1  # 0：五指展开 1：五指并拢

# 创建一个Hand对象，用于检测和识别手部
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, model_complexity=1,
                       min_tracking_confidence=0.5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 初始化3d界面

# 获得手部关节对与三元组
landmark_pairs = get_pairs()
landmark_triples = get_triples()

# 创建一个Drawing对象，用于绘制手部地标
mp_drawing = mp.solutions.drawing_utils

# 创建一个VideoCapture对象，用于获取摄像头的视频流
cap = cv2.VideoCapture(0)

# 创建一个计数器，用于记录保存的数据条数
n = 0
standard_score = 80  # 设置评分阈值,低于这个分数开始生成建议

# 定义用于对齐的关键点的索引这里取手腕和四指根部
keypoints = [0, 5, 9, 13, 17]

weights = get_tri_weights()  # 获得权重（三元组）
# weights = get_weights()  # 获得权重（原始）

# 定义一个空的列表，用于存储标准数据的坐标
# standard_data = []


# 初始化参数
calc_score = True
calc_stop = False

# df = pd.read_csv('score_sample\hand_rec.csv')

df = pd.read_csv('score_sample/hand_rec.csv')

'''
录新的数据和测试代码
'''

rd = np.ones((21, 3))
sd = np.ones((21, 3))
connections = []

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
            connections = mp_hands.HAND_CONNECTIONS

            # 获取手部地标的坐标列表，每个坐标是一个(x,y,z)元组
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            # 将坐标列表转换为numpy数组，并添加一列类别标签
            landmarks = np.array(landmarks)
            # 定义一个空的列表，用于存储实时识别的坐标
            real_time_data = [landmarks]
            real_time_data = np.array(real_time_data)  # 评分用数据存入realtime_data
            real_time_data = np.squeeze(real_time_data)

            # landmarks = np.append(landmarks, label)
            # 获得手部地标的角度等信息
            landmarks_ang = hand_landmarks.landmark
            _, angles = get_real_time(landmarks_ang, landmark_pairs, landmark_triples)  # 这里舍弃掉距离信息
            realtime_angl = angles

            get_label = label  # 占位，用来获得当前手部分类标签

            df1 = df[df.iloc[:, -1] == get_label]

            # 提取出关键点的坐标，并转换为numpy数组
            all_data = df1.iloc[:].to_numpy()

            all_data = all_data[:, :-1]  # 去掉最后一个标签值

            all_angles_data = all_data[:, 63:]  # 得到全部角度信息，用于后续评分

            all_standard_data = all_data[:, 0:63]

            standard_data = all_standard_data.reshape(-1, 3)  # 将每三个值划分为一组
            standard_data = np.array(standard_data)
            n = len(standard_data) // 21
            standard_data = np.stack(np.split(standard_data, n), axis=0)

            # print('score_data', standard_data)
            # print('realtime_data', real_time_data)

            # break
            # print('加载成功')

            if calc_score:
                # print(real_time_data)
                score, min_distance_index, low_score_points, rd, sd = evaluate_similarity(real_time_data, standard_data,
                                                                                          weights, keypoints,
                                                                                          realtime_angl,
                                                                                          all_angles_data)

                if score < standard_score:
                    feedback = generate_feedback(realtime_angl, all_angles_data, min_distance_index, weights,
                                                 low_score_points)
                    print(f"评分: {score:.2f}")
                    print("改进建议:")
                    for suggestion in feedback:
                        print(suggestion)
                else:
                    print(f"评分: {score:.2f}")
                if calc_stop:
                    calc_stop = False
                    calc_score = False

            # 等待按键输入
            key = cv2.waitKey(1) & 0xFF

            # # 如果按下w键，开始保存数据
            # if key == ord('w'):
            #     save_data = True

            # 如果按下s键，开始计算评分
            if key == ord('s'):
                calc_score = True

            # 如果按下t键，停止评分
            if key == ord('t'):
                calc_stop = True
    update_data(ax, rd, sd, connections)
    # 显示图像
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
