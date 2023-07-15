'''all'''
import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from func_score import get_real_time, get_pairs, get_triples, evaluate_similarity, generate_feedback, \
    update_data, get_tri_weights, get_weights, angle
import warnings
from classify.CNN_upper import CNN
import matplotlib.pyplot as plt
import pandas as pd
from cfg_parameter import FixedSizeQueue

"""
这个是评分的主要函数，把分类和评分功能整合起来的代码，为了方便调试我直接设置了选择的模型，
要做ui的话记得改choice参数
"""
warnings.filterwarnings("ignore")  # 用来忽略依赖错误
# 创建一个归一化器对象
# scaler = MinMaxScaler()

standard_score = 80  # 设置评分阈值，小于这个评分则生成建议

weights = get_tri_weights()  # 获得权重
# weights = get_weights()
# 获得手部关节对与三元组
landmark_pairs = get_pairs()
landmark_triples = get_triples()


# 定义用于对齐的关键点的索引这里取手腕和四指根部
keypoints = [0, 5, 17]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 初始化3d界面

# df = pd.read_csv('score_sample\hand_rec.csv')
df = pd.read_csv('score_sample/hand_rec.csv')
rd = np.ones((21, 3))
sd = np.ones((21, 3))
connections = []
score_list = []  # 用来存储评分
# 初始化mediapipe手部地标模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, model_complexity=1,
                       min_tracking_confidence=0.5)

# 初始化pose对象，用于检测全身地标
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# 获取用户的输入，选择要调用的模型
# choice = input("Please choose a model to use: svm, mlp, knn or cnn.")

choice = 'cnn'  # 方便调试预设了，这里记得修改

if choice == "svm":
    model = load('SVM_model/svm_model.pkl')

elif choice == "mlp":
    model = load('prework/mlp_hand1.pkl')

elif choice == "knn":
    model = load('KNN_model/hand_rec.pkl')

elif choice == "cnn":
    # 载入CNN模型
    model = torch.load('CNN_model/upper_model.pth')
    model.eval()
    # 将模型移动到GPU上（如果有）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    print("Invalid choice. Please try again.")

print(choice + '模型加载完毕')
pred = None
pre_pred = None # 用于存储上一次预测结果
score_queue = FixedSizeQueue(8)  # 设置定长队列,这里设置8帧，即1/3秒

# 初始化参数
calc_score = True
calc_stop = False
score = 0
num = 0
# 从摄像头读取图像
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # 将图像转换为RGB格式，并翻转水平方向
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # 设置图像为不可写，以提高处理速度
    image.flags.writeable = False
    # 使用手部地标模型处理图像，并获取结果
    results = hands.process(image)
    pose_results = pose.process(image)
    # 将图像转换回BGR格式，并设置为可写
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 定义左右手的分类结果变量
    left_label = None
    right_label = None

    # 如果有检测到手部地标，绘制出来，并预测手势类别
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            upper_angles = []
            # 绘制手部地标和连接线
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # 获取手部地标的三维坐标（归一化到[0,1]范围）
            landmarks = hand_landmarks.landmark
            connections = mp_hands.HAND_CONNECTIONS
            distances, angles = get_real_time(landmarks, landmark_pairs, landmark_triples)
            # 获取手部的左右信息（0表示左手，1表示右手）
            handness = handedness.classification[0].label
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

                if handness == 'Left':
                    joints = [[14, 16, 22], [14, 16, 20], [14, 16, 18]]
                    # 遍历每个手腕关节
                    for joint in joints:
                        # 添加一组三个地标，表示手腕的角度
                        pose_triples.append((joint[0], joint[1], joint[2]))
                elif handness == 'Right':
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

                # 将距离和角度合并为一个特征向量，并添加到DataFrame中
                hand_data = np.array(distances + angles + upper_angles)



                # 根据用户的输入，调用相应的模型，并打印预测结果
                if choice != "cnn":

                    # 将数据reshape为(1,-1)的形状
                    hand_data = hand_data.reshape(1, -1)

                    pred = model.predict(hand_data)
                    # print(f"svm_pred: {svm_pred}")
                    # 根据左右信息，更新分类结果
                    if handness == 'Left':
                        left_label = pred
                    else:
                        right_label = pred

                elif choice == "cnn":

                    pred = model.predict(hand_data)
                    # print(f"cnn_pred: {cnn_pred}")
                    # 根据左右信息，更新分类结果
                    if handness == 'Left':
                        left_label = pred
                    else:
                        right_label = pred

                '''
                用于评分的部分
                '''
                # # print(pred)
                # # 若存在预测结果，则评分
                # # if pred is not None:
                # # 获取手部地标的坐标列表，每个坐标是一个(x,y,z)元组
                # landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                # # 将坐标列表转换为numpy数组，并添加一列类别标签
                # landmarks = np.array(landmarks)
                # # 定义一个空的列表，用于存储实时识别的坐标
                # real_time_data = [landmarks]
                # real_time_data = np.array(real_time_data)
                # real_time_data = np.squeeze(real_time_data)
                # # real_time_data是用于评分的坐标信息
                #
                # # 获得手部地标的角度等信息
                # landmarks_ang = hand_landmarks.landmark
                # _, angles = get_real_time(landmarks_ang, landmark_pairs, landmark_triples)  # 这里舍弃掉距离信息
                # realtime_angl = angles
                #
                # get_label = int(pred)  # 用来获得当前手部分类标签
                #
                # df1 = df[df.iloc[:, -1] == get_label]
                #
                # # 提取出关键点的坐标，并转换为numpy数组
                # all_data = df1.iloc[:].to_numpy()
                #
                # all_data = all_data[:, :-1]  # 去掉最后一个标签值
                #
                # all_angles_data = all_data[:, 63:]  # 得到全部角度信息，用于后续评分
                #
                # all_standard_data = all_data[:, 0:63]
                #
                # standard_data = all_standard_data.reshape(-1, 3)  # 将每三个值划分为一组
                # standard_data = np.array(standard_data)
                # n = len(standard_data) // 21
                # standard_data = np.stack(np.split(standard_data, n), axis=0)
                #
                # if calc_score:
                #     # print(real_time_data)
                #     score, min_distance_index, low_score_points, rd, sd = evaluate_similarity(real_time_data,
                #                                                                               standard_data,
                #                                                                               weights, keypoints,
                #                                                                               realtime_angl,
                #                                                                               all_angles_data)
                #
                #     if score < standard_score:
                #         feedback = generate_feedback(realtime_angl, all_angles_data, min_distance_index, weights,
                #                                      low_score_points)
                #         # 这里是用来生成建议和评分的地方，
                #         print(f"评分: {score:.2f}")
                #         print("改进建议:")
                #         for suggestion in feedback:
                #             print(suggestion)
                #     else:
                #         print(f"评分: {score:.2f}")
            if calc_stop:
                calc_stop = False
                calc_score = False
            key = cv2.waitKey(1) & 0xFF
            # 如果按下s键，开始计算评分
            if key == ord('s'):
                calc_score = True

            # 如果按下t键，停止评分
            if key == ord('t'):
                calc_stop = True

            if pre_pred == None:
                print('')

            # # 计算总分的部分
            # elif pred == pre_pred:   # 若类别不变
            #     if score != 0:
            #         score_queue.push(score)
            #         num = num+1
            #     if num == 8:  # 若队列满则存储前8个评分的平均
            #         score_list.append(score_queue.average)  # 向最终评分列表添加数据
            #         num = 0
            # elif pred != pre_pred:   # 若类别改变
            #     if score_queue.size() != 0:
            #         score_list.append(score_queue.average)
            #         num = 0
            #         score_queue.clear()
            #     if score != 0:
            #         score_queue.push(score)
            #         num = num + 1

            pre_pred = pred  # 存储上一次的预测结果
            '''
            score_list是平滑后的分数列表，最终得分可以根据这里的数据取平均得到
            '''
        # 在图像上显示左右手的分类结果
        cv2.putText(image, 'Left: ' + str(left_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, 'Right: ' + str(right_label), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, 'Current score: ' + str(score), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    update_data(ax, rd, sd, connections)
    # 显示图像
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
