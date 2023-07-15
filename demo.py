import cv2
import mediapipe as mp
import numpy as np
"""
这个是全身和手部的地标检测，后面扩展到上半身的化会用到这段代码，
可以跑一下看看效果，直接运行就可以
"""
# 初始化MediaPipe人体姿态估计
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化前一帧的三维关键点
prev_landmarks_3d = None


# 初始化mediapipe手部地标模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)                      #普通模型


# 定义函数：估计前后两帧之间的变换矩阵
def estimate_transform_matrix(prev_landmarks_3d, curr_landmarks_3d):
    # 将关键点转换为numpy数组
    prev_landmarks_3d = np.array(prev_landmarks_3d)
    curr_landmarks_3d = np.array(curr_landmarks_3d)

    # 计算两个关键点集的中心点
    prev_center = np.mean(prev_landmarks_3d, axis=0)
    curr_center = np.mean(curr_landmarks_3d, axis=0)

    # 将关键点集平移到中心点处
    prev_landmarks_3d_centered = prev_landmarks_3d - prev_center
    curr_landmarks_3d_centered = curr_landmarks_3d - curr_center

    # 计算旋转矩阵R和平移向量t，使得R*prev_landmarks_3d_centered + t = curr_landmarks_3d_centered
    H = prev_landmarks_3d_centered.T @ curr_landmarks_3d_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = curr_center - R @ prev_center

    # 构造变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t

    return transform_matrix

# 定义函数：将前一帧的人体模型应用到当前帧上
def apply_transform_matrix(prev_landmarks_3d, transform_matrix):
    # 将关键点转换为齐次坐标形式
    prev_landmarks_3d = np.hstack([prev_landmarks_3d, np.ones((len(prev_landmarks_3d), 1))])
    # 将变换矩阵应用到关键点上
    curr_landmarks_3d = transform_matrix @ prev_landmarks_3d.T
    # 转换为非齐次坐标形式
    curr_landmarks_3d = curr_landmarks_3d[:3, :].T

    return curr_landmarks_3d

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 读取图像
    success, image = cap.read()
    if not success:
        break

    # 将图像从BGR格式转换为RGB格式，并传入MediaPipe人体姿态估计模型进行处理
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    # 执行手部地标检测
    results_hand = hands.process(image)

    # 如果检测到了手部地标，则绘制它们的关键点
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 提取估计结果中的三维关键点
    if results.pose_landmarks is not None:
        landmarks_3d = []
        for landmark in results.pose_landmarks.landmark:
            landmark_3d = [landmark.x, landmark.y, landmark.z]
            landmarks_3d.append(landmark_3d)

        # 如果前一帧的三维关键点不为空，计算前后两帧之间的变换矩阵，并将前一帧的人体模型应用到当前帧上
        if prev_landmarks_3d is not None:
            transform_matrix = estimate_transform_matrix(prev_landmarks_3d, landmarks_3d)
            landmarks_3d = apply_transform_matrix(prev_landmarks_3d, transform_matrix)

        # 保存当前帧的三维关键点，供下一帧使用
        prev_landmarks_3d = landmarks_3d

        # 在图像上绘制三维关键点
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(vis_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        # 显示图像
        cv2.imshow("MediaPipe Pose Estimation", vis_image)
        if cv2.waitKey(1) == ord('q'):
            break

# 释放资源
pose.close()
cap.release()
cv2.destroyAllWindows()




