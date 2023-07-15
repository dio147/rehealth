import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

"""
这个是用来录全身数据的，录取的是用来做分类的数据，还是按w保存，按esc退出
全身评分用的数据录取还没写，想直接加到这个这段代码里
"""
# 获取手部的类别标签
label = 0

# 初始化MediaPipe人体姿态估计
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化前一帧的三维关键点
prev_landmarks_3d = None


# 定义函数：估计前后两帧之间的变换矩阵
def estimate_transform_matrix(prev_landmarks_3d, curr_landmarks_3d):
    """
    用来做数据平滑，前一帧的坐标映射到下一帧，减轻抖动用的
    :param prev_landmarks_3d: 上一帧的人体坐标
    :param curr_landmarks_3d: 当前帧的人体坐标
    :return: 转换矩阵，用于映射，平移
    """
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

# 创建一个空的DataFrame，用于存储手部地标的信息
df = pd.DataFrame()

# 创建一个标志变量，用于控制是否保存数据
save_data = False


# 定义一个辅助函数，用于计算两点之间的欧氏距离
def distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


# 定义一个辅助函数，用于计算三点之间的夹角（单位为弧度）
def angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos_theta)


# 定义一些手部地标对，包括每个手指尖端两两间的距离和五指尖于手腕的距离 15对
landmark_pairs = []

body_parts = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# 遍历所有可能的组合
for i in range(len(body_parts)):
    for j in range(i + 1, len(body_parts)):
        # 添加一对手指尖端
        landmark_pairs.append((body_parts[i], body_parts[j]))
    # 添加一对手指尖端和手腕
    landmark_pairs.append((body_parts[i], 0))

# 定义一些手部地标三元组，包括每个手指关节的角度和两个相邻手指的角度
landmark_triples = []

# 全身动作地标索引列表 每个点只有在存在三个相邻的点时才能确当该点的空间特征
joints = [
    [11, 13, 15], [13, 15, 17], [15, 17, 19], [23, 11, 13], [11, 13, 23], [13, 15, 12],  # Left arm
    [12, 14, 16], [14, 16, 18], [14, 16, 20], [24, 12, 14], [12, 14, 24], [14, 16, 11],  # Right arm
    [12, 24, 26], [11, 23, 25], [26, 24, 23], [25, 23, 24],  # Hip
    [12, 11, 24], [12, 23, 24], [11, 12, 23], [11, 24, 23],  # Waist
    [11, 12, 24], [12, 11, 23], [11, 23, 24], [12, 24, 23],  # Chest
    [24, 26, 28], [26, 28, 32], [26, 28, 30], [26, 28, 24], [26, 28, 23],  # Left leg
    [23, 25, 27], [25, 27, 31], [25, 27, 29], [25, 27, 23], [25, 27, 24],  # Right leg
]

# 遍历每个手指关节
for joint in joints:
    # 添加一组三个地标，表示一个手指关节的角度
    landmark_triples.append((joint[0], joint[1], joint[2]))

while cap.isOpened():
    # 读取图像
    success, image = cap.read()
    if not success:
        break

    # 将图像从BGR格式转换为RGB格式，并传入MediaPipe人体姿态估计模型进行处理
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # 提取估计结果中的三维关键点
    if results.pose_world_landmarks is not None:
        landmarks_3d = []
        for landmark in results.pose_world_landmarks.landmark:
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


        # 如果需要保存数据
        if save_data:
            # 获取全身地标的三维坐标（归一化到[0,1]范围）
            landmarks = results.pose_landmarks.landmark

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
            df = df.append(pd.Series(row), ignore_index=True)

            # 添加类别标签到DataFrame中
            df['label'] = label
            print('记录成功')

            save_data = False

            # # 添加列名到DataFrame中（你需要自己定义）
            # columns = ['d' + str(i) for i in range(15)] + ['a' + str(i) for i in range(18)] + ['handedness', 'label']
            # df.columns = columns

            # 保存DataFrame到文件中，使用mode='a'表示追加模式，header=False表示不写入列名
            df.to_csv('pose_data/body_data.csv', index=False, mode='a', header=False)

            # 清空df中的数据，保留列名
            df = df.drop(df.index)

        # 等待按键输入
        key = cv2.waitKey(1) & 0xFF
        # 如果按下w键，开始保存数据
        if key == ord('w'):
            save_data = True

        # # 如果按下q键，停止保存数据，并将DataFrame保存到一个文件中（你需要自己指定文件名）
        # if key == ord('q'):
        #     save_data = False
        #
        #     # # 添加列名到DataFrame中（你需要自己定义）
        #     # columns = ['d' + str(i) for i in range(15)] + ['a' + str(i) for i in range(18)] + ['handedness', 'label']
        #     # df.columns = columns
        #
        #     # 保存DataFrame到文件中，使用mode='a'表示追加模式，header=False表示不写入列名
        #     df.to_csv('pose_data/body_data.csv', index=False, mode='a', header=False)
        #
        #     # 清空df中的数据，保留列名
        #     df = df.drop(df.index)
        # 显示图像
    cv2.imshow("MediaPipe Pose Estimation", vis_image)
    # 如果按下esc键，退出循环
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 释放资源
pose.close()
cap.release()
cv2.destroyAllWindows()
