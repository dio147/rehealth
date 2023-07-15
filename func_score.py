import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from skimage.transform import PiecewiseAffineTransform
from scipy.linalg import svd

'''
这里我整合了所有用到的函数，包括一些模型类和评分的方法
'''


def generate_attention_list(thumb, index, middle, ring, pinky):
    attention_list = []
    thumb_list = [1, 2, 3, 4]
    index_list = [5, 6, 7, 8]
    middle_list = [9, 10, 11, 12]
    ring_list = [13, 14, 15, 16]
    pinky_list = [17, 18, 19, 20]
    if thumb:
        attention_list.extend(thumb_list)
    if index:
        attention_list.extend(index_list)
    if middle:
        attention_list.extend(middle_list)
    if ring:
        attention_list.extend(ring_list)
    if pinky:
        attention_list.extend(pinky_list)

    return attention_list


def get_weights():
    """
    原始的权重函数，修改前按关节点定义权重，这个函数后面就不用了
    :return: weights，即各关键点权重，21*1的一维数组
    """
    # 定义每个关键点的权重，表示它在评分中的重要性
    weights = [1 for x in range(21)]
    mark_point = {"WRIST": 0, "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4, "INDEX_FINGER_MCP": 5,
                  "INDEX_FINGER_PIP": 6, "INDEX_FINGER_DIP": 7, "INDEX_FINGER_TIP": 8, "MIDDLE_FINGER_MCP": 9,
                  "MIDDLE_FINGER_PIP": 10, "MIDDLE_FINGER_DIP": 11, "MIDDLE_FINGER_TIP": 12, "RING_FINGER_MCP": 13,
                  "RING_FINGER_PIP": 14, "RING_FINGER_DIP": 15, "RING_FINGER_TIP": 16, "PINKY_MCP": 17, "PINKY_PIP":
                      18, "PINKY_DIP": 19, "PINKY_TIP": 20}  # 定义手部关键点

    name = ["INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP"]  # 这里设置期望的权重的关键名

    for point_name in name:
        weights[mark_point[point_name]] = 2
    weights = np.array(weights)
    return weights


def get_tri_weights(landmark_triples, attention_list=None, weights=None):
    """
    获得三元组的权重，即获得用于评分的角度的权重

    :return: weights，24*1的一维数组
    """
    # 定义每个关键点的权重，表示它在评分中的重要性
    if attention_list is None:
        attention_list = [5, 9, 13, 17]
    if weights is None:
        weights = [1.0] * len(landmark_triples)  # 初始权重数组，全为1.0
    for i, triple in enumerate(landmark_triples):
        if triple[1] in attention_list:
            weights[i] = weights[i] * 1.7

    return np.array(weights)


def get_pairs():
    """
    用于获得二元组，用来计算两点间距离，因为好几处都复用我就整合到这里了
    :return: 一个15*2的2维数组
    """
    # 定义一些手部地标对，包括每个手指尖端两两间的距离和五指尖于手腕的距离 15对
    landmark_pairs = []
    # 手指尖端的索引列表
    fingertips = [4, 8, 12, 16, 20]
    # fingerpoint = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # 遍历所有可能的组合
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            # 添加一对手指尖端
            landmark_pairs.append((fingertips[i], fingertips[j]))
        # 添加一对手指尖端和手腕
        landmark_pairs.append((fingertips[i], 0))
    return landmark_pairs


def get_triples():
    """
      用于获得三元组，用来计算三点的角度，因为好几处都复用我就整合到这里了
      :return: 一个24*3的2维列表
    """
    # 定义一些手部地标三元组，包括每个手指关节的角度和两个相邻手指的角度
    landmark_triples = []

    # 手指关节的索引列表 15对+9= 24对
    joints = [[1, 0, 5],  # weigth :1.3
              [2, 1, 5], [0, 1, 2],  # 1.3, 1.0
              [1, 2, 3],  # 1.0
              [2, 3, 4],  # 1.0
              [0, 5, 6], [6, 5, 9], [1, 5, 6],  # 1.3, 1.3, 1.3
              [5, 6, 7],  # 1.0
              [6, 7, 8],  # 1.0
              [10, 9, 13], [5, 9, 10], [0, 9, 10],  # 1.3, 1.3, 1.3
              [9, 10, 11],  # 1.0
              [10, 11, 12],  # 1.0
              [9, 13, 14], [14, 13, 17], [0, 13, 14],  # 1.3, 1.3, 1.3
              [13, 14, 15],  # 1.0
              [14, 15, 16],  # 1.0
              [0, 17, 18], [13, 17, 18],  # 1.3, 1.3
              [17, 18, 19],  # 1.0
              [18, 19, 20]]  # 1.0

    # 遍历每个手指关节
    for joint in joints:
        # 添加一组三个地标，表示一个手指关节的角度
        landmark_triples.append((joint[0], joint[1], joint[2]))
    return landmark_triples


def get_real_time(landmarks, landmark_pairs, landmark_triples):
    """
      用与获得实时的距离和角度信息，用来计算两点间距离，三点的角度，传入的是手部地标
      landmarks：坐标数据
      landmark_pairs:用来计算距离的二元组
      landmark_triples：用来计算角度的三元组
      :return: 每个点的距离和角度
        distances：15*1的一维列表
        angles：24*1的一维列表
      """
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
    return distances, angles


# 定义一个辅助函数，用于计算两点之间的欧氏距离
def distance(p1, p2):
    """
    接受两个点坐标，用来计算两点间距离
    :param p1: 点1坐标
    :param p2: 点2坐标
    :return: 两点距离
    """
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


# 定义一个辅助函数，用于计算三点之间的夹角（单位为弧度）
def angle(p1, p2, p3):
    """
    用于计算三点之间的夹角（单位为弧度）
    :param p1: 点1坐标，作为边的一点
    :param p2: 点2坐标，作为角的顶点
    :param p3: 点3坐标，作为边的一点
    :return: 三点的角度
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos_theta)


def plot_data(ax, aligned_data, standard_data, connections):
    """
    用于可视化手部三维点
    :param aligned_data: 对齐后的数据坐标
    :param standard_data: 标准数据坐标
    :param connections: 手部地标的链接，可之间从mediapipe获得

    """
    # 清空图形
    ax.clear()

    # 绘制aligned_data
    ax.scatter(aligned_data[:, 0], aligned_data[:, 1], aligned_data[:, 2], c='b', label='Aligned Data')

    # 绘制standard_data
    ax.scatter(standard_data[:, 0], standard_data[:, 1], standard_data[:, 2], c='r', label='Standard Data')

    # 连接手部地标
    plot_lines(ax, connections, aligned_data, 'b')
    plot_lines(ax, connections, standard_data, 'r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.draw()  # 更新图形


def plot_lines(ax, connections, landmarks, c):
    """
    用于绘制连线
    :param connections: 手部地标的链接，可之间从mediapipe获得
    :param landmarks: 手部地标的坐标
    :param c: matplotlib的颜色参数
    """
    for connection in connections:
        x0, y0, z0 = landmarks[connection[0]]
        x1, y1, z1 = landmarks[connection[1]]
        ax.plot([x0, x1], [y0, y1], [z0, z1], c=c)


def update_data(ax, aligned_data, standard_data, connections):
    """
    在绘图时只用调用这个就可以
    接受参数，和希望绘制的连线
    :param aligned_data: 对齐后的数据坐标
    :param standard_data: 标准数据坐标
    :param connections: 手部地标的链接，可之间从mediapipe获得

    """
    # 绘制数据
    plot_data(ax, aligned_data, standard_data, connections)

    # 暂停一段时间，等待下一次更新
    plt.pause(0.01)


def animate(i, ax, aligned_data, standard_data, connections):
    """
    用于更新图像的函数，接受参数和希望绘制的连线
    :param i: 动画帧数，不需要使用
    :param ax: 3d界面
    :param aligned_data: 对齐后的数据坐标
    :param standard_data: 标准数据坐标
    :param connections: 手部地标的链接，可之间从mediapipe获得
    """
    # 绘制数据
    plot_data(ax, aligned_data, standard_data, connections)


def kabsch(P, Q):
    """
    kabsch算法实现，用于对齐两组点云
    inputs: P  N x 3 numpy matrix representing the coordinates of the points in P
            Q  N x 3 numpy matrix representing the coordinates of the points in Q

    return: A 4 x 4 matrix where the first 3 rows are the rotation and the last is translation,
            and the last column is the scaling factor
    """
    if (P.size == 0 or Q.size == 0):
        raise ValueError("Empty matrices sent to kabsch")
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    # 均值归一到0
    P_centered = P - centroid_P  # Center both matrices on centroid
    Q_centered = Q - centroid_Q
    H = P_centered.T.dot(Q_centered)  # covariance matrix
    U, S, VT = np.linalg.svd(H)  # SVD
    R = U.dot(VT).T  # calculate optimal rotation

    if np.linalg.det(R) < 0:  # correct rotation matrix for
        VT[2, :] *= -1  # right-hand coordinate system
        R = U.dot(VT).T
    t = centroid_Q - R.dot(centroid_P)  # translation vector

    # 计算缩放因子
    scale = np.sum(S) / np.sum(np.square(P_centered))

    # 注意这是一个右乘矩阵，即使一个4行4列的矩阵
    return np.vstack((R, t, [scale, scale, scale]))


def align_keypoints(real_time_data, standard_data, keypoints):
    """
 用来把实时数据映射到标准数据的坐标系中，
 :param real_time_data: 实时数据，21*3的二维数组
 :param standard_data: 标准数据组，21*3*n的三维数组，用于和实时数据比较
 :param keypoints: 用于对齐的关键点，是一个列表，用于做计算转换矩阵
 :return: 对齐后的数据，是21*3*n的二维数组
 """
    real_time_data = np.array(real_time_data)
    standard_data = np.array(standard_data)

    aligned_data = []

    for std_data in standard_data:
        # 选择关键点作为输入
        src_points = real_time_data[keypoints]
        dst_points = std_data[keypoints]

        # 使用kabsch算法求解变换矩阵
        transform_matrix = kabsch(src_points, dst_points)

        # 使用变换矩阵对齐实时数据
        aligned_real_time_data = np.dot(real_time_data, transform_matrix[:3, :3].T) + transform_matrix[3, :3]
        aligned_real_time_data *= transform_matrix[4, :3]
        aligned_data.append(aligned_real_time_data)

    return np.array(aligned_data)


def evaluate_similarity(real_time_data, standard_data, weights, keypoints,
                        realtime_angl, standard_angl):
    """
    用于评估相似性，将实时数据映射到目标坐标系，对比相似度，得到相似度最高的标准数据，作为最终数据，进行评分
    返回的是评分，实时数据坐标和标准数据坐标
    :param real_time_data: 实时坐标，21*3二维数组
    :param standard_data: 标准坐标，21*3*n，三维数组
    :param weights: 权重列表，每个关键点的权重（待修改为角度的权重）
    :param keypoints: 用于做矩阵映射的关键点索引列表
    :param realtime_angl: 实时的角度信息
    :param standard_angl: 标准的角度信息

    :return: score：最终总评分
     min_distance_index：最小距离的索引，即相似度最高的标准数据在当前类别数据的索引，用于提取信息
      low_score_points: 低分点列表，用于对这些点生成建议
     real_time_data：映射到标准数据坐标系后的实时数据，即于标准数据对齐后的实时坐标
      standard_data:标准数据坐标，和上面实时数据一块拿来用于3d作图
    """
    distance_threshold = 0.175  # 这里是0.1度弧度，可以改
    alpha = 0.2  # 不合格点所占评分的比重，也可改
    """
    :param distance_threshold: 得分阈值，用于判断差异大小的阈值，0.9，低于该值的设为低分
    :param alpha: 用于评分的参数，低分点的干预程度
    """
    global point_distances, angle_distances, i
    aligned_data = align_keypoints(real_time_data, standard_data, keypoints)

    min_distance = float('inf')  # 初始化为正无穷
    # 用于获得对齐的数据
    min_distance_index = -1
    for i, standard_angle in enumerate(standard_angl):
        point_distance = abs(realtime_angl - standard_angle)
        distance = np.sum(point_distance * weights)  # 计算关键点距离/计算总距离
        if distance < min_distance:
            min_distance_index = i
            min_distance = distance
    diff_list = abs(realtime_angl - standard_angl[min_distance_index])  # 计算差异向量(角度）

    angle_distance = np.sum(diff_list * weights) / np.sum(weights)  # 计算关键角差异均值（加权重）

    min_distances = angle_distance  # 获得最小差异的角度均值(加权重）

    angle_distances = diff_list * weights  # 每个点的差异列表 （24*1）（权重）

    # min_distance, min_distance_index = calculate_similarity(aligned_data, standard_data, weights)   #用于获取距离更近的点

    # key_distance = min_distance

    real_time_data = aligned_data[min_distance_index]
    standard_data = standard_data[min_distance_index]  # 获得可视化所需的数据

    # point_distances = point_distances  # 消除关键点没有对准带来的误差- key_distance
    # 权重在1-2之间，小数
    low_score_points = np.where(angle_distances > distance_threshold)[0]  # 获取赋权重差评分异大于阈值的点

    if len(low_score_points) > 0:
        low_score_points = low_score_points.astype(int)
        weighted_below_threshold_ratio = np.sum(weights[low_score_points]) / np.sum(weights)
    else:
        # 处理 low_score_points 为空的情况，设置默认值
        weighted_below_threshold_ratio = 0.0  # 设置默认值为0.0

    average_weighted_diff = min_distances

    # 计算总分
    score = 100 * (1 - (1 - alpha) * average_weighted_diff - alpha * weighted_below_threshold_ratio)

    return score, min_distance_index, low_score_points, real_time_data, standard_data


def generate_feedback(realtime_angl, standard_angl, min_distance_index, weights, low_score_points, landmark_triples):
    """
    用于生成建议，还需要完善修改，勉强能用
    :param realtime_angl: 实时角度信息
    :param standard_angl: 标准坐标的角度信息
    :param min_distance_index: 最小距离索引，即相似度最高的标准数据索引
    :param weights: 权重
    :param low_score_points:低分点，用于对这些点生成建议
    :return: 返回建议的语句
    """
    feedback = []
    # diff = real_time_data - standard_data[min_distance_index]
    diff_list = realtime_angl - standard_angl[min_distance_index]  # 计算差异向量(角度）



    # 手指的名称列表，与三元组对应
    finger_names = ["手腕",
                    "大拇指", "大拇指", "大拇指", "大拇指",
                    "食指", "食指", "食指", "食指", "食指",
                    "中指", "中指", "中指", "中指", "中指",
                    "无名指", "无名指", "无名指", "无名指", "无名指",
                    "小拇指", "小拇指", "小拇指", "小拇指"]

    # 创建一个空字典，用于存储低分点对应的手指名称和调整幅度的平均值
    low_score_fingers = {}

    for i in low_score_points:
        coord_diff = diff_list[i]
        direction = "增加" if coord_diff > 0 else "减少"
        magnitude = abs(coord_diff) * weights[i]
        # magnitude以弧度为单位，差乘权重，大约在0-2pi之间（取决于最大权重）

        # 获取低分点对应的三元组的中间那一项的索引值，即手指关节的索引值
        finger_joint_index = landmark_triples[i][1]

        # 根据手指关节的索引值，获取手指的名称，并添加到低分手指字典中，如果该手指已经在字典中，则累加调整幅度的值，否则初始化为调整幅度的值和方向
        finger_name = finger_names[finger_joint_index]
        if finger_name in low_score_fingers:
            low_score_fingers[finger_name][0] += magnitude
            low_score_fingers[finger_name][1] += 1
        else:
            low_score_fingers[finger_name] = [magnitude, 1, direction]

    # 遍历低分手指字典，计算每个手指的调整幅度的平均值，并根据平均值确定幅度标签
    for finger_name, values in low_score_fingers.items():
        average_magnitude = values[0] / values[1]
        direction = values[2]
        if average_magnitude < math.pi /18:
            magnitude_label = "小幅度"
        elif average_magnitude < math.pi /6:
            magnitude_label = "中等"
        else:
            magnitude_label = "大幅度"

        # 将手指名称和调整幅度的信息添加到反馈列表中
        feedback.append(f"{finger_name} {magnitude_label}")

    # 返回总的建议语句
    return feedback





def calculate_vector_difference(realtime_data, standard_data):
    """
    计算两个向量的差异

    参数:
    realtime_data -- 实时数据向量
    standard_data -- 标准数据向量

    返回值:
    差异向量
    """
    diff_vector = [abs((realtime - standard) / math.pi) - 1 for realtime, standard in zip(realtime_data, standard_data)]
    diff_vector = [abs(value) for value in diff_vector]

    return diff_vector


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(get_triples())
    print(len(get_triples()))
    # # 测试用
    # standard_data = [[[3.01866710e-01, 8.28894496e-01, 6.69649637e-07],
    #                   [3.77057999e-01, 7.91031897e-01, -3.51564176e-02],
    #                   [4.41321075e-01, 7.19688952e-01, -5.67869581e-02],
    #                   [4.96528208e-01, 6.63501024e-01, -7.64441714e-02],
    #                   [5.48490345e-01, 6.33633614e-01, -9.64385197e-02],
    #                   [3.93934667e-01, 5.73823988e-01, -4.16929647e-02],
    #                   [4.37346518e-01, 4.79541898e-01, -7.10376799e-02],
    #                   [4.61667418e-01, 4.16048110e-01, -9.33514908e-02],
    #                   [4.79386181e-01, 3.59444797e-01, -1.09791696e-01],
    #                   [3.43127877e-01, 5.49942732e-01, -4.67148274e-02],
    #                   [3.57075483e-01, 4.20411587e-01, -7.29946494e-02],
    #                   [3.66705388e-01, 3.39576244e-01, -9.23661813e-02],
    #                   [3.71138752e-01, 2.69583285e-01, -1.06176451e-01],
    #                   [2.93940604e-01, 5.57734907e-01, -5.57998046e-02],
    #                   [2.87263811e-01, 4.36750919e-01, -8.34967345e-02],
    #                   [2.86508799e-01, 3.57195914e-01, -1.03134103e-01],
    #                   [2.86576211e-01, 2.84940183e-01, -1.16065674e-01],
    #                   [2.47988850e-01, 5.92344761e-01, -6.78040385e-02],
    #                   [2.13940829e-01, 5.07199168e-01, -9.46967229e-02],
    #                   [1.92679718e-01, 4.51163113e-01, -1.07310034e-01],
    #                   [1.75713435e-01, 3.94218385e-01, -1.14778280e-01]],
    #                  [[7.06321716e-01, 8.16470861e-01, 5.06378058e-07],
    #                   [6.16540432e-01, 7.87114620e-01, -2.51387078e-02],
    #                   [5.48127651e-01, 6.97230637e-01, -3.11285667e-02],
    #                   [5.04636765e-01, 6.13532007e-01, -3.66037339e-02],
    #                   [4.58472788e-01, 5.72359085e-01, -4.21583764e-02],
    #                   [5.85599899e-01, 5.50921738e-01, 6.86548185e-04],
    #                   [5.50724983e-01, 4.45607752e-01, -1.36867957e-02],
    #                   [5.34188986e-01, 3.79494548e-01, -2.98216511e-02],
    #                   [5.24575591e-01, 3.19167137e-01, -4.34319004e-02],
    #                   [6.40215993e-01, 5.19154370e-01, -4.11894871e-03],
    #                   [6.29367292e-01, 3.94095272e-01, -1.31005421e-02],
    #                   [6.23407364e-01, 3.17277014e-01, -2.75436658e-02],
    #                   [6.18710995e-01, 2.53099412e-01, -4.03080173e-02],
    #                   [6.90847456e-01, 5.16795039e-01, -1.48799932e-02],
    #                   [6.88039422e-01, 3.93514633e-01, -3.05321384e-02],
    #                   [6.81625605e-01, 3.22413683e-01, -4.39237319e-02],
    #                   [6.74291611e-01, 2.60293216e-01, -5.50857112e-02],
    #                   [7.38647103e-01, 5.37588000e-01, -2.86500026e-02],
    #                   [7.56118119e-01, 4.48039949e-01, -4.52039838e-02],
    #                   [7.66029775e-01, 3.87612015e-01, -5.24619669e-02],
    #                   [7.73187518e-01, 3.29795659e-01, -5.83132692e-02]]
    #                  ]
    #
    # real_time_data = [[3.06212723e-01, 7.66658902e-01, 5.50058360e-07],
    #                   [3.77670467e-01, 8.17454219e-01, -3.74324583e-02],
    #                   [4.65854675e-01, 8.37444365e-01, -5.55928238e-02],
    #                   [5.42995751e-01, 8.48547459e-01, -7.16903806e-02],
    #                   [6.04859591e-01, 8.64479542e-01, -8.77629966e-02],
    #                   [5.31676650e-01, 6.91025257e-01, -3.65596339e-02],
    #                   [6.24229372e-01, 6.62229300e-01, -6.54059425e-02],
    #                   [6.81784153e-01, 6.39792383e-01, -8.76609907e-02],
    #                   [7.29124129e-01, 6.17402554e-01, -1.03630356e-01],
    #                   [5.11841178e-01, 6.19928598e-01, -3.97887528e-02],
    #                   [6.04587972e-01, 5.48786938e-01, -6.66758940e-02],
    #                   [6.62742078e-01, 5.04180551e-01, -8.83615762e-02],
    #                   [7.08114564e-01, 4.64432865e-01, -1.02698885e-01],
    #                   [4.74991798e-01, 5.70486486e-01, -4.74530943e-02],
    #                   [5.53359330e-01, 4.90317971e-01, -7.55695403e-02],
    #                   [6.03320360e-01, 4.41133708e-01, -9.45037976e-02],
    #                   [6.45674229e-01, 4.01094794e-01, -1.06173024e-01],
    #                   [4.24649626e-01, 5.40069759e-01, -5.84298559e-02],
    #                   [4.68285859e-01, 4.58168745e-01, -8.76287371e-02],
    #                   [4.96070385e-01, 4.04198378e-01, -1.00348294e-01],
    #                   [5.22821367e-01, 3.56767356e-01, -1.06205106e-01]]
    #
    # real_time_data = np.array(real_time_data)
    # standard_data = np.array(standard_data)
    #
    # weights = np.ones(21)
    #
    # keypoints = [0, 5, 17]
    #
    # score, min_distance_index, low_score_points = evaluate_similarity(real_time_data, standard_data, weights, keypoints)
    # feedback = generate_feedback(real_time_data, standard_data, min_distance_index, weights, low_score_points)
    #
    # print(f"评分: {score:.2f}")
    # print("改进建议:")
    # for suggestion in feedback:
    #     print(suggestion)
