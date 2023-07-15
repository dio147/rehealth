import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
"""
想试着做数据增强，左右手各拍一张照片就可以获得多组数据，
但是改了底层评分逻辑之后不需要那么多数据了，这个也没啥用了
不过可以试着改下下面的旋转次数，这样录数据也方便一些（或许)效果待检验
"""
# 获取手部的类别标签
label = 1  # 0：五指展开 1：五指并拢

# 创建一个Hand对象，用于检测和识别手部
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                                    model_complexity=1,
                                    min_tracking_confidence=0.5)

# 创建一个Drawing对象，用于绘制手部地标
mp_drawing = mp.solutions.drawing_utils
csv_path = 'score_sample/hand_landmarks_test.csv'
# 读取图片
image_path = 'test_pic/righthand1.jpg'
image = cv2.imread(image_path)

# 创建一个空的DataFrame，用于存储手部地标的信息
df = pd.DataFrame()

# 创建一个标志变量，用于控制是否保存数据
save_data = False

n = 0  # 计数器


# 图片旋转函数
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 0.8)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image


# 图片翻转函数
def flip_image(image, flip_code):
    flipped_image = cv2.flip(image, flip_code)
    return flipped_image


# 仿射变换函数
def affine_transform(image, scale, shear):
    if len(image.shape) == 2:  # 如果是灰度图像，将其转换为彩色图像
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rows, cols, _ = image.shape
    M = np.float32([[scale, shear, 0], [shear, scale, 0]])
    transformed_image = cv2.warpAffine(image, M, (cols, rows))
    return transformed_image


# 执行数据增强操作
for flip_code in [0, 1]:  # 上下翻转、左右翻转
    flipped_image = flip_image(image, flip_code)
    for angle in range(0, 360, 10):  # 旋转角度：0度到360度，每次旋转10度
        rotated_image = rotate_image(flipped_image, angle)
        # for shear in np.linspace(-0.3, 0.3, num=24):  # 剪切变换：-0.3到0.3，共36种
        #     transformed_image = affine_transform(rotated_image, 1.0, shear)

        # 将图像转换为RGB格式
        image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

        # 将图像传入Hand对象，得到检测和识别的结果
        results = mp_hands.process(image_rgb)

        # 如果有检测到手部
        if results.multi_hand_landmarks:
            # 遍历每一只手
            for hand_landmarks in results.multi_hand_landmarks:
                # 在图像上绘制三维关键点
                # mp_drawing.draw_landmarks(transformed_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 获取手部地标的坐标列表，每个坐标是一个(x,y,z)元组
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # 将坐标列表转换为numpy数组，并添加一列类别标签
                landmarks = np.array(landmarks)
                landmarks = np.append(landmarks, label)

                # 将numpy数组转换为Series，并添加到DataFrame中
                series = pd.Series(landmarks)
                df = df.append(series, ignore_index=True)

                # 将DataFrame保存到一个文件中 mode='a'设置为追加模式
                df.to_csv(csv_path, index=False, mode='a', header=False)

                # 计数器加一，并打印提示信息
                n = n + 1
                print('保存成功，当前第' + str(n) + '条')

                # 使用df.drop方法来删除已经保存的数据，保持df为空
                df = df.drop(df.index)

# 释放窗口
cv2.destroyAllWindows()
