'''all'''
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from classify.CNN_hand import CNN
# from classify.KNN_hand import KNN
# from classify.SVM_hand import SVM
# from classify.MLP_hand import MLP
from func_score import get_real_time, get_pairs, get_triples, evaluate_similarity, generate_feedback, \
    update_data, get_tri_weights, get_weights, generate_attention_list
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from cfg_parameter import FixedSizeQueue, CfgData
import time
from joblib import load
import torch
import pickle
# 导入Tkinter模块
import tkinter as tk

"""
这个是评分的主要函数，把分类和评分功能整合起来的代码，为了方便调试我直接设置了选择的模型，
要做ui的话记得改参数
"""

f_read = open('public/zero_nine.pkl', 'rb')
motion_dict = pickle.load(f_read)
f_read.close()
# print(motion_dict)
# 定义一个函数，用于将中文字符转换为图像
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    # 判断是否为opencv格式图片
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式，修改为黑体
    fontStyle = ImageFont.truetype("/usr/share/fonts/truetype/simhei.ttf", 32)
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回opencv格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def load_model(model_name, model_path, class_num):
    if model_name == "CNN":
        model = CNN(class_num=class_num)
    # elif model_name=="KNN":
    #     model = KNN(class_num=class_num)
    # elif model_name=="SVM":
    #     model = SVM(class_num=class_num)
    # elif model_name=="MLP":
    #     model = MLP(class_num=class_num)
    else:
        print("Invalid choice. Please try again.")
        model = None
    model.load_state_dict(torch.load(model_path))  # 加载保存的参数
    model.eval()
    return model

def hand_realtime_score(standard_score, model_name, cycle_num, action_list,classify_model,score_dataset):
    warnings.filterwarnings("ignore")  # 用来忽略依赖错误
    # 定义一个栈，用来存储动作的顺序
    class_num=len(action_list)
    action_num = cycle_num * class_num
    action_count = 0
    qualified_num = 0
    change_flag = False
    end_flag = False
    device = torch.device('cpu')
    full_process_fps=[]
    Classify_fps=[]
    Score_fps=[]
    Mediapipe_fps=[]
    score_path = score_dataset
    model_path = classify_model
    model=load_model(model_name, model_path, class_num)
    # 将模型移动到GPU上（如果有）
    model.to(device)

    df=pd.read_csv(score_path)
    y = df.iloc[:, -1].to_numpy()  # 最后一列是标签
    y1=list(set(y))
    y1.sort()
    for i in range(class_num):
        y[y == y1[i]] = i
    #将排序好的y存回df
    df.iloc[:, -1]=y
    print(model_name + '模型加载完毕')

    # 获得手部关节对与三元组
    landmark_pairs = get_pairs()
    landmark_triples = get_triples()

    weights = get_tri_weights(landmark_triples)  # 获得权重
    bar_height = 20
    bar_color = (255, 128, 0)

    next_action = action_list[0]
    attention_list = generate_attention_list(True, True, True, True, True)
    # print(attention_list)
    if len(attention_list) != 0:
        weights = get_tri_weights(landmark_triples, attention_list=attention_list, weights=weights)  # 获得权重

    # 定义用于对齐的关键点的索引这里取手腕和四指根部
    keypoints = [0, 5, 9, 13, 17]
    # 创建一个Tk对象，作为主窗口
    root = tk.Tk()

    # 创建一个StringVar对象，用于存储评分的变量
    score_var = tk.StringVar()

    # 创建一个Label对象，用于显示评分，并绑定到score_var上
    score_label = tk.Label(root, textvariable=score_var, font=("Arial", 20))
    score_label.pack()

    # 创建一个Text对象，用于显示建议
    feedback_text = tk.Text(root, height=10, width=40, font=("Arial", 16))
    feedback_text.pack()

    # df = pd.read_csv('score_sample\hand_rec.csv')
      # 读入评分数据

    score_list = []  # 用来存储评分
    action_stack_all = []
    # 初始化mediapipe手部地标模型
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, model_complexity=1,
                           min_tracking_confidence=0.5)

    pre_pred = -1  # 用于存储上一次预测结果
    score_queue = FixedSizeQueue(48)  # 设置定长队列,同一动作最大存储评分数量
    color = (255, 255, 255)
    # 初始化参数
    score = 0
    # 从摄像头读取图像
    cap = cv2.VideoCapture(0)
    # 循环开始前获取当前时间
    last_time = time.time()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # 获取画面的宽度和高度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 将图像转换为RGB格式，并翻转水平方向
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # 设置图像为不可写，以提高处理速度
        image.flags.writeable = False
        # 使用手部地标模型处理图像，并获取结果
        mp_start=time.time()
        results = hands.process(image)
        mp_end=time.time()
        mp_fps=1/(mp_end-mp_start)
        Mediapipe_fps.append(mp_fps)


        # 将图像转换回BGR格式，并设置为可写
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 定义左右手的分类结果变量
        left_label = None
        right_label = None

        # 如果有检测到手部地标，绘制出来，并预测手势类别
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 绘制手部地标和连接线
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 获取手部地标的三维坐标（归一化到[0,1]范围）
                landmarks = hand_landmarks.landmark
                connections = mp_hands.HAND_CONNECTIONS
                distances, angles = get_real_time(landmarks, landmark_pairs, landmark_triples)

                # 将距离和角度合并为一个特征向量，并添加到DataFrame中
                hand_data = np.array(distances + angles)

                # 获取手部的左右信息（0表示左手，1表示右手）
                handness = handedness.classification[0].label
                # 根据用户的输入，调用相应的模型，并打印预测结果
                if model_name != 'CNN':

                    # 将数据reshape为(1,-1)的形状
                    hand_data = hand_data.reshape(1, -1)

                    pred = model.predict(hand_data)[0]
                   

                    # 根据左右信息，更新分类结果
                    if handness == 'Left':
                        left_label = action_list[pred]
                    else:
                        right_label = action_list[pred]

                elif model_name == "CNN":
                    # 将数据reshape为(1,-1)的形状
                    # hand_data = hand_data.reshape(1, -1)
                    classify_start=time.time()
                    pred = model.predict(hand_data)
                    classify_end=time.time()
                    if classify_end-classify_start==0:
                        classify_fps=1000
                    else:
                        classify_fps=1/(classify_end-classify_start)
                    Classify_fps.append(classify_fps)
                    # print(f"cnn_pred: {cnn_pred}")
                    # 根据左右信息，更新分类结果
                    if handness == 'Left':
                        left_label = action_list[pred]
                    else:
                        right_label = action_list[pred]

                '''
                用于评分的部分
                '''

                # 获取手部地标的坐标列表，每个坐标是一个(x,y,z)元组
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                # 将坐标列表转换为numpy数组，并添加一列类别标签
                landmarks = np.array(landmarks)
                # 定义一个空的列表，用于存储实时识别的坐标
                real_time_data = landmarks
                # real_time_data是用于评分的坐标信息

                # 获得手部地标的角度等信息
                landmarks_ang = hand_landmarks.landmark
                _, angles = get_real_time(landmarks_ang, landmark_pairs, landmark_triples)  # 这里舍弃掉距离信息
                realtime_angl = angles

                df1 = df[df.iloc[:, -1] == pred]

                # 提取出关键点的坐标，并转换为numpy数组
                all_data = df1.to_numpy()
                all_data = all_data[:, :-1]  # 去掉最后一个标签值

                all_angles_data = all_data[:, 63:]  # 得到全部角度信息，用于后续评分
                all_standard_data = all_data[:, :63]

                standard_data = all_standard_data.reshape(-1, 3)  # 将每三个值划分为一组
                standard_data = np.array(standard_data)
                if len(standard_data)==0:
                    continue
                n = len(standard_data) // 21
                standard_data = np.stack(np.split(standard_data, n), axis=0)
                # print(standard_data)

                score_start=time.time()
                score, min_distance_index, low_score_points, rd, sd = evaluate_similarity(real_time_data,
                                                                                          standard_data,
                                                                                          weights, keypoints,
                                                                                          realtime_angl,
                                                                                          all_angles_data)
                score_end=time.time()
                score_fps=1/(score_end-score_start)
                Score_fps.append(score_fps)

                if score < standard_score:
                    # 调用generate_feedback函数，返回每个手指建议信息的列表
                    feedback_list = generate_feedback(realtime_angl, all_angles_data, min_distance_index, weights,
                                                      low_score_points, landmark_triples)

                    # 更新评分变量的值
                    score_var.set(f"评分: {score:.2f}")

                    # 清空Text对象的内容
                    feedback_text.delete(1.0, tk.END)
                    # 插入新的建议内容
                    feedback_text.insert(1.0, "改进建议:\n")

                    # 定义一个字典，用于存储每个手指对应的行号
                    finger_lines = {"手腕": 2,
                                    "大拇指": 3,
                                    "食指": 4,
                                    "中指": 5,
                                    "无名指": 6,
                                    "小拇指": 7}

                    # 先插入所有手指的空行
                    for finger_name in finger_lines:
                        line_number = finger_lines[finger_name]
                        feedback_text.insert(f"{line_number}.0", "\n")

                    # 遍历每个手指建议信息的列表
                    for finger_feedback in feedback_list:
                        # 分割字符串，获取手指名称和幅度标签
                        finger_name, magnitude_label = finger_feedback.split()
                        # 根据幅度标签，确定字体颜色
                        if magnitude_label == "小幅度":
                            color = "blue"
                        elif magnitude_label == "中等":
                            color = "green"
                        else:
                            color = "red"
                        # 根据手指名称，获取对应的行号
                        line_number = finger_lines[finger_name]
                        # 替换该行，并使用tag参数设置字体颜色
                        feedback_text.delete(f"{line_number}.0", f"{line_number}.end")
                        feedback_text.insert(f"{line_number}.0", finger_feedback + "\n")
                        # 配置tag参数，设置前景色为color
                        feedback_text.tag_config(color, foreground=color)

                else:
                    # 更新评分变量的值
                    score_var.set(f"评分: {score:.2f}")
                    # 清空Text对象的内容
                    feedback_text.delete(1.0, tk.END)
                    # 插入新的反馈内容
                    feedback_text.insert(1.0, "恭喜你，你的手势很标准！\n")
                    feedback_text.insert(2.0, "继续保持，你可以做得更好！")
                # 调用update方法，让窗口刷新
                root.update()

                if pre_pred == -1 and pred == 0:
                    pre_pred = pred
                    action_count += 1  # 完成一个动作，计数器加一
                    # 将当前的预测结果压入栈中
                    action_stack_all.append(pred)
                    if score != 0:
                        score_queue.push(score)
                    continue

                # 计算总分的部分
                elif pred == pre_pred:  # 若类别不变
                    if change_flag==True:
                        change_flag=False
                        action_count += 1  # 完成一个动作，计数器加一
                        # 将当前的预测结果压入栈中
                        action_stack_all.append(pred)
                        if score_queue.size() != 0:
                            score_list.append(score_queue.average())
                            if score_queue.average() > standard_score:
                                qualified_num = qualified_num + 1
                            score_queue.clear()
                    if score != 0:
                        score_queue.push(score)

                elif pred != pre_pred and action_list[pred]==next_action:  # 若类别改变
                    if change_flag==False:
                        change_flag=True
                    else:
                        change_flag=False
                    
                    
                pre_pred = pred  # 存储上一次的预测结果

                # print(class_num)
                # 获取下一个应该做的动作类别，如果栈为空或者已经到达class_num，则为0，否则为栈顶元素加一
            next_action_no = (action_count)%len(action_list)
            next_action = action_list[next_action_no]
            # print(pred)
        # 在图像上显示左右手的分类结果
        image = cv2ImgAddText(image, '左手: ' + str(left_label), 10, 30, textSize=40, textColor=(0, 128, 255))
        image = cv2ImgAddText(image, '右手: ' + str(right_label), 10, 60, textSize=40, textColor=(0, 128, 255))

        # 计算进度条的宽度
        bar_width = int(width * qualified_num / action_num)

        # 绘制进度条的背景，这里用灰色
        cv2.rectangle(image, (0, height - bar_height), (width, height), (128, 128, 128), -1)

        # 绘制进度条的前景，这里用bar_color
        cv2.rectangle(image, (0, height - bar_height), (bar_width, height), bar_color, -1)

        # 在进度条上显示百分比，这里用白色
        percent = str(int(qualified_num / action_num * 100)) + "%"
        cv2.putText(image, percent, (width // 2, height - bar_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        # 根据评分的高低设置不同的颜色
        if score > 90:
            color = (0, 255, 0)  # 绿色
        elif score > 80:
            color = (255, 255, 0)  # 黄色
        else:
            color = (255, 0, 0)  # 红色

        # 在图像上显示当前评分和颜色
        image = cv2ImgAddText(image, '当前评分: ' + str(round(score, 1)), 10, 90, textSize=40, textColor=color)

        # 在图像上显示下一个应该做的动作和完成的动作数
        image = cv2ImgAddText(image, '下一个动作: ' + str(next_action), 10, 120, textSize=40, textColor=color)
        image = cv2ImgAddText(image, '动作计数: ' + str(action_count), 10, 150, textSize=40, textColor=color)
        # 在每一轮结束后获取当前时间，并计算运行时间

        this_time = time.time()
        fps=1/(this_time-last_time)
        full_process_fps.append(fps)
        last_time = this_time

        # 打印每一轮的运行时间
        # print(f"该帧评分时间：{iteration_time} 秒")

        # update_data(ax, rd, sd, connections) # 显示图像

        cv2.imshow('MediaPipe Hands', image)
        cv2.setWindowProperty('MediaPipe Hands', cv2.WND_PROP_TOPMOST, 1)
        if action_count == action_num:
            if end_flag==True:
                score_list.append(score)
                print(score_list)
                print(action_stack_all)
                print('full_process_FPS=',full_process_fps)
                print('Mediapipe_FPS=',Mediapipe_fps)
                print('Classify_FPS=',classify_fps)
                print('Score_FPS=',score_fps)
                break
            else:
                end_flag=True
        if cv2.waitKey(5) & 0xFF == 27:
            # print(score_list)
            # print(action_stack_all)
            break
    plt.close("all")
    # 关闭摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()

    # 返回评分列表和动作列表，用来做折线图
    return score_list, action_stack_all, qualified_num, standard_score


if __name__ == '__main__':
    action_list = np.load('test.npy')
    print(action_list)
    hand_realtime_score(standard_score=70, model_name='CNN', class_num=len(action_list),database='test',
                        cycle_num=3, action_list=action_list)
