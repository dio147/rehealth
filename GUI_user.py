# 导入所需的模块
import sys

import numpy as np
import pandas as pd
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import datetime

from hand_realtime_classify import hand_realtime_score

# 定义一个线程类，用于在后台运行你的函数
class HandThread(QThread):
    # 定义一个信号，用于传递结果
    result_sign = pyqtSignal(dict)

    # 定义初始化方法，接收参数
    def __init__(self, standard_score, model_name, class_num,database,
                cycle_num, action_list):
        super().__init__()
        self.standard_score = standard_score
        self.cycle_num = cycle_num
        self.model_name = model_name
        self.class_num = class_num
        self.database = database
        self.action_list = action_list

    # 定义线程运行方法，调用你的函数，并发出信号
    def run(self):
        score_list, lable, num, qualified_num = hand_realtime_score(standard_score=70, model_name='CNN', class_num=len(action_list),database='Zhangsan',
                        cycle_num=cycle_num, action_list=action_list)
        qualified_rate = qualified_num / num
        # 在run方法中，把几个值封装成一个字典
        result = {'score_list': score_list, 'lable': lable, 'num': num,
                  'qualified_rate': qualified_rate, 
                  'standard_score': self.standard_score}
        # 发出信号，注意名字要和定义时一致
        self.result_sign.emit(result)


# 定义一个窗口类，用于显示gui界面
class HandGUI(QWidget):
    # 定义初始化方法，设置窗口属性和布局
    def __init__(self, standard_score, model_name, class_num,database,
                cycle_num, action_list):
        super().__init__()
        self.standard_score=standard_score
        self.model_name=model_name
        self.cycle_num = cycle_num
        self.class_num = class_num
        self.database = database
        self.action_list = action_list
        self.setWindowTitle('RehabAssist远程智能上肢康复系统')
        self.resize(500, 400)
        # 设置应用的风格
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        # 在初始化方法中，设置窗口的大小为800x600像素
        self.resize(800, 600)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 16))

        # 加载背景图片
        pixmap = QPixmap('bgpic.jpg')
        # 创建一个标签，用来显示背景图片
        label = QLabel(self)
        # 设置标签的大小和窗口一致
        label.resize(self.size())
        # 设置标签的样式表，使用background-image属性设置背景图片为pixmap
        label.setStyleSheet('''
            background-image: url(bgpic.jpg), url(bgpic.jpg);
            background-position: left top, right bottom;
            background-repeat: no-repeat;
            transform: scale(-1, 1); # 只在水平方向上翻转
        ''')

        palette = self.palette()
        # 创建一个QBrush对象，用QPixmap作为参数
        brush = QBrush(pixmap)
        palette.setBrush(QPalette.Active, QPalette.Window, brush)

        self.setPalette(palette)
        # 创建网格布局
        grid = QGridLayout()
        self.setLayout(grid)
        # 显示今日训练计划
        plan_group = QGroupBox('今日训练计划')
        grid.addWidget(plan_group, 0, 0)
        # 创建设置组的网格布局
        plan_grid = QGridLayout()
        plan_group.setLayout(plan_grid)
        # 创建标签和下拉菜单，添加到设置组的网格布局中
        self.labelx = []
        for i in range(self.class_num):
            train_plan = str(str(i+1)+'.'+action_list[i])
            self.labelx.append(QLabel(train_plan))
            plan_grid.addWidget(self.labelx[i],i//2,i%2)
        train_plan = str('重复次数：%d'%self.cycle_num)
        self.labelx.append(QLabel(train_plan))
        plan_grid.addWidget(self.labelx[i+1],i//2+1,0)
        start_train=QPushButton('开始训练')
        start_train.clicked.connect(self.start_thread)
        plan_grid.addWidget(start_train)


        # 创建手指组
        fingers_group = QGroupBox('训练成果记录')
        grid.addWidget(fingers_group, 0, 1)
        # 创建手指组的网格布局
        actions_grid = QGridLayout()
        fingers_group.setLayout(actions_grid)
        self.label5 = QLabel('选择绘图数据：')
        actions_grid.addWidget(self.label5, 1, 0)
        self.combo4 = QComboBox()
        actions_grid.addWidget(self.combo4, 1, 1)
        # 连接下拉菜单的currentTextChanged信号和plot_score槽函数
        # self.combo4.currentTextChanged.connect(self.plot_score)

        # 获取log文件夹中的所有文件名
        file_names = os.listdir('patients/张三,Zhangsan/result')
        # 把文件名添加到下拉菜单中
        for file_name in file_names:
            if file_name[-4:]== '.pkl' and len(file_name.split('_'))>2:
                self.combo4.addItem(file_name)

        # 设置下拉菜单的样式表，使用text-overflow属性设置省略号的位置为右侧
        self.combo4.setStyleSheet('text-overflow: ellipsis;')

        # 点击按钮时创建线程对象，并连接信号和槽函数，启动线程
        # 线程结束时释放线程对象，并恢复按钮可用状态
        # 线程运行时禁用按钮，避免重复点击
        self.button2 = QPushButton('绘制评分报告')
        self.button2.setIcon(QIcon('chart.png'))
        # 设置按钮的图标大小为32x32像素
        self.button2.setIconSize(QSize(32, 32))
        actions_grid.addWidget(self.button2, 0, 1)
        # 点击按钮时调用绘图函数
        self.button2.clicked.connect(self.plot_score)
        # 创建复选框，添加到手指组的网格布局中
        # self.check1 = QCheckBox('拇指')
        # fingers_grid.addWidget(self.check1, 0, 0)
        # self.check2 = QCheckBox('食指')
        # fingers_grid.addWidget(self.check2, 0, 1)
        # self.check3 = QCheckBox('中指')
        # fingers_grid.addWidget(self.check3, 0, 2)
        # self.check4 = QCheckBox('无名指')
        # fingers_grid.addWidget(self.check4, 1, 0)
        # self.check5 = QCheckBox('小指')
        # fingers_grid.addWidget(self.check5, 1, 1)
        # 创建动作组
        # actions_group = QGroupBox('动作')
        # grid.addWidget(actions_group, 1, 0, 1, 2)  # 跨两列
        # # 创建动作组的网格布局
        # actions_grid = QGridLayout()
        # actions_group.setLayout(actions_grid)
        # # 创建按钮，设置图标，添加到动作组的网格布局中，连接槽函数
        # self.button1 = QPushButton('开始识别')
        # self.button1.setIcon(QIcon('icon.png'))

        # self.label6 = QLabel('目标动作数：')
        # actions_grid.addWidget(self.label6, 2, 0)
        # self.slider = QSlider(Qt.Horizontal)  # 创建一个水平方向的滑动条
        # self.slider.setMinimum(1)  # 设置最小值为1
        # self.slider.setMaximum(30)  # 设置最大值为30
        # actions_grid.addWidget(self.slider, 2, 1)
        # self.slider.valueChanged.connect(self.set_target_action)
        

    def start_thread(self):
        # 获取下拉菜单的当前值
        # 创建线程对象，传入参数
        self.thread = HandThread(self.standard_score, self.model_name, self.class_num,self.database,self.cycle_num, self.action_list)
        # 连接信号和槽函数，接收result和motion
        self.thread.result_sign.connect(lambda result: self.get_score(result))
        # 启动线程
        self.thread.start()

    # 定义获取result的槽函数，保存result到文件中，并释放线程对象，恢复按钮可用状态
    def get_score(self, result):
        # 在get_score方法中，获取当前时间，并转换成字符串格式
        now = datetime.datetime.now()
        time_str = now.strftime('%Y_%m_%d_%H_%M_%S')
        # 使用动作名和时间作为文件名
        file_name = self.database + '_' + time_str + '.pkl'

        # 把动作名，模型等参数以及result封装成一个字典
        data = {'motion': self.database, 'class_num': self.class_num, 'result': result}
        # 拼接文件的路径，把log文件夹作为第一个参数
        project_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(project_dir, 'patients/张三,Zhangsan/result', file_name)
        # 使用pickle.dump方法把字典保存到文件中
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        # 把新生成的文件名添加到下拉菜单中
        self.combo4.addItem(file_name)

        # 释放线程对象
        del self.thread
        # 恢复开始按钮可用状态

    def plot_score(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号
        # plt.style.use('ggplot')
        # 获取下拉菜单的当前选择
        file_name = self.combo4.currentText()
        # 如果没有选择任何文件，则返回
        if not file_name:
            return
        # 拼接文件的路径，把log文件夹作为第一个参数
        file_path = os.path.join('patients/张三,Zhangsan/result', file_name)
        # 使用pickle.load方法从文件中加载数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # # 从字典中获取需要的值
        motion = data['motion']

        score_list = data['result']['score_list']
        lable = data['result']['lable']
        num = data['result']['num']
        standard_score = data['result']['standard_score']
        # 创建一个空列表，用来存储重点关注的手指
        fingers = []
        # 把列表转换成字符串，用逗号分隔
        fingers_str = ','.join(fingers)

        # 创建一个新的图形窗口
        plt.figure(figsize=(10, 10))
        average_score = np.mean(score_list)
        # chart_data = '手 部 训 练 与 评 分 报 告'.upper() + '\n动作名：' + motion.lower() + '    模型名：' \
        #              + model_name.lower() + '    类别数：' + str(class_num).lower() + '\n合格分数线' + str(standard_score)\
        #              + '    ' + '整体平均分:' + str(round(average_score, 1))

        # 设置窗口的标题，包含动作名，模型名，类别数等信息
        # plt.suptitle(chart_data, fontsize=20)
        # 创建一个子图，用来绘制折线图
        plt.subplot(2, 2, (1, 2))

        # 定义一个颜色映射函数color_map()，根据分数的高低返回不同的颜色值
        def color_map(score):
            if score < 70:
                return '#FF0000'  # 红色
            elif score < 90:
                return '#FFFF00'  # 黄色
            else:
                return '#00FF00'  # 绿色

        # 绘制折线图，横轴为序号，纵轴为评分，设置颜色为分数对应的颜色值，设置标签
        plt.plot(score_list, c='red')
        # 绘制散点图，横轴为序号，纵轴为评分，设置颜色为分数对应的颜色值，设置标签
        plt.scatter(range(len(lable)), score_list, c=list(map(color_map, score_list)), marker='o')

        # 设置横轴的刻度和标签，使用lable作为标签
        plt.xticks(range(len(lable)), lable)
        # 设置纵轴的范围为0-100
        plt.ylim(0, 100)
        # 设置标题和坐标轴标签，加上完成的动作数，减小字体大小
        plt.title('手部识别与评分折线图（完成' + str(num) + '个动作）', fontsize=15)
        plt.xlabel('动作评分序列')
        plt.ylabel('评分')
        # 显示图例和网格
        # 创建一个空列表，用来存储图例标签
        plt.legend()
        legend_labels = []
        # 遍历lable中的值，根据值获取action_list中的内容，然后拼接成字符串，添加到列表中
        # 获取下拉菜单的当前选项，即动作的中文名
        motion_chinese = motion
        # 遍历字典的键值对，找到对应的英文名
        for motion_english, motion_chinese in motion_dict.items():
            if motion_chinese == motion_chinese:
                break
        # 在DataFrame中查找对应的行，返回一个Series对象
        row = df[df['motion name'] == motion_english].iloc[0]
        # 从Series中获取class num的值，转换为整数
        class_num = int(row['class num'])
        # 从Series中获取动作列表，去掉空值和nan值
        action_list = [x for x in row[2:] if x and not pd.isna(x)]
        for i in lable:
            legend_label = str(i) + ': ' + action_list[i]
            legend_labels.append(legend_label)

        # 显示图例，使用labels参数传递自定义的图例标签
        plt.legend(labels=legend_labels, handlelength=0)

        plt.grid(axis='y')

        # 创建一个子图，用来绘制柱状图
        plt.subplot(2, 2, (3, 4))
        # 计算每个动作的平均分，并存储到一个列表中
        average_scores = []
        for action in set(lable):
            average_score = sum(score_list[i] for i in range(len(lable)) if lable[i] == action) / lable.count(action)
            average_scores.append(average_score)

        # 把lable转换为一个集合，去除重复的元素，并排序
        x = sorted(set(lable))

        # 获取完成率和达标率的值
        qualified_rate = data['result']['qualified_rate']

        # 把完成率和达标率添加到x和average_scores列表中
        x.append('达标率/%')
        average_scores.append(qualified_rate * 100)
        x = list(map(str, x))
        # 绘制竖直柱状图，横轴为动作类别，纵轴为平均分，设置颜色和标签
        plt.bar(x, average_scores, color='green', label='平均分', width=0.25)

        # 设置横轴的刻度和标签，使用x作为标签
        plt.xticks(x, x)

        # 设置标题和坐标轴标签，减小字体大小
        plt.title('手部识别与评分柱状图', fontsize=16)
        plt.xlabel('动作类别')
        plt.ylabel('平均分')
        # 显示图例和网格
        plt.legend()
        plt.grid()

        # 调整子图之间的间距
        plt.tight_layout()
        # 调整标题和子图之间的间距
        plt.subplots_adjust(top=0.8)

        plt.show()


if __name__ == '__main__':
    # 读取csv文件，返回一个DataFrame对象
    df = pd.read_csv('motion_data.csv', encoding='utf-8')
    # 创建应用程序对象和窗口对象，并显示窗口
    app = QApplication(sys.argv)
    f_read = open('zero_nine.pkl', 'rb')
    motion_dict = pickle.load(f_read)
    f_read.close()
    action_list = np.load('patients\张三,Zhangsan\Zhangsan.npy')
    action_list,cycle_num=action_list[:-1].tolist(),int(action_list[-1])
    print(action_list)
    window = HandGUI(standard_score=70, model_name='CNN', class_num=len(action_list),database='Zhangsan',
                cycle_num=cycle_num, action_list=action_list)
    window.show()
    # 进入应用程序的事件循环
    sys.exit(app.exec_())
