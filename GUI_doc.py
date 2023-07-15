# 导入必要的模块
import os
import sys
import pickle

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QLineEdit, QInputDialog, QMessageBox, QComboBox
# 导入必要的模块
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QCheckBox, QPushButton, QGridLayout, \
    QStyleFactory, QGroupBox, QLineEdit, QMessageBox, QSlider
from PyQt5.QtGui import QIcon, QFont, QBrush, QPalette, QPixmap
from PyQt5.QtCore import QSize

# 定义一个类来表示第一步的界面
from matplotlib import pyplot as plt

from classify.CNN_hand import CNN_train
from classify.KNN_hand import KNN_train
from classify.MLP_hand import MLP_train
from classify.SVM_hand import SVM_train
from get_hand_data import get_hand_data
from get_hand_score_data import get_hand_score_data
from cfg_parameter import CfgData
import re

# 定义一个类来表示第一步的界面
class StepOne(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('第一步：录取新手势数据或跳过')
        self.resize(400, 300)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 10))
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
        # 创建一个标签，用来提示用户输入动作数
        self.label2 = QLabel('如需要加入新动作，必须是一个正整数', self)
        self.label2.move(50, 50)

        # 创建一个按钮，用来弹出一个对话框，让用户输入动作数，并检查是否合法
        self.button2 = QPushButton('输入', self)
        self.button2.move(150, 100)
        self.button2.clicked.connect(self.getInput)

        self.button3 = QPushButton('跳过', self)
        self.button3.move(150, 150)
        self.button3.clicked.connect(self.skip)

        str1=''
        for i in motion_dict_np:
            str1=str1+i+" "
            if len(str1)==30:
                str1=str1+"\n"
        self.label3 = QLabel('当前已有动作：\n%s'%str1, self)
        self.label3.move(50, 200)

    def skip(self):
        self.stepThree = StepThree(0)
        self.close()
        self.stepThree.show()

    def getInput(self):
        # 弹出一个对话框，让用户输入动作数，并返回一个整数和一个布尔值
        classNum, ok = QInputDialog.getInt(self, '输入', '请输入新动作数量')

        # 检查用户输入是否合法，即是否是一个正整数
        if not ok:
            QMessageBox.warning(self, '警告', '请输入一个正整数')
            return
        if classNum <= 0:
            QMessageBox.warning(self, '警告', '动作数必须大于0')
            return
        self.stepTwo = StepTwo(classNum)
        self.close()
        self.stepTwo.show()

    def checkData(self):
        # 读取motion_data.csv文件
        motionData = pd.read_csv('motion_data.csv', encoding='utf-8')
        # 获取当前动作组的行
        row = motionData.loc[motionData['motion name'] == self.motion]
        # 获取当前动作组的动作数
        classNum = row['class num'].values[0]
        # 获取当前动作组的动作列
        actionCols = ['动作{}'.format(i) for i in range(classNum)]
        # 检查动作列是否都有值
        if row[actionCols].notnull().all().all():
            # 弹出一个对话框，询问用户是否开始训练模型
            reply = QMessageBox.question(self, '提示', '分类数据记录完毕，是否开始记录评分数据？', QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)
            # 如果用户选择是，关闭当前窗口，并打开第五步界面
            if reply == QMessageBox.Yes:
                self.close()
                self.stepFive = StepFive(self.motion)
                self.stepFive.show()

            # 如果用户选择否，返回上一步
            else:
                skip = True
                return skip
        else:
            skip = False
            return skip

class StepTwo(QWidget):
    def __init__(self,classnum):
        super().__init__()
        self.newclassnum=classnum#我这里因为只有一个输入框所以目前都只添加一个新动作，后续就根据newclassnum来添加输入框即可(代码也要有点改变)
        self.k=1#self.k记录当前输入的新动作数量
        self.initUI()     

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('第二步：记录新动作名称')
        self.resize(400, 300)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 10))
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
        # 创建一个标签，用来提示用户输入动作组的中文名
        self.label1 = QLabel('请输入第%d个动作的名称'%self.k, self)
        self.label1.move(50, 50)

        str1=''
        for i in motion_dict_np:
            str1=str1+i+" "
            if len(str1)==30:
                str1=str1+"\n"
        self.label2 = QLabel('当前已有动作：\n%s'%str1, self)
        self.label2.move(50, 200)

        # 创建一个文本框，用来接收用户输入的动作组的中文名
        self.lineEdit1 = QLineEdit(self)
        self.lineEdit1.move(50, 100)
        self.lineEdit1.resize(300, 30)

        # 创建一个按钮，用来确认用户输入的动作组的中文名，并检查是否合法
        self.button1 = QPushButton('确认', self)
        self.button1.move(150, 150)
        self.button1.clicked.connect(self.checkInput)

    def checkInput(self):
        global motion_dict_np
        # 获取用户输入的动作组的中文名
        inputText = self.lineEdit1.text()
        # 检查用户输入是否合法，即是否为空
        if inputText == '':
            QMessageBox.warning(self, '警告', '请输入动作组的名称')
            return
        motionDict = motion_dict_pkl
        motionDict[inputText] = len(motionDict)

        motion_dict_np.append(inputText)
        print(motion_dict_np)
        np.save('public/zero_nine.npy',motion_dict_np)
        with open('public/zero_nine.pkl', 'wb') as f:
            pickle.dump(motionDict, f)
        print(motionDict)

        
        # 若达到设定值，关闭当前窗口，并打开下一个窗口，即录取分类数据的窗口
        if(self.k==self.newclassnum):
            self.stepThree = StepThree(self.newclassnum)
            self.close()
            self.stepThree.show()
        else:
            self.k=self.k+1
            self.lineEdit1.clear()
            return
        

class StepThree(QWidget):
    def __init__(self,classnum):
        super().__init__()
        self.newclassnum=classnum
        self.initUI()
    
    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('第三步：选择需要的动作')#后面还可以改为可以设置顺序
        self.resize(400, 300)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 10))
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
        grid = QGridLayout()
        self.setLayout(grid)

        input=QGroupBox('本组动作名称(英文)')
        grid.addWidget(input, 0, 0)
        input_grid = QGridLayout()
        input.setLayout(input_grid)
        self.name = QLineEdit(self)
        input_grid.addWidget(self.name, 0, 0)
        
        # 创建一个按钮，让用户确定
        self.button4 = QPushButton('确定', self)
        self.button4.move(450, 250)
        self.button4.clicked.connect(lambda:self.next())
        input_grid.addWidget(self.button4, 1, 0)

        self.button5 = QPushButton('跳过', self)
        self.button5.move(450, 300)
        self.button5.clicked.connect(lambda:self.skip())
        input_grid.addWidget(self.button5, 2, 0)

        actions_group = QGroupBox('本次训练所需要的动作')
        grid.addWidget(actions_group, 0, 1)
        # 创建手指组的网格布局
        actions_grid = QGridLayout()
        actions_group.setLayout(actions_grid)
        self.check=[]
        # 创建复选框，添加到手指组的网格布局中
        for i in range(len(motion_dict_np)-self.newclassnum):
            self.check.append(QCheckBox(motion_dict_np[i]))
            actions_grid.addWidget(self.check[i], i//3, i-i//3*3)
        for i in range(len(motion_dict_np)-self.newclassnum,len(motion_dict_np)):
            self.check.append(QCheckBox(motion_dict_np[i]))
            actions_grid.addWidget(self.check[i], i//3, i-i//3*3)
            #将这些复选框的状态设置为选中
            self.check[i].setChecked(True)

    def next(self):
        action_list=self.generate_action_list()
        if len(action_list)==0:
            QMessageBox.warning(self, '警告', '请至少选择一个动作')
            return
        r = re.compile("^[a-zA-Z]+$")
        self.name1=self.name.text()
        if not r.match(self.name1):
            QMessageBox.warning(self, '警告', '命名必须是英文的格式')
            return
        np.save(self.name1+".npy",action_list)
        self.stepFour = StepFour(action_list,self.name1)
        self.close()
        self.stepFour.show()

    def skip(self):
        action_list=[]
        for i in range(len(motion_dict_np)):
            action_list.append(motion_dict_np[i])
        self.name1="none"
        self.stepFour = StepFour(action_list,self.name1)
        self.close()
        self.stepFour.show()
        
    def generate_action_list(self):
        action_list=[]
        for i in range(len(motion_dict_np)):
            if self.check[i].isChecked():
                action_list.append(motion_dict_np[i])
        return action_list
    
# 定义一个类来表示第四步的界面
class StepFour(QWidget):
    def __init__(self, action_list,name):
        super().__init__()
        self.action_list = action_list
        self.name=name
        self.record =old_actions
        self.initUI() 

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('第四步：录取分类数据')
        self.resize(400, 300)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 10))
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
        grid = QGridLayout()
        self.setLayout(grid)

        self.setPalette(palette)
        # 创建一个标签，用来提示用户选择当前录取数据的标签
        self.label3 = QLabel('请选择想要录取分类数据的动作', self)
        self.label3.move(50, 50)
        self.combo = QComboBox(self)
        grid.addWidget(self.combo, 0, 0)

        # 把文件名添加到下拉菜单中
        for i in self.action_list:
            self.combo.addItem(i)

        # 设置下拉菜单的样式表，使用text-overflow属性设置省略号的位置为右侧
        self.combo.setStyleSheet('text-overflow: ellipsis;')
        # 创建一个按钮，用来弹出一个对话框，让用户选择当前录取数据的标签，并检查是否合法
        self.button5 = QPushButton('确定', self)
        self.button5.move(150, 100)
        self.button5.clicked.connect(self.getLabel)
        self.button6 = QPushButton('跳过', self)
        self.button6.move(150, 200)
        self.button6.clicked.connect(self.skip)

    def skip(self):
        for i in self.action_list:
            if i not in self.record:
                QMessageBox.warning(self, '警告', '有新增动作无分类数据')
                return
        cfg = CfgData()
        all_actions=np.load('public/zero_nine.npy').tolist()
        data = pd.read_csv(cfg.project_dir()+'/pose_data/zero_nine.csv', header=None)
        # data = pd.read_csv('pose_data\jsb_data.csv', header=None)
        action_index=[]
        for i in self.action_list:
            action_index.append(all_actions.index(i))
        data=data.loc(axis=0)[data.iloc[:, -1].isin(action_index)]
        df=pd.DataFrame(data)
        df.to_csv(cfg.project_dir()+'/pose_data/'+self.name+'.csv',index=False,header=False)
        self.stepFive = StepFive(self.action_list,self.name)
        self.close()
        self.stepFive.show()
    def getLabel(self):
        label = self.combo.currentText()
        # 检查字典中是否已经存在该标签
        if label in self.record:
            # 弹出一个对话框，提示用户该标签已经录过，并询问是否继续录数据
            reply = QMessageBox.question(self, '提示', '该标签有历史数据，是否继续录数据？',
                                            QMessageBox.Yes | QMessageBox.No,
                                            QMessageBox.No)
            # 如果用户选择否，返回上一步
            if reply == QMessageBox.No:
                return
        #获得标签在motion_dict_np中的索引
        num = motion_dict_np.index(label)
        print(num)
        # 如果用户输入合法且选择继续录数据，关闭当前窗口，并打开下一个窗口，即录取分类数据的窗口
        # 调用get_hand_data函数，传入标签和动作组的英文名，开始录取分类数据
        #label是动作序号，之后可以改成动作名称，motion是动作组的英文名,目前是把所有动作都放到zero_nine里面了,之后可以分别放
        get_hand_data(label=num, database='zero_nine')
        # 弹出一个对话框，提示用户录取完成，并询问是否继续录取其他标签的数据
        reply = QMessageBox.question(self, '提示', '录取完成，是否继续录取其他标签的数据？', QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.Yes)

        # 如果用户选择是，关闭当前窗口，并打开上一个窗口，即选择标签的窗口
        if reply == QMessageBox.Yes:
            self.close()
            # 在这里更新字典中的动作名
            self.record.append(label)
            self.stepFour = StepFour(self.action_list,self.name)
            self.stepFour.show()

        # 如果用户选择否，生成相应的分类数据文件，关闭当前窗口，并打开下一个窗口，即录取评分数据的窗口
        else:
            cfg = CfgData()
            all_actions=np.load('public/zero_nine.npy').tolist()
            data = pd.read_csv(cfg.project_dir()+'/pose_data/zero_nine.csv', header=None)
            # data = pd.read_csv('pose_data\jsb_data.csv', header=None)
            action_index=[]
            for i in self.action_list:
                action_index.append(all_actions.index(i))
            data=data.loc(axis=0)[data.iloc[:, -1].isin(action_index)]
            df=pd.DataFrame(data)
            df.to_csv(cfg.project_dir()+'/pose_data/'+self.name+'.csv',index=False,header=False)
            self.close()
            self.stepFive = StepFive(self.action_list,self.name)
            self.stepFive.show()
        

# 定义一个类来表示第五步的界面
class StepFive(QWidget):
    def __init__(self,action_list,name):
        super().__init__()
        self.action_list = action_list
        self.name=name
        self.record=old_score_actions
        self.initUI()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('第五步：录取评分数据')
        self.resize(400, 300)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 10))
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
        grid = QGridLayout()
        self.setLayout(grid)

        self.setPalette(palette)
        # 创建一个标签，用来提示用户选择想要录取的动作
        self.label5 = QLabel('请选择想要录取评分数据的动作', self)
        self.label5.move(50, 50)

        self.combo = QComboBox(self)
        grid.addWidget(self.combo, 0, 0)

        # 把文件名添加到下拉菜单中
        for i in self.action_list:
            self.combo.addItem(i)

        # 设置下拉菜单的样式表，使用text-overflow属性设置省略号的位置为右侧
        self.combo.setStyleSheet('text-overflow: ellipsis;')
        # 创建一个按钮，用来弹出一个对话框，让用户选择想要录取的动作，并检查是否合法
        self.button6 = QPushButton('确定', self)
        self.button6.move(150, 100)
        self.button6.clicked.connect(self.getAction)
        self.button7 = QPushButton('跳过', self)
        self.button7.move(150, 200)
        self.button7.clicked.connect(self.skip)

    def skip(self):
        for i in self.action_list:
            if i not in self.record:
                QMessageBox.warning(self, '警告', '有新增动作无分类数据')
                return
        cfg = CfgData()
        all_actions=motion_dict_np
        data = pd.read_csv(cfg.project_dir()+'/score_sample/zero_nine.csv', header=None)
        # data = pd.read_csv('pose_data\jsb_data.csv', header=None)
        action_index=[]
        for i in self.action_list:
            action_index.append(all_actions.index(i))
        data=data.loc(axis=0)[data.iloc[:, -1].isin(action_index)]
        df=pd.DataFrame(data)
        df.to_csv(cfg.project_dir()+'/score_sample/'+self.name+'.csv',index=False,header=False)
        self.stepSix = StepSix(self.action_list,self.name)
        self.close()
        self.stepSix.show()
    def getAction(self):
        label = self.combo.currentText()
        # 检查字典中是否已经存在该标签
        if label in self.record:
            # 弹出一个对话框，提示用户该标签已经录过，并询问是否继续录数据
            reply = QMessageBox.question(self, '提示', '该标签有历史数据，是否继续录数据？',
                                            QMessageBox.Yes | QMessageBox.No,
                                            QMessageBox.No)
            # 如果用户选择否，返回上一步
            if reply == QMessageBox.No:
                return
        #获得标签在motion_dict_np中的索引
        num = motion_dict_np.index(label)
        #label是动作序号，之后可以改成动作名称，motion是动作组的英文名,目前是把所有动作都放到zero_nine里面了,之后可以分别放
        get_hand_score_data(label=num, database='zero_nine')

        # 弹出一个对话框，提示用户录取完成，并询问是否继续录取其他动作的数据
        reply = QMessageBox.question(self, '提示', '录取完成，是否继续录取其他动作的数据？', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes)

        # 如果用户选择是，关闭当前窗口，并打开上一个窗口，即选择动作的窗口
        if reply == QMessageBox.Yes:
            self.close()
            self.stepFive = StepFive(self.action_list,self.name)
            self.stepFive.show()

        # 如果用户选择否，关闭当前窗口，并打开下一个窗口，即训练模型的窗口
        else:
            cfg = CfgData()
            all_actions=motion_dict_np
            all_actions.append(label)
            data = pd.read_csv(cfg.project_dir()+'/score_sample/zero_nine.csv', header=None)
            # data = pd.read_csv('pose_data\jsb_data.csv', header=None)
            action_index=[]
            for i in self.action_list:
                action_index.append(all_actions.index(i))
            data=data.loc(axis=0)[data.iloc[:, -1].isin(action_index)]
            df=pd.DataFrame(data)
            df.to_csv(cfg.project_dir()+'/score_sample/'+self.name+'.csv',index=False,header=False)
            self.close()
            self.stepSix = StepSix(self.action_list,self.name)
            self.stepSix.show()


# 定义一个类来表示第六步的界面
class StepSix(QWidget):
    def __init__(self,action_list,name):
        super().__init__()
        self.action_list = action_list
        self.name=name
        self.initUI()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('第六步：训练模型')
        self.resize(400, 300)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 10))
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
        # 创建四个按钮，分别用于训练CNN,SVM,KNN,MLP模型
        self.button6_1 = QPushButton('训练模型', self)
        self.button6_1.move(50, 50)
        self.button6_1.clicked.connect(self.trainCNN)

        self.button6_2 = QPushButton('跳过', self)
        self.button6_2.move(50, 100)
        self.button6_2.clicked.connect(self.skip)

        # 创建一个新的按钮，用于进入下一步
        self.button6_5 = QPushButton('进入下一步', self)
        # 将按钮放在合适的位置
        self.button6_5.move(150, 250)
        # 为按钮绑定一个槽函数，用于关闭当前窗口，并打开下一个窗口
        self.button6_5.clicked.connect(self.nextStep)

    def skip(self):
        self.close()
        self.end = End()
        self.end.show()

    def trainCNN(self):
        # 从motion_data.csv文件中读取当前动作组的动作数
        classNum = len(self.action_list)
        print(classNum)
        # 调用CNN_train函数，传入动作数和动作组的英文名，开始训练CNN模型
        CNN_train(class_num=classNum, database=self.name,action_list=self.action_list)

        # 弹出一个对话框，提示用户训练完成，并询问是否继续训练其他模型
        reply = QMessageBox.question(self, '提示', '训练完成，是否继续训练其他模型？', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        # 如果用户选择是，关闭当前窗口，并打开上一个窗口，即训练模型的窗口
        if reply == QMessageBox.Yes:
            self.close()
            self.stepSix = StepSix()
            self.stepSix.show()

        # 如果用户选择否，关闭当前窗口，并打开下一个窗口，即结束的窗口
        else:
            self.close()
            self.end = End()
            self.end.show()


    def nextStep(self):
        # 关闭当前窗口
        self.close()
        # 打开下一个窗口，假设是第七步界面
        self.end = End()
        self.end.show()

class StepSeven(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('查看接手患者信息')
        self.resize(400, 300)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 10))
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
        grid = QGridLayout()
        self.setLayout(grid)

        self.setPalette(palette)

        self.combo = QComboBox(self)
        grid.addWidget(self.combo, 0, 0)

        # 获取log文件夹中的所有文件名
        fileNames = os.listdir('patients')
        # 把文件名添加到下拉菜单中
        for fileName in fileNames:
            patient,pinyin=fileName.split(',')
            self.combo.addItem(patient+','+pinyin)

        # 设置下拉菜单的样式表，使用text-overflow属性设置省略号的位置为右侧
        self.combo.setStyleSheet('text-overflow: ellipsis;')

        grid.addWidget(self.combo, 0, 0)

        # 创建一个按钮，用来弹出一个对话框，让用户选择想要录取的动作，并检查是否合法
        self.button6 = QPushButton('确定', self)
        self.button6.move(150, 100)
        self.button6.clicked.connect(lambda:self.next())
    
    def next(self):
        patient_pinyin = self.combo.currentText()
        print(type(patient_pinyin))
        patient, pinyin = patient_pinyin.split(',')
        # 关闭当前窗口
        self.close()
        # 打开下一个窗口，假设是第七步界面
        self.stepEight = StepEight(patient, pinyin)
        self.stepEight.show()

class StepEight(QWidget):
    def __init__(self, patient, pinyin):
        self.patient=patient
        self.pinyin=pinyin
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('查看患者%s信息'%self.patient)
        self.resize(400, 300)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 10))
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
        grid = QGridLayout()
        self.setLayout(grid)

        self.setPalette(palette)

        train_plan1=np.load('patients/'+self.patient+','+self.pinyin+'/'+self.pinyin+'.npy',allow_pickle=True)
        print(train_plan1)
        train_plan=[train_plan1[:-1].tolist(),int(train_plan1[-1])]

        # 创建一个下拉菜单，用来选择可视化的数据
        self.combo = QComboBox(self)
        grid.addWidget(self.combo, 0, 0)
        # 获取log文件夹中的所有文件名
        fileNames = os.listdir('patients/'+self.patient+','+self.pinyin+'/result')
        
        # 把文件名添加到下拉菜单中
        for fileName in fileNames:
            if fileName[-4:]=='.pkl' and len(fileName.split('_'))>2:
                self.combo.addItem(fileName)
        # 设置下拉菜单的样式表，使用text-overflow属性设置省略号的位置为右侧
        self.combo.setStyleSheet('text-overflow: ellipsis;')

        self.button3 = QPushButton('查看训练结果', self)
        grid.addWidget(self.button3, 1, 0)
        self.button3.clicked.connect(lambda:self.plot_score())

        actions_group = QGroupBox('训练方案')
        grid.addWidget(actions_group, 2, 0)
        # 创建手指组的网格布局
        actions_grid = QGridLayout()
        actions_group.setLayout(actions_grid)
        self.check=[]
        # 创建复选框，添加到手指组的网格布局中
        for i in range(len(motion_dict_np)):
            self.check.append(QCheckBox(motion_dict_np[i]))
            actions_grid.addWidget(self.check[i], i//3, i%3)
            if motion_dict_np[i] in train_plan[0]:
                self.check[i].setChecked(True)
        # 创建一个按钮，让用户确定
        self.button4 = QPushButton('确定', self)
        grid.addWidget(self.button4, 3, 0)
        self.button4.clicked.connect(lambda:self.next())
    
    def next(self):
        action_list=self.generate_action_list() 
        if len(action_list)==0:
            QMessageBox.warning(self, '警告', '请至少选择一个动作', QMessageBox.Yes)
        else:
            action_list.append('3')
            cfg = CfgData()
            np.save('patients/'+self.patient+','+self.pinyin+'/'+self.pinyin+'.npy', action_list,allow_pickle=True)
            all_actions=motion_dict_np
            action_index=[]
            for i in action_list[:-1]:
                action_index.append(all_actions.index(i))
            data = pd.read_csv(cfg.project_dir()+'/pose_data/zero_nine.csv', header=None)
            data=data.loc(axis=0)[data.iloc[:, -1].isin(action_index)]
            df=pd.DataFrame(data)
            df.to_csv(cfg.project_dir()+'/pose_data/'+self.pinyin+'.csv',index=False,header=False)
            data = pd.read_csv(cfg.project_dir()+'/score_sample/zero_nine.csv', header=None)
            data=data.loc(axis=0)[data.iloc[:, -1].isin(action_index)]
            df=pd.DataFrame(data)
            df.to_csv(cfg.project_dir()+'/score_sample/'+self.pinyin+'.csv',index=False,header=False)
            self.close()
            self.stepSix = StepSix(action_list[:-1],self.pinyin)
            self.stepSix.show()
    
    def generate_action_list(self):
        action_list=[]
        for i in range(len(motion_dict_np)):
            if self.check[i].isChecked():
                action_list.append(motion_dict_np[i])
        return action_list

    def plot_score(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号
        # plt.style.use('ggplot')
        # 获取下拉菜单的当前选择
        file_name = self.combo.currentText()
        # 如果没有选择任何文件，则返回
        if not file_name:
            return
        # 拼接文件的路径，把log文件夹作为第一个参数
        file_path = os.path.join('patients/'+self.patient+','+self.pinyin+'/result', file_name)
        # 使用pickle.load方法从文件中加载数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # # 从字典中获取需要的值
        # motion = data['motion']
        # model_name = data['model_name']
        # class_num = data['class_num']
        score_list = data['result']['score_list']
        lable = data['result']['lable']
        # num = data['result']['num']
        # standard_score = data['result']['standard_score']
        # # 创建一个空列表，用来存储重点关注的手指
        # fingers = []
        # # 把列表转换成字符串，用逗号分隔
        # fingers_str = ','.join(fingers)

        # 创建一个新的图形窗口
        plt.figure(figsize=(10, 10))
        # average_score = np.mean(score_list)
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
        # plt.scatter(range(len(lable)), score_list, c=list(map(color_map, score_list)), marker='o')

        # 设置横轴的刻度和标签，使用lable作为标签
        # plt.xticks(range(len(lable)), lable)
        # # 设置纵轴的范围为0-100
        # plt.ylim(0, 100)
        # # 设置标题和坐标轴标签，加上完成的动作数，减小字体大小
        # plt.title('手部识别与评分折线图（完成' + str(num) + '个动作）', fontsize=15)
        # plt.xlabel('动作评分序列')
        # plt.ylabel('评分')
        # # 显示图例和网格
        # # 创建一个空列表，用来存储图例标签
        # plt.legend()
        # legend_labels = []
        # # 遍历lable中的值，根据值获取action_list中的内容，然后拼接成字符串，添加到列表中
        # # 获取下拉菜单的当前选项，即动作的中文名
        # motion_chinese = motion
        # # 遍历字典的键值对，找到对应的英文名
        # for motion_english, motion_chinese in motion_dict.items():
        #     if motion_chinese == motion_chinese:
        #         break
        # # 在DataFrame中查找对应的行，返回一个Series对象
        # row = df[df['motion name'] == motion_english].iloc[0]
        # # 从Series中获取class num的值，转换为整数
        # class_num = int(row['class num'])
        # # 从Series中获取动作列表，去掉空值和nan值
        # action_list = [x for x in row[2:] if x and not pd.isna(x)]
        # for i in lable:
        #     legend_label = str(i) + ': ' + action_list[i]
        #     legend_labels.append(legend_label)

        # # 显示图例，使用labels参数传递自定义的图例标签
        # plt.legend(labels=legend_labels, handlelength=0)

        # plt.grid(axis='y')

        # # 创建一个子图，用来绘制柱状图
        # plt.subplot(2, 2, (3, 4))
        # # 计算每个动作的平均分，并存储到一个列表中
        # average_scores = []
        # for action in set(lable):
        #     average_score = sum(score_list[i] for i in range(len(lable)) if lable[i] == action) / lable.count(action)
        #     average_scores.append(average_score)

        # # 把lable转换为一个集合，去除重复的元素，并排序
        # x = sorted(set(lable))

        # # 获取完成率和达标率的值
        # qualified_rate = data['result']['qualified_rate']

        # # 把完成率和达标率添加到x和average_scores列表中
        # x.append('达标率/%')
        # average_scores.append(qualified_rate * 100)
        # x = list(map(str, x))
        # # 绘制竖直柱状图，横轴为动作类别，纵轴为平均分，设置颜色和标签
        # plt.bar(x, average_scores, color='green', label='平均分', width=0.25)

        # # 设置横轴的刻度和标签，使用x作为标签
        # plt.xticks(x, x)

        # # 设置标题和坐标轴标签，减小字体大小
        # plt.title('手部识别与评分柱状图', fontsize=16)
        # plt.xlabel('动作类别')
        # plt.ylabel('平均分')
        # # 显示图例和网格
        # plt.legend()
        # plt.grid()

        # # 调整子图之间的间距
        # plt.tight_layout()
        # # 调整标题和子图之间的间距
        # plt.subplots_adjust(top=0.8)

        plt.show()

# 定义一个类来表示结束的界面
class End(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('结束')
        self.resize(400, 300)
        # 设置窗口的字体为微软雅黑，大小为16
        self.setFont(QFont('Microsoft YaHei', 10))
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
        # 创建一个标签，用来提示用户已经完成所有步骤
        self.label7 = QLabel('成功增加新的动作，你已经完成了所有步骤！', self)
        self.label7.move(100, 100)

        # 创建一个按钮，用来退出程序
        self.button7 = QPushButton('回到初始界面', self)
        self.button7.move(150, 200)
        self.button7.clicked.connect(self.exit)

    def exit(self):
        # 退出程序
        self.close()
        self.start = Start()
        self.start.show()


# 定义一个类来表示开始界面
class Start(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('RehabAssist远程智能上肢康复系统')
        self.resize(400, 300)
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
        # 创建一个网格布局，用来放置按钮
        grid = QGridLayout()
        self.setLayout(grid)

        # 创建一个标签，用来显示欢迎信息
        self.label = QLabel('欢迎使用手势识别医生端系统！', self)
        self.label.move(100, 50)
        grid.addWidget(self.label, 0, 0, 1, 2)

        # 创建两个按钮，分别用于开始录取数据和可视化数据
        self.button1 = QPushButton('添加数据/训练方案', self)
        self.button1.setIcon(QIcon('record.png'))
        # 设置按钮的图标大小为32x32像素
        self.button1.setIconSize(QSize(32, 32))
        grid.addWidget(self.button1, 1, 0)
        # 点击按钮时调用开始录取数据的函数
        self.button1.clicked.connect(self.start_public)

        # self.button2 = QPushButton('可视化数据', self)
        # self.button2.setIcon(QIcon('chart.png'))
        # # 设置按钮的图标大小为32x32像素
        # self.button2.setIconSize(QSize(32, 32))
        # grid.addWidget(self.button2, 2, 1)
        # # 点击按钮时调用可视化数据的函数
        # self.button2.clicked.connect(self.plot_score)

        self.button3 = QPushButton('查看接手患者', self)
        self.button3.setIcon(QIcon('record.png'))
        # 设置按钮的图标大小为32x32像素
        self.button3.setIconSize(QSize(32, 32))
        grid.addWidget(self.button3, 2, 0)
        # 点击按钮时调用开始录取数据的函数
        self.button3.clicked.connect(self.start_private)

    def start_public(self):
        # 关闭当前窗口，并打开第一步的窗口，即录取分类数据的窗口
        self.close()
        self.stepOne = StepOne()
        self.stepOne.show()
    
    def start_private(self):
        # 关闭当前窗口，并打开第一步的窗口，即录取分类数据的窗口
        self.close()
        self.stepSeven = StepSeven()
        self.stepSeven.show()

    


if __name__ == '__main__':
    # 读取之前保存的动作字典
    with open("public/zero_nine.pkl", "rb") as tf:
        motion_dict_pkl=pickle.load(tf)
    motion_dict_np = np.load('public/zero_nine.npy', allow_pickle=True).tolist()
    print(motion_dict_np)
    old_actions=motion_dict_np.copy()
    old_score_actions=motion_dict_np.copy()


    # 创建应用程序对象和窗口对象，并显示窗口
    app = QApplication(sys.argv)
    window = Start()
    window.show()
    # 进入应用程序的事件循环
    sys.exit(app.exec_())
