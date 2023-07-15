import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import seaborn as sns
from cfg_parameter import CfgData

"""
用于训练分类模型的，注意修改数据路径和保存模型的文件名就行
改完路径和文件名直接运行就可以
"""

def SVM_train(motion = 'hand_rec'):
    model_name = 'svm'
    cfg = CfgData(motion_class=motion, model=model_name)

    # 读取csv文件
    df = pd.read_csv(cfg.classify_path())
    # df = pd.read_csv('pose_data\jsb_data.csv')
    # 提取特征和标签
    X = df.iloc[:, :-2].values  # 前63列是特征
    y = df.iloc[:, -1].values  # 最后一列是标签

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 定义要搜索的参数范围
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
    parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1], 'gamma': [0.01, 1]}
    # 创建一个SVC对象
    svc = SVC()

    # 创建一个GridSearchCV对象，用5折交叉验证和准确率作为评分标准
    clf = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')

    # 在训练集上进行网格搜索
    clf.fit(X_train, y_train)

    # 打印最优的参数组合和得分
    print(f"Best parameters: {clf.best_params_}")
    print(f"Best score: {clf.best_score_}")

    # 使用最优的参数在测试集上进行预测
    y_pred = clf.predict(X_test)

    # 计算并打印测试集上的准确率
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy}")

    # 保存模型到文件
    # Compute the confusion matrix

    joblib.dump(clf, cfg.model_path())
    # joblib.dump(clf, 'SVM_model/svm_game.pkl')
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()