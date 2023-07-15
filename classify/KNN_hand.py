import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from cfg_parameter import CfgData

"""
用于训练分类模型的，注意修改数据路径和保存模型的文件名就行
改完路径和文件名直接运行就可以
"""


def KNN_train(motion='hand_rec'):
    # 这里修改动作名
    # 方便调试预设了，这里记得修改
    model_name = 'knn'
    cfg = CfgData(motion_class=motion, model=model_name)

    # 读取csv文件
    df = pd.read_csv(cfg.classify_path())
    # df = pd.read_csv('F:\medical_match\pose_data\jsb_data.csv')
    # 提取特征和标签
    X = df.iloc[:, :-2].values  # 倒数第2列前是特征
    y = df.iloc[:, -1].values  # 最后一列是标签

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 定义要搜索的参数范围
    parameters = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan', 'minkowski']}

    # 创建一个KNN对象
    knn = KNeighborsClassifier()

    # 创建一个GridSearchCV对象，用5折交叉验证和准确率作为评分标准
    clf = GridSearchCV(knn, parameters, cv=5, scoring='accuracy')

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

    # 打印分类报告和混淆矩阵
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # 绘制真实标签和预测标签的柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(['True', 'Predicted'], [y_test.mean(), y_pred.mean()])
    plt.ylabel('Label')
    plt.title('Comparison of true and predicted labels')
    plt.show()

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

    # 保存模型到文件

    joblib.dump(clf, cfg.model_path())
    # joblib.dump(clf, 'KNN_model/KNN_game.pkl')
