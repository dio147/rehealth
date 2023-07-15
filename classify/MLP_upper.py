import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
"""
用于训练分类模型的，注意修改数据路径和保存模型的文件名就行
改完路径和文件名直接运行就可以
mlp的训练时间长主要是因为网格搜索的参数比较多，运行时的警告可以不用管
"""
# 读取csv文件
df = pd.read_csv('F:\medical_match\pose_data\upper_data.csv')
# df = pd.read_csv('pose_data\jsb_data.csv')


# 提取特征和标签
X = df.iloc[:, :-2].values # 倒数第2列前是特征
y = df.iloc[:, -1].values # 最后一列是标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 定义要搜索的参数范围
parameters = {'hidden_layer_sizes': [(50, 2), (100, 2), (200, 2)], 'activation': ["logistic", "relu", "tanh"],'alpha': [0.0001, 0.001, 0.01], 'max_iter': [500]} # 增加迭代次数到500

# 创建一个MLP对象
mlp = MLPClassifier()

# 创建一个GridSearchCV对象，用5折交叉验证和准确率作为评分标准
clf = GridSearchCV(mlp, parameters, cv=5, scoring='accuracy')

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

joblib.dump(clf, 'F:\medical_match\MLP_model\mlp_upper.pkl')
# joblib.dump(clf, 'MLP_model/mlp_game.pkl')


# 对分类的评估结果做可视化

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# 打印分类报告
print(classification_report(y_test, y_pred))
