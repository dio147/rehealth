import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""
用于训练分类模型的，注意修改数据路径和保存模型的文件名就行
改完路径和文件名直接运行就可以
"""



# 定义一个CNN模型类
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)  # 这里不需要指定输入形状
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 20, out_features=16)  # 这里修改第一个全连接层的输入数，因为池化后的特征维度为32 * 20
        self.fc2 = nn.Linear(in_features=16, out_features=class_num)  # 这里修改最后全连接层的输出数为2

    def forward(self, x):
        # x = x.unsqueeze(1)  # 这里增加一个通道维度，因为你的输入是一维的
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)  # 这里使用softmax激活函数
        return x

    def predict(self, x):
        # 将输入转换为张量，并增加一个批次维度
        x = torch.from_numpy(x).float().unsqueeze(0)
        # 将输入移动到GPU上（如果有）
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)

        # 计算输出
        output = self.forward(x)

        # 获取预测结果
        pred = output.argmax(1).item()

        return pred

if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv('F:/medical_match/pose_data/upper_data.csv', header=None)
    class_num = 2  # 这里预设分类数
    # 分离特征和标签
    X = data.iloc[:, :-2]  # 前42列是特征数据
    y = data.iloc[:, -1]  # 最后一列是标签
    hand = data.iloc[:, -2]  # 倒数第二列是左右手信息
    # 将特征转换为三维数组，形状为(样本数, 1, 42)，1是通道数，42是特征的维度
    X = np.array(X).reshape(-1, 1, 42)

    # 将标签转换为整数数组，形状为(样本数,)
    y = np.array(y)

    # 划分训练集和测试集，比例为8:2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 将训练集和测试集转换为张量
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    # 将训练集和测试集封装为数据集对象
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # 创建一个数据加载器对象，用于批次处理
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    # 创建一个CNN模型对象，并移动到GPU上（如果有）
    model = CNN()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 定义迭代次数
    epochs = 300
    # epochs = 50

    # 创建两个空列表，用于存储训练损失和准确率
    train_loss_list = []
    train_acc_list = []

    # 创建两个空列表，用于存储测试损失和准确率
    test_loss_list = []
    test_acc_list = []

    # 循环进行训练
    for epoch in range(epochs):
        # 初始化训练损失和准确率为0
        train_loss = 0.0
        train_acc = 0.0

        # 遍历每一个批次的数据
        for inputs, labels in train_loader:
            model.train()
            # 将输入和标签移动到GPU上（如果有）
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播，计算输出和损失
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播，更新参数
            loss.backward()
            optimizer.step()

            # 累加损失和准确率
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()

        # 计算并打印平均训练损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_dataset)
        print('Epoch %d, Train loss: %.3f, Train accuracy: %.3f' % (epoch + 1, train_loss, train_acc))

        # 将训练损失和准确率添加到列表中
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # 初始化测试损失和准确率为0
        test_loss = 0.0
        test_acc = 0.0

        # 遍历每一个批次的数据
        for inputs, labels in test_loader:
            model.eval()
            # 将输入和标签移动到GPU上（如果有）
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播，计算输出和损失
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 累加损失和准确率
            test_loss += loss.item()
            test_acc += (outputs.argmax(1) == labels).sum().item()

        # 计算并打印平均测试损失和准确率
        test_loss = test_loss / len(test_loader)
        test_acc = test_acc / len(test_dataset)
        print('Epoch %d, Test loss: %.3f, Test accuracy: %.3f' % (epoch + 1, test_loss, test_acc))

        # 将测试损失和准确率添加到列表中
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    # 可视化训练效果，绘制损失和准确率的曲线图
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_loss_list, label='test_loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(train_acc_list, label='train_accuracy')
    plt.plot(test_acc_list, label='test_accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    # 保存模型到文件中（你需要自己指定文件名）
    torch.save(model, '/prework/upper_model.pth')
    # torch.save(model, 'CNN_model/cnn_game.pth')
