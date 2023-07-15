import os
from collections import deque

class FixedSizeQueue:
    def __init__(self, max_length):
        self.queue = deque(maxlen=max_length)

    def push(self, item):
        self.queue.append(item)

    def pop(self):
        return self.queue.popleft()

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear()

    def average(self):
        if not self.queue:
            return None
        return sum(self.queue) / len(self.queue)

class CfgData:
    """
    把文件命名规范一下，动作命名为‘部位_类型’ 例如“hand_five_motion"
    第二个是模型类别，设置为可选的就行，有：cnn，knn，mlp，svm
    """

    def __init__(self, database='zero_nine', model='CNN'):
        self.database = database
        self.model = model.upper()

    def project_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def score_path(self):
        # 添加文件名后缀.csv
        filename = self.database + '.csv'
        # 拼接路径
        score_path = os.path.join(os.getcwd(), 'score_sample', filename)
        return score_path

    def model_path(self):

        if self.model == 'CNN':
            filename = self.database + '.pth'
        else:
            filename = self.database + '.pkl'

        model_path = self.model + '_model'
        # 拼接classify_path和filename
        model_path = os.path.join(self.project_dir(), model_path, filename)
        return model_path

    def classify_path(self):
        filename = self.database + '.csv'

        # 拼接路径
        classify_path = 'pose_data'

        # 拼接classify_path和filename
        classify_path = os.path.join(self.project_dir(), classify_path, filename)

        return classify_path


# import pickle
# # 保存
# motion_dict = {'hand_rec': '手部张开', 'hand_hold': '手部握拳', 'hand_show': '手部展示'}
# f_save = open('motion_dict.pkl', 'wb')
# pickle.dump(motion_dict, f_save)
# f_save.close()
# # 加载
# f_read = open('motion_dict.pkl', 'rb')
# motion_dict2 = pickle.load(f_read)
# print(motion_dict2)
# f_read.close()

# cfg = CfgData('hand_rec_model', 'KNN')
# print(cfg.score_path())
# print(cfg.model_path())
