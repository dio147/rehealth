from django.db import models

# Create your models here.
class User(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    user_identify = models.CharField(max_length=50, default='患者')

    class Meta:
        # 该表名
        db_table = 'user'

    def __str__(self):
        return self.name


class Doctor(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=10, verbose_name='姓名')
    password = models.CharField(max_length=50, verbose_name='密码')
    pinyin = models.CharField(max_length=10)

    class Meta:
        # 该表名
        db_table = 'doctor'

    def __str__(self):
        return self.name


class Patient(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=10, verbose_name='姓名')
    password = models.CharField(max_length=50, verbose_name='密码')
    pinyin = models.CharField(max_length=10)
    doc_name = models.ForeignKey(Doctor, on_delete=models.CASCADE, verbose_name='外键')
    age = models.IntegerField(default=0, verbose_name='年龄')
    gender = models.CharField(max_length=10, verbose_name='性别')
    medical_history = models.CharField(max_length=2000, verbose_name='病例')
    training_plan = models.CharField(max_length=2000, verbose_name='训练计划')
    training_result = models.CharField(max_length=2000, verbose_name='训练结果')
    classify_dataset = models.CharField(max_length=100, verbose_name='分类数据')
    classify_model = models.CharField(max_length=100, verbose_name='分类模型')
    score_dataset = models.CharField(max_length=100, verbose_name='评分数据')

    class Meta:
        # 该表名
        db_table = 'patient'

    def __str__(self):
        return self.name
