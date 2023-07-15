from django.shortcuts import render
from apps.models import User
from django.shortcuts import render, redirect
from django.http import HttpResponse
import patient_1
import datetime
from apps import models
from xpinyin import Pinyin
from django.forms.models import model_to_dict

# Create your views here.
def chakan_doc_user(request):
    doctor_name = request.GET.get('name')
    patient = patient_1.data_patient('李四')
    plot1_x, plot1_y, plot2_x, plot2_y, qualified_rate, standard_score = patient_1.plot_data(
        'patients/李四,Lisi/result/Lisi_2023_07_13_23_13_20.pkl', patient['action_list'], patient['cycle_num'])
    data = [20.1, 30, 40, 50, 60, 70, 80]
    plot1_x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    print(type(plot1_x), plot1_y)
    print(plot2_x, plot2_y)
    plot2_y[-1] = 90.55160941507427
    context = {
        'username': doctor_name,
        'data1': data,
        'plot1_x': plot1_x,
        'plot1_y': plot1_y,
        'plot2_x': plot2_x,
        'plot2_y': plot2_y,
        ##添加数据
    }
    return render(request, 'doc_chakan_user.html', context=context)

def create_doc(request):
    return HttpResponse('ok')

def chakan_doc(request):
    patient_name = request.GET.get('patient_name')
    doctor_name = request.GET.get('doctor_name')
    object_p = models.Patient.objects.filter(name__exact=patient_name).all()[0]
    object_p = model_to_dict(object_p)
    object_p['name'] = doctor_name
    object_p['patient_name'] = patient_name
    print(object_p)
    # print(object_p['age'])
    return render(request, 'doc_chakan.html', context=object_p)


def chakan_user(request):
    patient = patient_1.data_patient('李四')
    plot1_x,plot1_y,plot2_x,plot2_y,qualified_rate,standard_score=patient_1.plot_data('patients/李四,Lisi/result/Lisi_2023_07_13_23_13_20.pkl',patient['action_list'],patient['cycle_num'])
    data = [20.1,30,40,50,60,70,80]
    plot1_x = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
    plot2_y[-1]=90.55160941507427
    context = {
        'username': request.GET.get('name'),
        'data1': data,
        'plot1_x': plot1_x,
        'plot1_y': plot1_y,
        'plot2_x': plot2_x,
        'plot2_y': plot2_y,
        ##添加数据
    }
    return render(request, 'chakan_user.html', context=context)

def create(request):
    dict_p1 = {'name':'张三', 'pinyin':'Zhangsan', 'age':'50', 'gender':'男', 'password':'123456',
    'medical_history':"[['%s','xxx'],['%s','yyy']]" % (
     str(datetime.date(year=2020, month=8, day=31)), str(datetime.date(year=2021, month=9, day=13))),
    'training_plan':"['伸展','勾拳','握拳','7字型','直拳','3']", 'training_result':"patients/张三,Zhangsan/result/",
    'classify_dataset':'pose_data\Zhangsan.csv', 'classify_model':'CNN_model\Zhangsan.pth',
    'score_dataset':'score_sample\Zhangsan.csv','doc_name_id':'1'}
    dict_p2 = {'name': '李四', 'pinyin': 'Lisi', 'age': '53', 'gender': '女', 'password':'123456',
    'medical_history':"[['%s','xxx'],['%s','yyy']]" % (
     str(datetime.date(year=2020, month=8, day=31)), str(datetime.date(year=2021, month=9, day=13))),
    'training_plan':"['伸展','勾拳','握拳','7字型','直拳','3']", 'training_result':"patients/李四,Lisi/result/",
    'classify_dataset':'pose_data\Lisi.csv', 'classify_model':'CNN_model\Lisi.pth',
    'score_dataset':'score_sample\Lisi.csv','doc_name_id':'1'}
    models.Patient.objects.create(**dict_p2)
    models.Patient.objects.create(**dict_p1)
    return HttpResponse('OK')


def test(request):
    patient = patient_1.data_patient('李四')
    # print(patient['action_list'],patient['cycle_num'])
    # result=patient_1.run_classify(patient['action_list'],patient['cycle_num'],patient['classify_model'],patient['score_dataset'])
    # patient_1.save_score(result,patient['pinyin'],patient['training_result'])
    # file_list, file_list_path = patient_1.get_result__file(patient['training_result'])
    plot1_x,plot1_y,plot2_x,plot2_y,qualified_rate,standard_score=patient_1.plot_data('patients/李四,Lisi/result/Lisi_2023_07_13_23_13_20.pkl',patient['action_list'],patient['cycle_num'])
    plot1_x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    print(type(plot1_x), plot1_y)
    print(plot2_x, plot2_y)
    context = {
        'username': request.GET.get('name'),
        'plot1_x': plot1_x,
        'plot1_y': plot1_y,
        'plot2_x': plot2_x,
        'plot2_y': plot2_y,
    }
    return render(request, 'test.html', context=context)


def index_user(request):
    username = request.GET.get('name')
    context = {
        'username': username,
    }
    return render(request, 'index_user.html', context=context)


def index_doc(request):
    username = request.GET.get('name')
    context = {
        'username': username
    }
    return render(request, 'index_doc.html', context)


def login(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    if request.method == 'POST':
        print(request.POST)
        # query_dict = request.POST.copy()
        query_dict = dict(request.POST.copy())
        # patient = login_patient('')
        # result, _,_,_ _, _= plot_data('patients/李四,Lisi/result/Lisi_2023_07_12_15_26_13.pkl',
        #                               patient['action_list'],patient['cycle_num'])
        # print(query_dict.keys().__contains__('请输入用户名-登录'))
        if query_dict.keys().__contains__('请输入用户名-登录'):
                username = request.POST.get('请输入用户名-登录')
                password = request.POST.get('请输入密码-登录')
                usr_identify = request.POST.get('用户身份')

                context = {
                    'username': username,
                }
                if usr_identify=='医生':
                    user_obj = models.Doctor.objects.filter(name__exact=username, password__exact=password).first()
                    if user_obj:
                        return render(request, 'index_doc.html', context=context)
                    else:
                        return HttpResponse('用户名或密码错误')
                elif usr_identify=='患者':
                    user_obj = models.Patient.objects.filter(name__exact=username, password__exact=password).first()
                    if user_obj:
                        return render(request, 'index_user.html', context=context)
                    else:
                        return HttpResponse('用户名或密码错误')
                else:
                    return HttpResponse('请选择身份')

            # username = request.POST.get('请输入用户名-登录')
            # password = request.POST.get('请输入密码-登录')
            # usr_identify = request.POST.get('用户身份')
            # user_obj = User.objects.filter(name__exact=username, password__exact=password,
            #                                user_identify=usr_identify).first()
            # if user_obj:
            #     if usr_identify == '患者':
            #         # request.getRequestDispatcher(URL).forward(request, response)
            #         context = {
            #             'username': username,
            #         }
            #         return render(request, 'index-u.html', context=context)
            #     elif usr_identify == '医生':
            #         return render(request, 'index_doc.html', context=context)
            #     else:
            #         return HttpResponse('请选择登陆身份')
            # else:
            #     return HttpResponse('用户名或密码错误或身份错误')
        elif query_dict.keys().__contains__('请输入用户名-注册'):
            username = request.POST.get('请输入用户名-注册')
            password = request.POST.get('请输入密码-注册')
            repeat_password = request.POST.get('请输入确认密码-注册')
            user_identify = request.POST.get('用户身份')
            p = Pinyin()
            if not username:
                return HttpResponse('用户名不能为空')
            if not password:
                return HttpResponse('密码不能为空')
            if not repeat_password:
                return HttpResponse('确认密码不能为空')
            if not user_identify:
                return HttpResponse('请选择身份')
            if username and password and repeat_password and user_identify:
                if password == repeat_password:
                    # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
                    if user_identify=='医生':
                        user_project = models.Doctor.objects.filter(name__exact=username).first()
                        if user_project:
                            return HttpResponse('用户名已存在')
                        else:
                            models.Doctor.objects.create(name=username, password=password, pinyin=p.get_pinyin(username)).save()
                        return redirect('login/')
                    elif user_identify=='患者':
                        user_project = models.Patient.objects.filter(name__exact=username).first()
                        if user_project:
                            return HttpResponse('用户名已存在')
                        else:
                            models.Doctor.objects.create(name=username, password=password, pinyin=p.get_pinyin(username)).save()
                        return redirect('login/')
                else:
                    return HttpResponse('两次输入的密码不一致')
