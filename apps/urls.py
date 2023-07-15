from django.urls import path
from apps.views import login, index_doc, index_user, test, chakan_user, create, chakan_doc, create_doc, chakan_doc_user

urlpatterns = [
    path('login/', login),
    path('index_user/', index_user),
    path('index_doc/', index_doc),
    path('test/', test),
    path('chakan_user/', chakan_user),
    path('create/', create),
    path('chakan_doc/', chakan_doc),
    path('chakan_doc/chakan_user/', chakan_doc_user),
    path('chakan_doc/create_doc/', create_doc),

]

