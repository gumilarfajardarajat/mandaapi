"""mywebsite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.http import HttpResponse
from django.http import JsonResponse
import numpy as np
import pandas as pd
import requests
import json
import cv2
from sklearn import *


from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

#method view

def warna(request,r,g,b):
    result = rgbToHSV(r,g,b)
    df = pd.read_csv('Bibir Training.csv')
    x = df.drop(["class"], axis=1).values
    y = df['class'].values
    clf = svm.SVC(gamma=0.1, decision_function_shape='ovr', kernel='rbf')
    clf.fit(x, y)
    (x1,x2,x3) = (result[0],result[1],result[2])
    # (x1,x2,x3) = (255,252,105)

    testing = np.array([[x1, x2, x3]])
    temp = clf.predict(testing)
    classlip = temp[0]

    df_rek = pd.read_csv('Rekomendasi Lipstick.csv')
    data = df_rek.loc[df_rek['class'] == classlip]
    json_string = data.to_json(orient='records')
    obj = json.loads(json_string)
    # result = {'rekomendasi':obj}
    # return JsonResponse(obj, safe=False)
    return HttpResponse(json.dumps(obj), content_type="application/json")


def rgbToHSV(r,g,b):
    color = (r,g,b)
    color = np.array([[color]],dtype="uint8")
    hsv = cv2.cvtColor(color,cv2.COLOR_RGB2HSV)
    result = hsv[0][0]
    return result

def welcome(request):
    return HttpResponse("Welcome To Django")

urlpatterns = [
    path('welcome/', welcome),
    path('warna/<int:r>/<int:g>/<int:b>',warna)
]
