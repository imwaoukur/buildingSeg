from django.shortcuts import render

# Create your views here.
from django.core.files.base import ContentFile
from django.core.files.temp import NamedTemporaryFile
from django.http import JsonResponse, HttpResponse
import base64
import json
import io
import os
from PIL import Image

from .local.predict import predict_img


def D_BASE64(origStr):
    # 当输入的base64字符串不是3的倍数时添加相应的=号
    if (len(origStr) % 4 == 1):
        origStr += "==="
    elif (len(origStr) % 4 == 2):
        origStr += "=="
    elif (len(origStr) % 4 == 3):
        origStr += "="
    origStr = bytes(origStr, encoding='utf8') # 看情况进行utf-8编码
    dStr = base64.b64decode(origStr)
    return dStr

def pixel_count(request):
    if request.method == 'POST':
        # 获取前端发送过来的数据
        data = json.loads(request.body)
        image_data = data.get('image_data')
        resolution = data.get('resolution') / 100
        # print("image_data:", image_data)
        base64_str = image_data.split(',')[1]
        # print('len(base64_str):', len(base64_str))
        # print("resolution:", resolution)

        padding = 4 - (len(image_data) % 4)
        if padding:
            image_data += '=' * padding
        image_bytes = base64.b64decode(base64_str)
        # image_bytes = D_BASE64(image_data)

        # tmp_file = os.path.join('~/Document', 'tmp_image.png')
        # with open(tmp_file, 'rb') as f:
        #     f.write(image_bytes)

        with NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_file.flush()
            file_path = temp_file.name
            # print("temp_file_name: ", temp_file.name)
        count = predict_img(file_path, resolution)
        ABS_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        pre_img_path = os.path.join(ABS_ROOT, 'segProgram/local/pre_img.png')
        with open(pre_img_path, 'rb') as f:
            pre_img_data = base64.b64encode(f.read()).decode()

        # 构造响应数据
        response_data = {
            'count': count,
            'pre_img': pre_img_data
        }
        #  print('response_data:', response_data)
        response = HttpResponse(json.dumps(response_data))
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    else:
        # 处理不支持的请求方法
        response_data = {'message': '不支持的请求方法'}
        return JsonResponse(response_data, status=405)
