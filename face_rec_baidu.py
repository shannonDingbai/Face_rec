# encoding:utf-8
# from aip import AipFace
import requests
import base64
import os
import json
# 设置你的百度云API密钥和密钥
API_KEY = 'HIaGN6i2LnjjHdTGiZcSfq8v'
API_SECRET = 'BPkwCf4pv3NyXNCRrTsxaWugYUrjL99N'

HEADERS = {'content-type': 'application/json'}  # 请求头
null = ""

def get_access_token():
    '''获取access_token'''
    request_url = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_KEY}&client_secret={API_SECRET}'
    response = requests.get(url=request_url, headers=HEADERS)
    content = response.content
    access_token = json.loads(content)["access_token"]
    return access_token


def binary_pictures(img_path):
    '''二进制读取图片'''
    with open(img_path, 'rb') as f:
        imageB = base64.b64encode(f.read())
    image = str(imageB, 'utf-8')
    return image


def request_to_retrieve_api(image, access_token):
    '''请求检索api'''
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/search"
    params = {"image_type": "BASE64", "group_id_list": "img_face,pj_face,myj_face", "quality_control": "LOW"}
    params["image"] = image
    request_url = request_url + "?access_token=" + access_token
    response = requests.post(url=request_url, data=params, headers=HEADERS)
    content = eval(response.text)
    return content


if __name__ == '__main__':
    # 获取access_token
    access_token = get_access_token()
    # 二进制打开索要搜索的图片
    img_file=r'F:\face_data\hq_pj_face'
    cou=0
    for img in os.listdir(img_file):
        img_path=os.path.join(img_file,img)
        image = binary_pictures(img_path)
        # 请求
        respnse_content = request_to_retrieve_api(image, access_token)
        print(respnse_content)
        if respnse_content['error_msg'] == 'SUCCESS':
            cou+=1
    print(cou)
        # 提取user_id
        # if respnse_content:
        #     user_id = respnse_content['result']['user_list'][0]['user_id']
        #     print(user_id)






