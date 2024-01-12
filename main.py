# coding=utf-8

import os
import cv2
import streamlit as st
import base64
import requests
import json
import io
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

save_path = r'F:\identify_demo'
server = 'http://192.168.100.85:5000'
# server = 'http://192.168.0.5:5000'
outres = []

no_use_baidu=True
API_KEY = 'Ara6QdKSrskSyPl6DS5Esvjh'
API_SECRET = 'vqgN776TbvcDkCxCTmqHHNYvYycHoVLH'

HEADERS = {'content-type': 'application/json'}  # 请求头
null = ""

def extract_frames_and_convert_to_base64(video_path, frame_interval=2):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    base64_frames = []

    frame_rate = cap.get(5)  # 视频帧率
    frame_interval_frames = int(frame_rate * frame_interval)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval_frames == 0:
            # 将OpenCV帧转换为Pillow图像
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 将Pillow图像转换为Base64编码
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            base64_frames.append('data:image/png;base64,'+base64_image)

        frame_count += 1

    cap.release()

    return base64_frames

# 示例用法
# video_path = r"F:\identify_demo\视频测试用例\3.mp4"
# base64_frames = extract_frames_and_convert_to_base64(video_path, frame_interval=2)
#
# for base64_frame in base64_frames:
#     print(base64_frame)

def extract_frame_from_mp4(mp4_path,name, output_dir):
    frame_index=200
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 打开 MP4 文件
    cap = cv2.VideoCapture(mp4_path)

    # 检查是否成功打开文件
    if not cap.isOpened():
        return None

    # 获取视频的帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 检查帧索引是否有效
    if frame_index < 0 or frame_index >= total_frames:
        return None

    # 设置要读取的帧的索引
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # 读取指定帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        return None

    # 生成输出文件名
    output_filename = f"{name}.jpg"
    output_path = os.path.join(output_dir, output_filename)

    # 保存帧为 JPEG 图像
    cv2.imwrite(output_path, frame)

    # 释放资源
    cap.release()

    return output_path


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
    params = {"image_type": "BASE64", "group_id_list": "pj_face,pujing,myj", "quality_control": "LOW"}
    params["image"] = image
    request_url = request_url + "?access_token=" + access_token
    response = requests.post(url=request_url, data=params, headers=HEADERS)
    content = eval(response.text)
    return content


def cv2_base64(img):
    binary_str = cv2.imencode('.jpg', img)[1].tobytes()
    base64_str = base64.b64encode(binary_str) 
    base64_str = base64_str.decode('utf-8')
    return 'data:image/png;base64,' + base64_str


def find(image):
    url = f'{server}/face/find'
    base64Img = cv2_base64(image)
    body = {
        'base64Img': base64Img 
    }
    res = requests.post(url, json=body)
    return res


def find_video(video_base64):
    url = f'{server}/face/find'
    body = {
        'base64Img': video_base64
    }
    res = requests.post(url, json=body)
    return res



def add(imagebase,name):
    url = f'{server}/face/add'
    base64img = cv2_base64(imagebase)
    body = {
        'name': name,
        'base64Img': base64img
    }
    resbase = requests.post(url,json=body)
    print(resbase.content)
    return resbase
# imagebase = cv2.imread(r"F:\identify_demo\sbn.jpg")
# add(imagebase,'撒贝宁')

def find_scores(data,target_face_name):
    target_score = None
    min_score = float('inf')
    min_score_face_name = None

    for entry in data:
        if entry.get("face_name") == target_face_name:
            target_score = entry.get("score")

        if "score" in entry and entry["score"] < min_score:
            min_score = entry["score"]
            min_score_face_name = entry["face_name"]

    result = [ str(target_face_name)+":"+str(target_score), str(min_score_face_name)+":"+str(min_score)]

    return result


def process_data(data):
    result = {}  # 用于存储整理后的数据

    for entry in data:
        if isinstance(entry, list):
            for item in entry:
                if isinstance(item, dict) and 'face_name' in item:
                    face_name = item['face_name']
                    score = item.get('score')

                    if face_name in result:
                        result[face_name]['count'] += 1
                        if score is not None and (
                                result[face_name]['min_score'] is None or score < result[face_name]['min_score']):
                            result[face_name]['min_score'] = score
                    else:
                        result[face_name] = {'count': 1, 'min_score': score}

    return result

def find_min_score_except_unknown(result):
    min_score = float('inf')  # 初始化一个无穷大的值，用于存储最小的 score
    min_score_facename = None  # 初始化一个变量，用于存储对应的 facename

    for facename, data in result.items():
        if facename != 'Unknown' and data['min_score'] is not None and data['min_score'] < min_score:
            min_score = data['min_score']
            min_score_facename = facename

    return min_score_facename

def find_max_count_and_min_score_except_unknown(result):
    max_count = 0  # 初始化一个值为 0，用于存储最大的 count
    max_count_facename = None  # 初始化一个变量，用于存储最大 count 对应的 facename
    min_score = float('inf')  # 初始化一个无穷大的值，用于存储最小的 score
    min_score_facename = None  # 初始化一个变量，用于存储最小 score 对应的 facename

    for facename, data in result.items():
        if facename != 'Unknown':
            if data['count'] > max_count:
                max_count = data['count']
                max_count_facename = facename
                min_score = data['min_score']  # 更新最小 score
                min_score_facename = facename
            elif data['count'] == max_count and data['min_score'] is not None and data['min_score'] < min_score:
                min_score = data['min_score']
                min_score_facename = facename

    return min_score_facename

def maincode():
    images = []
    images_name =[]

    # # 打开并显示多张图片
    st.set_page_config(page_title="和信科技人脸识别demo", page_icon=None, layout="wide", initial_sidebar_state="expanded")

    uploaded_files = st.file_uploader("请选择需要识别的视频文件", accept_multiple_files=True)

    # 示例数据
    labels = '撒贝宁', '普京', '未知'
    sizes = [0, 0, 0]
    colors = ['yellowgreen', 'lightskyblue', 'lightcoral']
    textprops = {'color': 'white','size': 'larger'}

    image_placeholder_text = st.empty()

    # 创建一个空的占位符
    image_placeholder = st.empty()
    image_placeholder_charttext = st.empty()

    image_placeholder_chart = st.empty()


    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            if uploaded_file.type in ['image/jpeg', 'image/png']:
                img_path = os.path.join(save_path, uploaded_file.name)

                with open(img_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                    print(img_path)

                if no_use_baidu:

                    image = cv2.imread(img_path)
                    res = find(image)
                    print(image)
                    # 初始化标志变量为True
                    all_values_not_matched = True
                    if res.status_code == 200:
                        res_json = json.loads(res.content)
                        res_json_score = find_scores(res_json,'马英九')

                        face_names = [entry["face_name"] for entry in res_json if "face_name" in entry]

                        for i in face_names:
                            if i == '普京' :
                                images.append(Image.open(img_path))
                                images_name.append(uploaded_file.name)

                                sizes[1]=sizes[1]+1
                                image_placeholder_text.write(f"""
                                    <p style="font-size: 12px;">识别目标人物(马英九、普京)照片结果如下({sizes[0] + sizes[1]}/{len(uploaded_files)})：</p>
                                """, unsafe_allow_html=True)
                                image_placeholder.image(images, caption=images_name)
                                all_values_not_matched = False
                                break
                            if i == '马英九' :
                                images.append(Image.open(img_path))
                                images_name.append(uploaded_file.name)


                                sizes[0] = sizes[0] + 1

                                image_placeholder_text.write(f"""
                                        <p style="font-size: 12px;">识别目标人物(马英九、普京)照片结果如下({sizes[0]+sizes[1]}/{len(uploaded_files)})：</p>
                                    """, unsafe_allow_html=True)
                                image_placeholder.image(images ,caption=images_name)
                                all_values_not_matched = False
                                break
                        # 如果所有值都不等于'普京'或'马英九'，则递增全局变量
                        if all_values_not_matched:
                            sizes[2] += 1
                        # st.write(
                        #     f"图片{uploaded_file.name}识别目标和score值为: {res_json_score[0]},最小值为:{res_json_score[1]},两者{res_json_score[0]==res_json_score[1]}")

                    else:
                        st.write('服务器错误')
                        st.write(res.content)
                else:
                    access_token = get_access_token()
                    image = binary_pictures(img_path)
                    all_values_not_matched = True
                    # 请求

                    respnse_content = request_to_retrieve_api(image, access_token)
                    print(respnse_content)
                    if respnse_content['error_msg'] == 'SUCCESS':
                        # if respnse_content['result']['user_list'][0]['score']<40:
                        #     continue
                        images.append(Image.open(img_path))
                        images_name.append(uploaded_file.name)
                        all_values_not_matched = False
                        if respnse_content['result']['user_list'][0]['user_id'] == 'myj':
                            sizes[0] += 1
                        else:
                            sizes[1] += 1
                        image_placeholder_text.write(f"""
                           <p style="font-size: 12px;">识别目标人物(马英九、普京)照片结果如下({sizes[0] + sizes[1]}/{len(uploaded_files)})：</p>
                       """, unsafe_allow_html=True)
                        image_placeholder.image(images, caption=images_name)

                    if all_values_not_matched:
                        sizes[2]+=1

            elif uploaded_file.type == 'video/mp4':
                video_path = os.path.join('F:\视频截图', uploaded_file.name)
                index=0
                with open(video_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                    print(video_path)
                data=[]
                imges_video=extract_frames_and_convert_to_base64(video_path)
                all_values_not_matched=True
                for item in imges_video:
                    # jumpNext=False
                    res_video = find_video(item)
                    if res_video.status_code == 200:
                        res_json = json.loads(res_video.content)
                        # print(res_json)
                        data.append(res_json)

                        # video_face_names = [entry["face_name"] for entry in res_json if "face_name" in entry]
                        # for i in video_face_names:
                        #     if i == '撒贝宁':
                        #         images.append(
                        #             Image.open(extract_frame_from_mp4(video_path, uploaded_file.name.split('.')[0], "F:\\identify_demo\\video_jpg")))
                        #         images_name.append(uploaded_file.name)
                        #         index+=1
                        #         sizes[0]=sizes[0]+1
                        #         image_placeholder_text.write(f"""
                        #             <p style="font-size: 12px;">识别目标人物(撒贝宁、普京)照片结果如下({sizes[0] + sizes[1]}/{len(uploaded_files)})：</p>
                        #         """, unsafe_allow_html=True)
                        #         image_placeholder.image(images ,caption=images_name)
                        #         jumpNext=True
                        #         all_values_not_matched=False
                        #         break
                        #     if i == '普京':
                        #         images.append(
                        #             Image.open(extract_frame_from_mp4(video_path, uploaded_file.name.split('.')[0],
                        #                                               "F:\\identify_demo\\video_jpg")))
                        #         images_name.append(uploaded_file.name)
                        #         index += 1
                        #         sizes[1] = sizes[1] + 1
                        #         image_placeholder_text.write(f"""
                        #             <p style="font-size: 20px;">识别目标人物(撒贝宁、普京)照片结果如下({sizes[0] + sizes[1]}/{len(uploaded_files)})：</p>
                        #         """, unsafe_allow_html=True)
                        #         image_placeholder.image(images ,caption=images_name)
                        #         jumpNext = True
                        #         all_values_not_matched = False
                        #         break
                        # if jumpNext:
                        #     break

                    else:
                        print(res_video.content)

                print(process_data(data))
                # print(data)
                res_map=process_data(data)
                if find_max_count_and_min_score_except_unknown(res_map)=='撒贝宁':
                    images.append(
                                Image.open(extract_frame_from_mp4(video_path, uploaded_file.name.split('.')[0],
                                                                  "F:\\identify_demo\\video_jpg")))
                    images_name.append(uploaded_file.name)
                    index += 1
                    sizes[0] = sizes[0] + 1
                    image_placeholder_text.write(f"""
                        <p style="font-size: 20px;">识别目标人物(撒贝宁、普京)照片结果如下({sizes[0] + sizes[1]}/{len(uploaded_files)})：</p>
                    """, unsafe_allow_html=True)
                    image_placeholder.image(images ,caption=images_name)
                    # jumpNext = True
                    all_values_not_matched = False
                if find_max_count_and_min_score_except_unknown(res_map)=='普京':
                    images.append(
                                Image.open(extract_frame_from_mp4(video_path, uploaded_file.name.split('.')[0],
                                                                  "F:\\identify_demo\\video_jpg")))
                    images_name.append(uploaded_file.name)
                    index += 1
                    sizes[1] = sizes[1] + 1
                    image_placeholder_text.write(f"""
                        <p style="font-size: 20px;">识别目标人物(撒贝宁、普京)照片结果如下({sizes[0] + sizes[1]}/{len(uploaded_files)})：</p>
                    """, unsafe_allow_html=True)
                    image_placeholder.image(images ,caption=images_name)
                    # jumpNext = True
                    all_values_not_matched = False


                if all_values_not_matched:
                    sizes[2]+=1
            else:
                st.write('图片格式不正确，请重试')

        image_placeholder_charttext.write(f'视频文件人脸识别准确率：{round((sizes[0]+sizes[1])*100/(sizes[0]+sizes[1]+sizes[2]),1)}%')
        # 绘制饼图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,colors=colors,textprops=textprops)
        ax.axis('equal')  # 使饼图保持圆形
        fig.patch.set_facecolor('none')
        # 显示饼图
        image_placeholder_chart.pyplot(fig)

    # 使用HTML和CSS自定义图像位置和间距
    st.markdown(
        """
        <style>
        #root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(3) > div > div{
            width:100%;
            overflow-height:auto;
            border-radius:5px;
            background-color:rgb(38, 39, 48);
        }
        .css-1kyxreq > * {
            padding:5px;
        }
        img{
        height:80px;
        }
        #root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(5) > div > div > div > img{
            width:330px !important;
            height:250px;
            border-radius:5px;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )



maincode()


