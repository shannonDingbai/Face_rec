from flask import Flask, render_template, Response
import cv2
import dlib
from PIL import Image
app = Flask(__name__)
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from nets.facenet import Facenet as facenet
import os
class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture('test-video/liuyifei.mp4')
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))

    def __del__(self):
        self.video.release()

    def get_frame(self,cou):
        # self.video = cv2.VideoCapture(0)

        det = dlib.get_frontal_face_detector()
        person_name = ''
        # while self.video.isOpened():
        #     # 读取每一帧图片
        #     ret, img = self.video.read()
        #     if not ret:
        #         break
        #     if (count % fps == 0):
        #         # 搜寻相似度最高的人脸
        #         person_name = get_face(img, det)
        #         # 画框
        #         face_img = draw_box(img, person_name, det)
        #         print(face_img)
        #     count += 1
        ret, img = self.video.read()
        # 搜寻相似度最高的人脸
        person_name = get_face(img, det)
        # 画框
        face_img = draw_box(img, person_name, det)
        # success, image = self.video.read()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', face_img)

        return jpeg.tobytes()
def draw_box(img,name,detector):
    face=detector(img)
    for k, d in enumerate(face):
        cv2.rectangle(img, tuple([d.left()+5, d.top()+5]), tuple([d.right(), d.bottom()]),
                      (255, 255, 255), 2)
        cv2.putText(img, name, tuple([d.left(), d.top() + 8]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2)
        # cv2.imshow('camera', img)
        #判断文件夹是否为空，为空时写入文件
        cv2.resize(img,(180,180))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        return img
def letterBox(old_img,size):
    #将传入的img转为rgb格式
    # print(type(old_img))
    image= Image.fromarray(cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB))
    # image=old_img.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.Resampling.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    # if input_shape[-1]==1:
    #     new_image=new_image.convert("L")
    return new_image
def detect(img1):
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = facenet(backbone="mobilenet", mode="predict")
    model.load_state_dict(torch.load("model_data/facenet_mobilenet.pth", map_location=device), strict=False)
    net = model.eval()
    filePath = 'img'
    with torch.no_grad():
        all_l1s = []
        fileList = os.listdir(filePath)
        img = letterBox(img1, [160, 160])
        for item in fileList:
            # print(item,'oooo')
            item_path = os.path.join(filePath, item)
            item_img = cv2.imread(item_path)
            img2 = letterBox(item_img, [160, 160])
            photo_1 = torch.from_numpy(
                np.expand_dims(np.transpose(np.asarray(img).astype(np.float64) / 255, (2, 0, 1)), 0)).type(
                torch.FloatTensor)
            photo_2 = torch.from_numpy(
                np.expand_dims(np.transpose(np.asarray(img2).astype(np.float64) / 255, (2, 0, 1)), 0)).type(
                torch.FloatTensor)

            if 'cuda' == True:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            output1 = net(photo_1).cpu().numpy()
            output2 = net(photo_2).cpu().numpy()

            # ---------------------------------------------------#
            #   计算二者之间的距离
            # ---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)
            print(l1, item)
            all_l1s.append(l1)
        min_l1 = min(all_l1s)
        if min_l1 < 1.2:
            min_index = all_l1s.index(min_l1)
            min_img_name = fileList[min_index].split('.')[0]
        else:
            min_img_name = 'unknown'

    return min_img_name
def get_face(img, detector):
    # Dlib 预测器
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    # Dlib 检测
    faces = detector(img, 1)
    person_name = ''
    print('人脸数：', len(faces), '\n')
    if len(faces) != 0:
        for k, d in enumerate(faces):
            # 计算矩形大小
            # (x,y),(宽度width,高度height)
            # pos_start = tuple([d.left(), d.top()])
            # pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小
            height = d.bottom() - d.top()
            width = d.right() - d.left()

            # 根据人脸大小生成空的图象
            img_blank = np.zeros((height, width, 3), np.uint8)

            for i in range(height):
                for j in range(width):
                    img_blank[i][j] = img[d.top() + i][d.left() + j]
                # 存在本地
            person_name = detect(img_blank)
    return person_name


def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


def gen(camera):
    count=0
    while True:
        frame = camera.get_frame(count)
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
