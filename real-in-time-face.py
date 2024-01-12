''''
1、读取视频流
2、模型识别每一帧照片，搜寻与其差距最小的人脸，返回其姓名及相似度
3、在人脸上画框
'''
import os
from PIL import Image
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from nets.facenet import Facenet as facenet
import dlib
from flask import Flask, request, make_response
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
#对传入的图片进行预处理
def delete_files(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0
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
    filePath='img'
    with torch.no_grad():
        all_l1s=[]
        fileList=os.listdir(filePath)
        img = letterBox(img1, [160, 160])
        for item in fileList:
            # print(item,'oooo')
            item_path=os.path.join(filePath,item)
            item_img=cv2.imread(item_path)
            img2=letterBox(item_img, [160, 160])
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(np.asarray(img).astype(np.float64) / 255, (2, 0, 1)), 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(np.asarray(img2).astype(np.float64) / 255, (2, 0, 1)), 0)).type(torch.FloatTensor)

            if 'cuda'==True:
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
            print(l1,item)
            all_l1s.append(l1)
        min_l1=min(all_l1s)
        if min_l1<1.2:
            min_index = all_l1s.index(min_l1)
            min_img_name = fileList[min_index].split('.')[0]
        else:
            min_img_name='unknown'

    return min_img_name
def get_face(img,detector):
    # Dlib 预测器
    path_save='face_save'
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    # Dlib 检测

    faces = detector(img, 1)
    person_name=''
    print('人脸数：', len(faces), '\n')
    if len(faces)!=0:
        for k, d in enumerate(faces):
            # 计算矩形大小
            # (x,y),(宽度width,高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

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
def draw_box(img,name,detector):
    face=detector(img)
    for k, d in enumerate(face):
        cv2.rectangle(img, tuple([d.left()+5, d.top()+5]), tuple([d.right(), d.bottom()]),
                      (255, 255, 255), 2)
        cv2.putText(img, name, tuple([d.left(), d.top() + 8]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2)
        cv2.imshow('camera', img)
        #判断文件夹是否为空，为空时写入文件
        cv2.resize(img,(180,180))
        img_save_path="E:/Ding's_project/face_recognition_system/public/img_save/face.jpg"
        if os.path.exists(img_save_path):
            os.remove(img_save_path)
            print("文件已删除")
            cv2.imwrite(img_save_path,img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        return img_save_path
def binary_to_video(binary_data,path):
    with open(path,'wb')as f:
        f.write(binary_data)
    f.close()
@app.route('/face_rec', methods=['POST'])
def read_video():
    # data=request.json
    # print(request, '......')
    video=request.data
    # print(video,'......')
    video_path='save_video/video.mp4'
    binary_to_video(video,video_path)
    cap=cv2.VideoCapture(video_path)
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    det = dlib.get_frontal_face_detector()
    person_name=''
    count=0
    while cap.isOpened():
        #读取每一帧图片
        ret,img=cap.read()
        if not ret:
            break
        if(count % fps ==0):
            # 搜寻相似度最高的人脸
            person_name = get_face(img, det)
            # 画框
            face_img=draw_box(img, person_name, det)
            print(face_img)
        count+=1

    cap.release()
    cv2.destroyAllWindows()

    return face_img
#处理图片文件
def img_deal():
    imgfile_path='img'
    all_img=[]
    for imgItem in os.listdir(imgfile_path):
        pre_img=letterBox(imgItem)
        all_img.append(pre_img)
    return all_img

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
