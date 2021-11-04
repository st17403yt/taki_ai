
from PIL import Image
import numpy as np
import cv2
import sys
import os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from taki_ai.taki_convnet import DeepConvNet  # noqa


def image_convert(image, save=False, save_path="./cut", height=28, width=28, mono=False):
    # カラー画像の入力を想定
    # モノクロ画像 (height * width) に変換
    image_channel = 1
    if not mono:
        image_channel = 3

    rgb2xyz_rec709 = (
        0.412453, 0.357580, 0.180423, 0,
        0.212671, 0.715160, 0.072169, 0,  # RGB mixing weight
        0.019334, 0.119193, 0.950227, 0)

    gamma22LUT = [pow(x/255.0, 2.2)*255 for x in range(256)] * 3
    gamma045LUT = [pow(x/255.0, 1.0/2.2)*255 for x in range(256)]

    # image = cv2.imread(image_name)

    cascade_file = 'haarcascade_frontalface_alt2.xml'
    cascade_face = cv2.CascadeClassifier(cascade_file)

    face_list = cascade_face.detectMultiScale(image, minSize=(20, 20))
    result_images = np.empty([len(face_list), image_channel, height, width])
    for i, (x, y, w, h) in enumerate(face_list):
        trim = image[y: y+h, x:x+w]
        trim_pil = Image.fromarray(cv2.cvtColor(trim, cv2.COLOR_BGR2RGB))
        img = trim_pil.resize((width, height))

        if mono:
            im_rgb = img.convert("RGB")
            im_rgbL = im_rgb.point(gamma22LUT)
            im_grayL = im_rgbL.convert("L", rgb2xyz_rec709)
            img = im_grayL.point(gamma045LUT)

        if save:
            img.save(save_path + "/" + str(i) + ".jpg")

        if mono:
            result_images[i][0] = img
        else:
            img = np.array(img).transpose(2, 0, 1)
            result_images[i] = img

    return result_images


network = DeepConvNet()
network.load_params("deep_convnet_params8.pkl")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while(True):
    ret, frame = cap.read()

    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    faces = face_cascade.detectMultiScale(frame, minSize=(20, 20))

    color = [255, 0, 0]
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        result = image_convert(frame)
        if result.shape[0] != 0:
            a = network.output(result)
            if a[0][1] > 0.85:
                frame = cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, text=str(a[0][1]), org=(x, y),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1) & 0xFF  # キー操作取得。64ビットマシンの場合,& 0xFFが必要
    prop_val = cv2.getWindowProperty(
        "frame", cv2.WND_PROP_ASPECT_RATIO)  # ウィンドウが閉じられたかを検知する用

# qが押されるか、ウィンドウが閉じられたら終了
    if k == ord("q") or (prop_val < 0):

        break

cap.release()
cv2.destroyAllWindows()
