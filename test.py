import cv2
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
import sys
import os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from taki_ai.image_edit import image_convert  # noqa
from taki_convnet import DeepConvNet  # noqa


def image_convert(image_name, save=True, save_path="./cut", height=28, width=28, mono=True):
    network = DeepConvNet()
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

    image = cv2.imread(image_name)

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
