import cv2
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle


def image_convert(image_name, save=True, save_path="./cut", height=28, width=28, mono=True):
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


def image_clip(image_list, save=True):
    print("---------------image_clip---------------")
    counter = 0
    for image_name in image_list:
        image = cv2.imread(image_name)

        cascade_file = 'haarcascade_frontalface_alt2.xml'
        cascade_face = cv2.CascadeClassifier(cascade_file)

        # 顔を探して配列で返す
        face_list = cascade_face.detectMultiScale(image, minSize=(20, 20))
        for i, (x, y, w, h) in enumerate(face_list):
            trim = image[y: y+h, x:x+w]
            if save:
                cv2.imwrite('./cut/cut' + str(counter) + '.jpg', trim)
            counter = counter + 1
    print("-------------image_clip end-------------")


def image_resize(image_list, save=True):
    print("---------------image_resize---------------")
    for i in range(len(image_list)):
        im = Image.open(image_list[i])
        img = im.resize((28, 28))
        if save:
            img.save("./taki28/taki_resize" + str(i) + ".jpg")
        print("resizing :", i)
    print("-------------image_resize end-------------")


def image_load(image_path, image_channel, image_height, image_width):
    print("---------------image_load---------------")
    image_list = glob.glob(image_path + "/*.jpg")
    images = np.empty([len(image_list), image_channel,
                      image_height, image_width])

    for i, j in enumerate(tqdm(range(len(image_list)))):
        im = np.array(Image.open(image_list[i]))
        if image_channel == 1:
            images[i][0] = im
        elif image_channel == 3:
            im = im.transpose(2, 0, 1)
            images[i] = im

    print("images.shape = ", images.shape)
    print("-------------image_load end-------------")

    return images


def load_dataset(a_path="./resize28", b_path="./taki28", image_channel=3, height=28, width=28):
    print("--------------dataset_load--------------")
    data_num = len(glob.glob(a_path + "/*.jpg"))
    train_num = int(data_num*0.8)
    taki_num = len(glob.glob(b_path + "/*.jpg"))
    taki_train_num = int(taki_num*0.8)
    not_taki_data = image_load(
        a_path, image_channel=image_channel, image_height=height, image_width=width)
    train_data = not_taki_data[:train_num]
    test_data = not_taki_data[train_num:]

    taki_data = image_load(
        b_path, image_channel=image_channel, image_height=height, image_width=width)
    train_data = np.concatenate([train_data, taki_data[:taki_train_num]])
    test_data = np.concatenate([test_data, taki_data[taki_train_num:]])

    train_label = np.concatenate(
        [np.zeros(train_num, dtype=np.int), np.ones(taki_train_num, dtype=np.int)])
    test_label = np.concatenate(
        [np.zeros(data_num - train_num, dtype=np.int), np.ones(taki_num - taki_train_num, dtype=np.int)])

    train_data, train_label = shuffle_array(train_data, train_label)
    test_data, test_label = shuffle_array(test_data, test_label)

    print("------------dataset_load end------------")
    return (train_data, train_label), (test_data, test_label)


def shuffle_array(array1, array2):
    tmp1 = array1.copy()
    tmp2 = array2.copy()

    if tmp1.shape[0] != tmp2.shape[0]:
        print("[!] shape error")
        return

    shuffle_index = np.arange(tmp1.shape[0])
    shuffle_index = np.random.choice(
        shuffle_index, tmp1.shape[0], replace=False)

    for i, index in enumerate(shuffle_index):
        tmp1[i] = array1[index]
        tmp2[i] = array2[index]

    return tmp1, tmp2


def to_grayscale(image_path, out_path):
    print("--------------to_grayscale--------------")
    rgb2xyz_rec709 = (
        0.412453, 0.357580, 0.180423, 0,
        0.212671, 0.715160, 0.072169, 0,  # RGB mixing weight
        0.019334, 0.119193, 0.950227, 0)

    gamma22LUT = [pow(x/255.0, 2.2)*255 for x in range(256)] * 3
    gamma045LUT = [pow(x/255.0, 1.0/2.2)*255 for x in range(256)]

    image_list = glob.glob(image_path + "/*.jpg")

    for i in range(len(image_list)):
        im = Image.open(image_list[i])
        im_rgb = im.convert("RGB")  # any format to RGB
        im_rgbL = im_rgb.point(gamma22LUT)
        # RGB to L(grayscale BT.709)
        im_grayL = im_rgbL.convert("L", rgb2xyz_rec709)
        im_gray = im_grayL.point(gamma045LUT)
        # im_gray.save(outfile)
        im_gray.save(out_path + "/mono" + str(i) + ".jpg")

    print("------------to_grayscale end------------")


def test():
    train_data1 = [[[[1], [2]], [[1], [2]], [[1], [2]]],
                   [[[3], [4]], [[3], [4]], [[3], [4]]],
                   [[[5], [6]], [[5], [6]], [[5], [6]]]]
    train_data2 = [[[[9], [8]], [[9], [8]], [[9], [8]]],
                   [[[7], [6]], [[7], [6]], [[7], [6]]]]
    return np.concatenate([train_data1, train_data2])


if __name__ == "__main__":
    """
    image_list = ["./WIN_20211102_06_37_14_Pro.jpg"]
    a = image_convert("./WIN_20211102_06_37_14_Pro.jpg",
                      save=True, save_path="./", mono=False)
    print(a.shape)
    """
    a = 2
    if a == 0:
        (a, b), (c, d) = load_dataset()
        print(a.shape)
        print(b.shape)
        print(c.shape)
        print(d.shape)
    elif a == 1:
        b = image_load("./resize28", image_channel=3,
                       image_height=28, image_width=28)
        b = image_load("./taki28", image_channel=3,
                       image_height=28, image_width=28)
    elif a == 2:
        image = image_convert("./test_data/any.jpg", save=True,
                              save_path="./test_data/", mono=False)
        print(image.shape)
