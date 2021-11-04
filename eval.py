from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from taki_ai.image_edit import image_convert  # noqa
from taki_convnet import DeepConvNet  # noqa

network = DeepConvNet()


def eval(image_path="", image_channel=1, image_size=28, convert=False):
    # print("""not taki -> [[1, 0]]\ntaki     -> [[0, 1]]""")
    if convert:
        image = image_convert(image_path, save=False,
                              height=image_size, width=image_size, mono=False)
    else:
        image = np.empty([1, image_channel, image_size, image_size])
        im = np.array(Image.open(image_path))

        if image_channel == 1:
            image[0][0] = im
        elif image_channel == 3:
            im = im.transpose(2, 0, 1)
            image[0] = im

    if image_channel == 1:
        network.load_params("taki_params.pkl")
    elif image_channel == 3:
        network.load_params("deep_convnet_params8.pkl")

    a = network.output(image)
    print("""result   ->""", a)
    return a


if __name__ == "__main__":

    eval("./test_data/taki.jpg", image_channel=3, convert=True)

    eval("./test_data/yonekura.jpg", image_channel=3, convert=True)

    eval("./test_data/kurihara.jpg", image_channel=3, convert=True)

    eval("./test_data/yoshinari.jpg", image_channel=3, convert=True)

    eval("./test_data/hara.jpg", image_channel=3, convert=True)
