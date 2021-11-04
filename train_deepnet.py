# coding: utf-8

# 学習がうまくいかないのはNetworkの重みの初期値のせいっぽい

import numpy as np
import sys
import os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from taki_ai.image_edit import load_dataset  # noqa
from taki_ai.taki_convnet import DeepConvNet  # noqa
from taki_ai.trainer import Trainer  # noqa

(x_train, t_train), (x_test, t_test) = load_dataset(
    a_path="./resize28", b_path="./taki28", image_channel=3)

result_list = []
for i in range(10):
    network = DeepConvNet()
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=4, mini_batch_size=50,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=100)
    tmp = trainer.train()
    result_list.append(tmp)

    # パラメータの保存
    network.save_params("deep_convnet_params" + str(i) + ".pkl")
    print("Saved Network Parameters!")

print(result_list)
