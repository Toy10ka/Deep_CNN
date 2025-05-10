#deep_CNNの学習
#--------------------------------------------------
import time
import datetime
import sys, os 
sys.path.append(os.pardir) 
from common.mnist import load_mnist
#自作
from models.deep_CNN import DeepConvNet 
from common.trainer import Trainer 
#--------------------------------------------------
#データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

#インスタンス作成
network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer="Adam", optimizer_param={"lr":0.001},
                  evaluate_sample_num_per_epoch=1000)

#学習(+時間計測)
start_time = time.time()
trainer.train()
elapsed = time.time() - start_time
# 経過秒を h:m:s に変換
duration = datetime.timedelta(seconds=int(elapsed))
print(f"学習にかかった時間は {duration} (hh:mm:ss) ．")

#パラメータ保存
network.save_params("deep_convnet_params.pkl")
print("学習済みパラメータを deep_convnet_params.pkl に保存しました.")
