#深めのCNNを作る
#--------------------------------------------------
import numpy as np
import sys, os 
import pickle
from collections import OrderedDict
#自作モジュール
sys.path.append(os.pardir) 
from common.layers import Relu, Affine, SoftmaxWithLoss, \
         BatchNormalization, Dropout, Convolution, Pooling
#--------------------------------------------------
class DeepConvNet:
    """
    ネットワーク構成
        conv - BatchNorm - relu - conv - BatchNorm - relu - pool -
        conv - BatchNorm - relu - conv - BatchNorm - relu - pool -
        conv - BatchNorm - relu - conv - BatchNorm - relu - pool -
        affine - BatchNorm - relu - dropout
        affine - dropout - softmax
    """
    
    def __init__(self, input_dim=(1,28,28),#mnistサイズ
                 conv_params = [
                      {"filter_num":16, "filter_size":3, "pad":1, "stride":1},
                      {"filter_num":16, "filter_size":3, "pad":1, "stride":1},
                      {"filter_num":32, "filter_size":3, "pad":1, "stride":1},
                      {"filter_num":32, "filter_size":3, "pad":2, "stride":1}, #pad=2: pooling入力を偶数に
                      {"filter_num":64, "filter_size":3, "pad":1, "stride":1},
                      {"filter_num":64, "filter_size":3, "pad":1, "stride":1}],
                 hidden_size=50, output_size=10):

                #--------------
                 #W-paramsのhe係数を計算
                 he_scales = []
                 prev_channels = input_dim[0]
                 #Convレイヤのhe係数
                 for p in conv_params:
                    #1つの出力pixelに寄与した前層pixel数->フィルタpixel数*チャンネル
                    fan_in = prev_channels * p['filter_size'] * p['filter_size']
                    he_scales.append(np.sqrt(2.0 / fan_in))
                    prev_channels = p['filter_num']
                 #Affineレイヤのhe係数: fan-inは入力サイズ（フィルタ関係ない）
                 he_scales.extend([np.sqrt(2.0 / (64*4*4)), np.sqrt(2.0 / hidden_size)])

                 #--------------
                 #params初期化
                 self.params ={}
                 pre_channel_num = input_dim[0]
                 #Convパラ初期化：辞書からループで作成
                 for idx, conv_param in enumerate(conv_params):
                     self.params['W' + str(idx+1)] = he_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
                     self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])

                     #BatchNorm用:activation層一つにつきγ，β一つ
                     self.params['gamma' + str(idx+1)] = np.ones(conv_param['filter_num']) 
                     self.params['beta' + str(idx+1)] = np.zeros(conv_param['filter_num'])

                    #次のConvのC＝今のFN
                     pre_channel_num = conv_param['filter_num']

                 #Affineパラ初期化
                 self.params['W7'] = he_scales[6] * np.random.randn(64*4*4, hidden_size) 
                 self.params['b7'] = np.zeros(hidden_size)
                 self.params['gamma7'] = np.ones(hidden_size) 
                 self.params['beta7'] = np.zeros(hidden_size) 
                 self.params['W8'] = he_scales[7] * np.random.randn(hidden_size, output_size)
                 self.params['b8'] = np.zeros(output_size)

                 #--------------
                 #レイヤorder辞書作成
                 self.layers = OrderedDict()

                 #(Conv->Batch->ReLU->Conv->Batch->ReLU->Pool)*3
                 conv_count = 0
                 for idx, p in enumerate(conv_params, start=1):
                       #Conv
                       self.layers[f"Conv{idx}"] = Convolution(self.params[f"W{idx}"], self.params[f"b{idx}"],stride=p["stride"], pad=p["pad"])
                       #BatchNorm
                       self.layers[f"BatchNorm{idx}"] = BatchNormalization(self.params[f"gamma{idx}"], self.params[f"beta{idx}"])
                       #ReLU
                       self.layers[f"ReLU{idx}"] = Relu()
        
                       conv_count += 1
                       #Pooilng
                       if conv_count % 2 == 0:
                             pool_idx = conv_count // 2
                             self.layers[f"Pooling{pool_idx}"] = Pooling(pool_h=2, pool_w=2, stride=2)

                 #Affine->Batch->ReLU->Dropout
                 self.layers["Affine1"] = Affine(self.params["W7"], self.params["b7"])
                 self.layers["BatchNorm7"] = BatchNormalization(self.params["gamma7"], self.params["beta7"])
                 self.layers["ReLU7"] = Relu()
                 self.layers["Drop1"] = Dropout()

                 #Affine->Dropout
                 self.layers["Affine2"] = Affine(self.params["W8"], self.params["b8"])
                 self.layers["Drop2"] = Dropout()

                 #Softmaxwithloss
                 self.last_layer = SoftmaxWithLoss()

#--------------
    #推論
    def predict(self, x, train_flg=False):
        #forward (Dropならtrainflg渡す)
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x
    
#--------------
    #損失計算
    def loss(self, x, t):
          #推論
          y = self.predict(x, train_flg=True)
          return self.last_layer.forward(y, t)
    
#--------------
    #勾配計算
    def gradient(self, x, t):
          #損失計算
          self.loss(x, t)
          
          #backward
          dout = 1
          dout = self.last_layer.backward(dout)
          #レイヤをリスト化→逆順に(orderdictはそのままreverse()使えない)
          layers = list(self.layers.values())
          layers.reverse()

          for layer in layers:
              dout = layer.backward(dout)

          #勾配格納
          grads = {}
          for pname in self.params.keys():
              # W*, b*, gamma*, beta* のいずれか
              if pname.startswith('W'):
                  idx = int(pname[1:])  # 'W3' → '3'
                  if idx <= 6:
                      # Conv1～Conv6
                      target_layer = self.layers[f'Conv{idx}']
                  else:
                      # W7→Affine1, W8→Affine2
                      aff_i = idx - 6             # 7→1, 8→2 になる
                      target_layer = self.layers[f'Affine{aff_i}']  
                  grads[pname] = target_layer.dW

              elif pname.startswith('b') and pname[1].isdigit(): #betaも拾うから
                  idx = int(pname[1:])
                  if idx <= 6:
                      target_layer = self.layers[f'Conv{idx}']
                  else:
                      aff_i = idx - 6            
                      target_layer = self.layers[f'Affine{aff_i}']
                  grads[pname] = target_layer.db

              elif pname.startswith('gamma'):
                  idx = int(pname[5:])            # 'gamma3' → '3'
                  layer = self.layers[f'BatchNorm{idx}']
                  grads[pname] = layer.dgamma

              elif pname.startswith('beta'):
                  idx = int(pname[4:])            # 'beta3' → '3'
                  layer = self.layers[f'BatchNorm{idx}']
                  grads[pname] = layer.dbeta

              else:
                  raise KeyError(f"Unknown param: {pname}")
              
          return grads
    
#--------------
    #精度
    def accuracy(self, x, t, batch_size=100):
        # (N, C) のような one-hot 表現ならnp.argmax で各サンプルの正解クラスidxを取り出す
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]
    
#--------------------------------------------------
    #パラメータ保存，読み込み関数
    def save_params(self, file_name="params.pkl"):
        """self.params 中の全パラメータを file_name に pickle で保存"""
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"[Saved] {file_name}")

    def load_params(self, file_name="params.pkl"):
        """file_name からパラメータを読み込み、self.params と各レイヤに反映"""
        # 1) ファイルから辞書を復元
        with open(file_name, 'rb') as f:
            loaded_params = pickle.load(f)

        # 2) self.params を更新
        for key, val in loaded_params.items():
            self.params[key] = val

        # 3) 各レイヤにパラメータを流し込む
        for name, layer in self.layers.items():
            # Convolution レイヤ
            if isinstance(layer, Convolution):
                idx = name.replace("Conv", "")
                layer.W = self.params[f"W{idx}"]
                layer.b = self.params[f"b{idx}"]
            # BatchNormalization レイヤ
            elif isinstance(layer, BatchNormalization):
                idx = name.replace("BatchNorm", "")
                layer.gamma = self.params[f"gamma{idx}"]
                layer.beta  = self.params[f"beta{idx}"]
            # Affine レイヤ
            elif isinstance(layer, Affine):
                if name == "Affine1":
                    layer.W = self.params["W7"]
                    layer.b = self.params["b7"]
                elif name == "Affine2":
                    layer.W = self.params["W8"]
                    layer.b = self.params["b8"]
            # Dropout/ReLU/Pooling はパラメータなしなのでスルー

        print(f"[Loaded] {file_name}")

                
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 

