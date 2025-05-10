#レイヤをまとめる
#--------------------------------------------------
import numpy as np
#自作
import common.function as function
#--------------------------------------------------
#乗算レイヤ
class MulLayer:
    #入出力の属性決め
    def __init__(self):
        self.x = None
        self.y = None
    #順伝播
    def forward(self, x, y):
        self.x = x #入力値を保持（sigmoidは出力を保持）
        self.y = y
        out = x * y #入力の積をとる
        return out 
    #逆伝播
    def backward(self, dout):#上流から渡されてくる勾配
        dx = dout * self.y #(出力/入力=もう一個の入力)
        dy = dout * self.x #(∂xy/∂y =x)
        return dx, dy
    
#----------------
#加算レイヤ
class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x + y
        return out
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    
#--------------------------------------------------
#NN用のレイヤ
#ReLUレイヤ
class Relu:
    #初期設定
    def __init__(self):
        self.mask = None

    #順伝播
    def forward(self, x): #x=Wx+b:(batch_size,出力ノード数)
        self.mask = (x <= 0) #xと同じ形状のbool
        out = x.copy() #逆伝播で使うのでcopy
        #マスクかぶせるだけ(dout=x)
        out[self.mask] = 0

        return out #->(batch_size,出力ノード数)
    
    #逆伝播
    def backward(self, dout): #dout:上流から渡されてくる勾配：(batch_size, output)
        #出力にマスク被せて，そのまま流すだけ(dx=dout*1)
        dout[self.mask] = 0
        dx =dout

        return dx #->(batch_size,出力ノード数)
#----------------
#Sigmoidレイヤ   
class Sigmoid:
    #初期設定
    def __init__(self):
        self.out = None

    #順伝播
    def forward(self, x): #x=Wx+b:(batch_size,出力ノード数)
        out = 1 / (1 + np.exp(-x))
        self.out = out #勾配を出力使って書くので保存

        return out #->(batch_size,出力ノード数)
    
    #逆伝播
    def backward(self, dout): #dout:上流から渡されてくる勾配(batch_size,出力ノード数)
        dx = dout * (self.out * (1 - self.out))

        return dx #->(batch_size,出力ノード数)

#--------------------------------------------------
#Affineレイヤ
class Affine:
    #生成時処理
    def __init__(self, W, b): #W:(input,output), b:(output)
        #init引数はインスタンス作成時に呼ぶ必要がある
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    #順伝播
    def forward(self, x): #x:(batch,input(D or C,H,W))
        #入力がテンソル
        if x.ndim > 2:
            self.original_shape = x.shape      # 後で戻すなら覚えておく
            x = x.reshape(x.shape[0], -1)      # → (N, C*H*W)
        self.x = x #backwordでつかう
        out = np.dot(x, self.W) + self.b 
        return out #->(output,)
    #逆伝播
    def backward(self, dout):
        dx = np.dot(dout, self.W.T) #->(batch, input):空気読んだ行列積を取ってくれるので，自動で内積に
        # 元の形に戻したいなら
        if hasattr(self, 'original_shape'):
            dx = dx.reshape(self.original_shape)
        self.dW = np.dot(self.x.T, dout) #->(input, output)
        self.db = dout.sum(axis = 0) #->(output)
        return dx #->(batch,input)
#--------------------------------------------------
#Softmax-with-Lossレイヤ
class SoftmaxWithLoss:
    #生成時処理
    def __init__(self):
        #逆伝播で使うふたつ
        self.y = None
        self.t = None
        self.loss = None 

    #順伝播
    def forward(self, x, t): #x:(batch, class), t:(batch, class)
        #softmax
        self.t = t
        self.y = function.softmax(x)
        #loss
        self.loss = function.cross_entropy_error(self.y, self.t)

        return self.loss #->スカラ（batch毎にひとつのloss）
        
    #逆伝播
    def backward(self, dout=1): #最終層だから∂L/∂L=1だね
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vector(batch,class)の場合：yと一致
            dx = (self.y - self.t) / batch_size #各batchに流す勾配値（同じ重みと仮定）
        else: #t:(batch,), y:(batch, class)
            dx = self.y.copy()
            #最大のとこだけ1(tの値)引く
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx #->(batch,class)
    
#--------------------------------------------------
#Batch Normレイヤ
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim == 4:  # Conv層の出力 (N,C,H,W) の場合
            N, C, H, W = x.shape
            # 1) (N,C,H,W)->(N,H,W,C)->(N*H*W,C)
            x_flat = x.transpose(0,2,3,1).reshape(-1, C)
            out_flat = self.__forward(x_flat, train_flg)
            # 3) (N*H*W,C)->(N,H,W,C)->(N,C,H,W)
            out = out_flat.reshape(N, H, W, C).transpose(0,3,1,2)
        else:
            out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
    #内部処理
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        # --- conv 層出力 (N,C,H,W) の場合は、
        #     forward でやったのと同じく (N,H,W,C) → (N*H*W, C) にしてから __backward
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            # 1) (N,C,H,W)->(N,H,W,C)
            # 2) (N,H,W,C)->(N*H*W, C)
            dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
            # 3) 一次元化された平たんデータで勾配を計算
            dx_flat = self.__backward(dout_flat)
            # 4) (N*H*W, C)->(N,H,W,C)
            dx = dx_flat.reshape(N, H, W, C)
            # 5) (N,H,W,C)->(N,C,H,W)
            dx = dx.transpose(0, 3, 1, 2)
        else:
            # 全結合層入力 (N, D) の場合
            dx = self.__backward(dout)

        # もう input_shape に合っているので reshape は不要
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx

#--------------------------------------------------
#Dropoutレイヤ
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        #学習時: train_flg=True
        if train_flg:
            #xを同じ形状のマスクを作って適用
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        #推論時: train_flg=False
        else:
            #削られた分薄める
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask

#--------------------------------------------------
#Convolutionレイヤ
class Convolution:
    #ハイパラ（フィルタ，ストライド，pad等）はコンストラクタでメンバ変数に
    def __init__(self, W, b, stride=1, pad=0):
        """
        W: (FN, C, FH, FW)
        b:（FN, 1, 1）
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    
    #forward
    def forward(self, x):
        """
        x: (N,C,H,W)

        out: (N, FN, out_h, out_w)
        """
        #画像とフィルタの要素
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        #出力要素
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        #画像とフィルタを行列に
        #(N * out_h * out_w,  C * filter_h * filter_w)
        col = function.im2col(x, FH, FW, self.stride, self.pad)
        #(C*FH*FW, FN)
        col_W = self.W.reshape(FN, -1).T

        #行列計算
        out = np.dot(col, col_W) + self.b #-> (N * out_h * out_w, FN)
        #行列から画像へ
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out 

    #backward
    def backward(self, dout):
        """
        dout: (N, FN, out_h, out_w)

        dx: (N,C,H,W)
        """
        FN, C, FH, FW = self.W.shape
        #四次元行列->二次元行列
        dout = dout.transpose(0,2,3,1).reshape(-1, FN) #->(N*out_h*out_w, FN)
        
        #Affine逆伝播(フィルタ，バイアス)
        self.db = np.sum(dout, axis=0) 
        self.dW = np.dot(self.col.T, dout)
        #フィルタを行列->四次元へ
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        #Affine逆伝播（入力データ）
        dcol = np.dot(dout, self.col_W.T)
        #入力データをcol2imで行列->画像へ
        dx = function.col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
#--------------------------------------------------
#Poolingレイヤ
class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        #ハイパラ
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        #逆伝播用
        self.x = None
        self.arg_max = None

    #forward
    def forward(self, x):
        """
        x: (N,C,H,W):actから

        out: (N,C,out_h,out_w):convへ
        """
        N,C,H,W = x.shape

        #pooling通った後の縦横 (縮小)
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        #入力データを行列化 (N * out_h * out_w,  C * pool_h * pool_w)
        col = function.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        #axis=1をチャンネルに分ける (N * out_h * out_w * C, pool_h * pool_w)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        #maxpooling
        arg_max = np.argmax(col, axis=1) #maxのidxを逆伝播用に保存
        out = np.max(col, axis=1) #, pool_h * pool_w)->, 1
        #1次元を4次元にreshape,transposeで元の形状へ
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        #逆伝播用に保存
        self.x = x
        self.arg_max = arg_max

        return out

    #backward
    def backward(self, dout):
        """
        dout: (N,C,out_h,out_w): convから

        dx: (N,C,H,W): reluへ
        """
        #入れ替え：-> (N, out_h, out_w, C)
        dout = dout.transpose(0, 2, 3, 1)
        
        #空行列dmax：maxで1次元になる前のcol形状 (N * out_h * out_w * C, pool_h * pool_w)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))

        #流れてきた勾配（max後の1次元）を，最大値を取った要素に渡す．
        #dmax[[0,1,2…,N*out_h*out_w],[13,2,35…7]] = (N*C*out_h*out_w,)
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        #(N, out_h, out_w, C)->(N, out_h, out_w, C, pool_h * pool_w)
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        #im2col前の形状：(N * out_h * out_w,  C * pool_h * pool_w)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        #入れ替え：-> (N,C,H,W)
        dx = function.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
    
#--------------------------------------------------













        
        

