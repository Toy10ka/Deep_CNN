#メソッドをまとめる
import numpy as np 
#---------------------------------------------------
#活性化関数

#ステップ関数
def step_function(x): #配列が入るようになるトリック
    y = x > 0 #yにはTrue,Falseが入る
    return y.astype(int) #boolをintにすると0,1が得られる

#sigmoid関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ReLU関数
def relu(x):
    return np.maximum(0, x)

#恒等関数
def identity_function(x): #出力層に適用する活性化関数
    return x 

#softmax関数(batch対応)
def softmax(a): #入力:(batch, class)
    # 1次元なら(1,C)に拡張
    if a.ndim == 1:
        a = a.reshape(1, -1)  # -> shape (1, C)
        squeezed = True
    else:
        squeezed = False     # 既に (N, C)

    #オーバーフロー対策(定数±しても不変　を利用)
    c = np.max(a, axis=1, keepdims=True) #dim(batch,class)->(batch,1)
    exp_a = np.exp(a - c)
    #クラス方向に和
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True) #->(batch,1)
    y = exp_a / sum_exp_a #ブロードキャストで(batch,class)

    # もともと1次元入力だったら、(1,C) → (C,) に戻す
    if squeezed:
        y = y.reshape(-1) # -> (C,)

    return y #->(batch,class)
#---------------------------------------------------
#損失関数

#平均二乗誤差
def sum_squared_error(y, t): #y,t: ndarray
    return 0.5 * np.sum((y-t)**2)

#クロスエントロピー誤差(batch対応)
def cross_entropy_error(y, t): #(batch_size, class)
    #batch処理じゃないときは整形
    if y.ndim == 1: 
        t = t.reshape(1, t.size) #(class,)->(batch_size, class)
        y = y.reshape(1, y.size)
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
    #batchの場合
    batch_size = y.shape[0] #number of batch
    delta = 1e-7 #-∞発散を防ぎたい（10*e^-7）
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size #->スカラ（batch毎にひとつのloss）

#---------------------------------------------------
#その他関数

#数値微分
def numerical_diff(f, x):
    h = 1e-4 #小さすぎると丸め誤差に
    return (f(x+h) - f(x-h)) / (2*h) #中心差分

#多次元要素それぞれに勾配計算
def numerical_gradient(f, x):#x.shape()
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    #多次元配列の全要素を効率よくループ処理するためのイテレータ
    #op_flags=['readwrite']で，新しい配列を用意せず直接値を書き換えられる
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #もう最後の要素まで取り出したか=notなら
    while not it.finished:
        #今のインデックスについて同じ操作
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

#---------------------------------------------------
#im2col(入力データをフィルタサイズ, stride, padに応じて行列に変える)
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング（上下左右に pad 個ずつゼロ埋め）

    Returns
    -------
    col : 2次元配列(N * out_h * out_w,  C * filter_h * filter_w)
    """
    #imgから形を取り出して，出力の高さ・幅を計算
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    #(N,C) 軸はそのままに，高さ・幅軸だけ pad 分ずつ 0 埋め
    #constant: パディング領域を一定の定数で埋める
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    #パッチ(カーネルが重み掛けを行う、小さな局所領域)を詰め込むための空配列
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    #各パッチをスライドして格納
    for y in range(filter_h): #フィルタの縦
        y_max = y + stride*out_h #パッチ最下（原点左上）

        for x in range(filter_w): #フィルタの横
            x_max = x + stride*out_w#パッチ最右

            #スライス後のimg形状は `(N, C, out_h, out_w)
            #`col` の対応スロット `(N, C, y, x, out_h, out_w)` に格納
            # (y,x):フィルタ内部の位置
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    #(N, C, filter_h, filter_w, out_h, out_w)->(N, out_h, out_w, C, filter_h, filter_w)
    #そのあとreshape
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
#---------------------------------------------------
#col2im
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    col :(N * out_h * out_w,  C * filter_h * filter_w)
    input_shape : (N,C,H,W)）
    filter_h :
    filter_w
    stride
    pad

    img: (N,C,H,W)
    """
    N, C, H, W = input_shape
    #出力の縦横
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    #im2colでreshapeしたのを戻す
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    #col2img(逆変換)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad] #pad領域をトリミング

#---------------------------------------------------
