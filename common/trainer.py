#訓練クラスを作る
#--------------------------------------------------
import numpy as np
#自作
import common.optimization as opt
#--------------------------------------------------
#訓練関数
class Trainer:
    """
    ニューラルネットの訓練を行うクラス
    """
    #自動設定
    def __init__(self, network, 
                 x_train, t_train, x_test, t_test, 
                 epochs=20, mini_batch_size=100,
                 optimizer="SGD", optimizer_param={"lr":0.01},  
                 evaluate_sample_num_per_epoch=None, verbose=True):
        #属性
        self.network = network
        self.verbose = verbose #学習中の進捗を表示するかのフラグ
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch #accに使う数

        #--------------
        #optimizer
        optimizer_class_dict = {'sgd':opt.SGD, 'momentum':opt.Momentum, 
                                'adagrad':opt.AdaGrad,'adam':opt.Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param) 
        #lower: 大文字をすべて小文字に変換
        #**: 辞書アンパック

        #--------------
        #学習回数まわり
        self.train_size = x_train.shape[0]
        #1epochにbatchを流す回数
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        #総学習回数
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        #--------------
        #空リスト
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

#--------------
    #学習1step
    def train_step(self):
        #date整形
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        #勾配計算，最適化
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        #損失記録, 表示
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))

        #1epoch終了ごとにaccを表示
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            #評価用データ用意
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            #評価サンプル数を制限したい場合
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
            #精度計算，格納
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            #verbose=Trueなら表示
            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

#--------------
#学習全体
    def train(self):
        #stepをiter回行う
        for i in range (self.max_iter):
            self.train_step()

        #testでaccを計算，表示
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

#--------------------------------------------------


        




