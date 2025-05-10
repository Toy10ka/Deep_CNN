#optimizer
#--------------------------------------------------
import numpy as np
#--------------------------------------------------
#SGD optimizer
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    #更新式
    def update(self, params, grads): #orderdic
        for key in params.key():
            params[key] -= self.lr * grads[key]

#optimizer = SGD(lr=0.94)
#optimizer.update(params, grads)

#--------------------------------------------------
#Momentum optimizer
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9): 
        self.lr = lr
        self.momentum = momentum
        self.v = None

    #更新式
    def update(self, params, grads):
        #vのinitialize
        if self.v is None:
            self.v = {}
            #初期v：params辞書のvalを0にしたもの
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
#optimizer = Momentum(lr=0.1)
#optimizer.update(params,grads)

#--------------------------------------------------
#AdaGrad optimizer
class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr = lr
        self.h = None
    #更新式
    def update(self, params, grads):
        #hのinitialize
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros.like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # *:要素ごとの積
            params[key] -= self.lr * (grads[key] / (np.sqrt(self.h[key]) + 1e-7)), #h=0対策
#optimizer = AdaGrad(lr=0.3)
#optimizer.update(params, grads)

#--------------------------------------------------
#Adam optimizer
class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999): 
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

#--------------------------------------------------
