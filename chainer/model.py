import numpy as np
import chainer
from chainer import links as L
from chainer import functions as F
from chainer import reporter


class Model(chainer.Chain):
    def __init__(self, H=1024, pred_len=1, n_dims=821, dropout=None, activation=F.leaky_relu):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.pred_len = pred_len
        self.n_dims = n_dims
        with self.init_scope():
            self.l1 = L.Linear(None, H)
            self.r1 = L.LSTM(H, H)
            self.r2 = L.LSTM(H, H)
            self.l2 = L.Linear(H, n_dims)

        #for param in self.params():
        #    param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
            
    def __call__(self, x):
        h = self.activation(self.l1(x))
        if self.dropout: h = F.dropout(h, self.dropout)
        h = self.activation(self.r1(h))
        if self.dropout: h = F.dropout(h, self.dropout)
        h = self.activation(self.r2(h))
        if self.dropout: h = F.dropout(h, self.dropout)
        h = self.l2(h)
        h = F.reshape(h, (self.n_dims,))
        return h
    
    def reset_state(self):
        self.r1.reset_state()
        self.r2.reset_state()

        
class Loss(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.predictor = Model()
            
    def __call__(self, x, t):
        prediction = self.predictor(x)
        #print(prediction.data - F.log(t).data)
        #print(prediction.data, t.data)
        loss = F.sum(F.absolute(prediction - t)) / x.shape[0]
        #loss = F.mean_squared_error(prediction, t) #F.log(prediction), F.log(t))

        reporter.report({"loss": loss}, self)
        return loss
