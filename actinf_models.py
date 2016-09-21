from __future__ import print_function

import numpy as np

from sklearn.neighbors import KNeighborsRegressor

class ActInfModel(object):
    """Base class for active inference function approximators / regressors"""
    def __init__(self, idim = 1, odim = 1):
        self.model = None
        self.idim = idim
        self.odim = odim

    def bootstrap(self): None

    def predict(self, X):
        if self.model is None:
            print("%s.predict: implement me" % (self.__class__.__name__))
            return np.zeros((1, self.odim))
            
    def fit(self, X, Y):
        if self.model is None:
            print("%s.fit: implement me" % (self.__class__.__name__))


class ActInfKNN(ActInfModel):
    """k-NN function approximator for active inference"""
    def __init__(self, idim = 1, odim = 1):
        self.fwd = KNeighborsRegressor(n_neighbors=5)
        ActInfModel.__init__(self, idim, odim)

        self.X_ = []
        self.y_ = []

        self.bootstrap()

    def bootstrap(self):
        # bootstrap model
        print("bootstrapping")
        for i in range(10):
            self.X_.append(np.random.uniform(-0.1, 0.1, (self.idim,)))
            self.y_.append(np.random.uniform(-0.1, 0.1, (self.odim,)))
        # print(self.X_, self.y_)
        self.fwd.fit(self.X_, self.y_)

    def predict(self, X):
        return self.fwd.predict(X)

    def fit(self, X, y):
        self.X_.append(X[0,:])
        # self.y_.append(self.m[0,:])
        # self.y_.append(self.goal[0,:])
        self.y_.append(y[0,:])

        # print("len(X_), len(y_)", len(self.X_), len(self.y_))
        
        self.fwd.fit(self.X_, self.y_)
        

class ActInfSOESGP(ActInfModel):
    """sparse online echo state gaussian process function approximator
    for active inference"""
    def __init__(self, idim = 1, odim = 1):
        from otl_oesgp import OESGP
        ActInfModel.__init__(self, idim, odim)
        
        self.oesgp = OESGP()

        self.res_size = 100 # 20
        self.input_weight = 1.0
        
        self.output_feedback_weight = 0.0
        self.activation_function = 1
        # leak_rate: x <= (1-lr) * input + lr * x
        self.leak_rate = 0.1 # 0.1
        self.connectivity = 0.1
        self.spectral_radius = 0.7
        
        self.kernel_params = [1.0, 1.0]
        self.noise = 0.05
        self.epsilon = 1e-3
        self.capacity = 100
        self.random_seed = 100

        # self.X_ = []
        # self.y_ = []

        self.bootstrap()
    
    def bootstrap(self):
        from smp.reservoirs import res_input_matrix_random_sparse
        self.oesgp.init(self.idim, self.odim, self.res_size, self.input_weight,
                    self.output_feedback_weight, self.activation_function,
                    self.leak_rate, self.connectivity, self.spectral_radius,
                    False, self.kernel_params, self.noise, self.epsilon,
                    self.capacity, self.random_seed)
        im = res_input_matrix_random_sparse(self.idim, self.res_size, 0.2)
        print("im", type(im))
        self.oesgp.setInputWeights(im.tolist())

    def predict(self, X):
        X_ = X.flatten().tolist()
        self.oesgp.update(X_)
        pred = []
        var  = []
        self.oesgp.predict(pred, var)
        # return np.zeros((1, self.odim))
        return np.array(pred).reshape((1, self.odim))
    
    def fit(self, X, y):
        X_ = X.flatten().tolist()
        # print("X.shape", X.shape, len(X_), X_)
        self.oesgp.update(X_)
        # copy state into predefined structure
        # self.oesgp.getState(self.r)

        pred = []
        var  = []
        self.oesgp.predict(pred, var)

        y_ = y.flatten().tolist()
        self.oesgp.train(y_)
        
        # self.oesgp.predict(pred, var)
        # print(pred, var)
        # return np.array(pred).reshape((1, self.odim))

class ActInfSTORKGP(ActInfModel):
    """sparse online echo state gaussian process function approximator
    for active inference"""
    def __init__(self, idim = 1, odim = 1):
        from otl_storkgp import STORKGP
        ActInfModel.__init__(self, idim, odim)
        
        self.storkgp = STORKGP()

        self.res_size = 100 # 20
        
        self.bootstrap()
    
    def bootstrap(self):
        self.storkgp.init(self.idim, self.odim,
                          self.res_size, # window size
                          0, # kernel type
                          [0.5, 0.99, 1.0, self.idim],
                          1e-4,
                          1e-4,
                          100
                          )

    def predict(self, X):
        X_ = X.flatten().tolist()
        self.storkgp.update(X_)
        pred = []
        var  = []
        self.storkgp.predict(pred, var)
        # return np.zeros((1, self.odim))
        return np.array(pred).reshape((1, self.odim))
    
    def fit(self, X, y):
        X_ = X.flatten().tolist()
        # print("X.shape", X.shape, len(X_), X_)
        self.storkgp.update(X_)
        # copy state into predefined structure
        # self.storkgp.getState(self.r)

        pred = []
        var  = []
        self.storkgp.predict(pred, var)

        y_ = y.flatten().tolist()
        self.storkgp.train(y_)
        
        # self.storkgp.predict(pred, var)
        # print(pred, var)
        # return np.array(pred).reshape((1, self.odim))
        
