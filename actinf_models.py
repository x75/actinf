from __future__ import print_function

import numpy as np

from sklearn.neighbors import KNeighborsRegressor

try:
    from otl_oesgp import OESGP
    from otl_storkgp import STORKGP
    HAVE_SOESGP = True
except ImportError, e:
    print("couldn't import online GP models:", e)
    HAVE_SOESGP = False

try:
    import pypr.clustering.gmm as gmm
except ImportError, e:
    print("Couldn't import pypr.clustering.gmm", e)

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

################################################################################
# inference type multivalued models: GMM, SOMHebb, MDN
# these are somewhat different in operation than the models above
# - fit vs. fit_batch
# - can create conditional submodels

# GMM - gaussian mixture model
class ActInfGMM(ActInfModel):
    def __init__(self, idim = 1, odim = 1):
        ActInfModel.__init__(self, idim, odim)

        # number of mixture components
        self.K = 10
        # list of K component idim x 1    centroid vectors
        self.cen_lst = []
        # list of K component idim x idim covariances
        self.cov_lst = []
        # K mixture coeffs
        self.p_k = None
        # log loss after training
        self.logL = 0

        # data
        self.Xy_ = []
        self.Xy = np.zeros((1, self.idim))
        # fitting configuration
        self.fit_interval = 100
        self.fitted =  False

    def fit(self, X, y):
        """single step fit: X, y are single patterns"""
        if X.shape[0] == 1:
            # single step update, add to internal data and refit if length matches update intervale
            self.Xy_.append(np.hstack((X, y)))
            if len(self.Xy_) % self.fit_interval == 0:
                self.fit_batch(np.asarray(self.Xy_))
        else:
            # batch fit, just fit mode to the input data
            self.Xy = np.hstack((X, y))
            self.fit_batch(self.Xy)
        
    def fit_batch(self, Xy):
        """fit the model"""
        # print("%s.fit X.shape = %s, y.shape = %s" % (self.__class__.__name__, X.shape, y.shape))
        # self.Xy = np.hstack((X[:,3:], y[:,:]))
        # self.Xy = np.hstack((X, y))
        # self.Xy = np.asarray(self.Xy_)
        self.Xy = Xy
        print("%s.fit_batch self.Xy.shape = %s" % (self.__class__.__name__, self.Xy.shape))
        # fit gmm
        self.cen_lst, self.cov_lst, self.p_k, self.logL = gmm.em_gm(self.Xy, K = 10, max_iter = 1000,
                                                                    verbose = False, iter_call = None)
        self.fitted =  True
        print("%s.fit Log likelihood (how well the data fits the model) = %f" % (self.__class__.__name__, self.logL))

    def predict(self, X):
        return self.sample(X)

    def sample(self, X):
        """sample from the model with conditioning single input pattern X"""
        if not self.fitted:
            return np.zeros((3,1))
        cond = X
        (cen_con, cov_con, new_p_k) = gmm.cond_dist(cond, self.cen_lst, self.cov_lst, self.p_k)
        return gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
        
    def sample_batch(self, X, cond_dims = [0], out_dims = [1], resample_interval = 1):
        """sample from the model with conditioning batch input X"""
        # compute conditional
        sampmax = 20
        numsamplesteps = X.shape[0]
        odim = len(out_dims) # self.idim - X.shape[1]
        self.y_sample_  = np.zeros((odim,))
        self.y_sample   = np.zeros((odim,))
        self.y_samples_ = np.zeros((sampmax, numsamplesteps, odim))
        self.y_samples  = np.zeros((numsamplesteps, odim))
        self.cond       = np.zeros_like(X[0])

        print("%s.sample_batch: y_samples_.shape = %s" % (self.__class__.__name__, self.y_samples_.shape))
        
        for i in range(numsamplesteps):
            # if i % 100 == 0:
            if i % resample_interval == 0:
                print("%s.sample_batch: sampling gmm cond prob at step %d" % (self.__class__.__name__, i))
                ref_interval = 1
                # self.cond = self.logs["EP"][(i+ref_interval) % self.logs["EP"].shape[0]] # self.X__[i,:3]
                self.cond = X[(i+ref_interval) % numsamplesteps] # self.X__[i,:3]
                # self.cond = np.array()
                # self.cond[:2] = X_
                self.cond[out_dims] = np.nan
                (self.cen_con, self.cov_con, self.new_p_k) = gmm.cond_dist(self.cond, self.cen_lst, self.cov_lst, self.p_k)
                # print "run_hook_e2p_sample gmm.cond_dist:", np.array(self.cen_con).shape, np.array(self.cov_con).shape, self.new_p_k.shape
                samperr = 1e6
                j = 0
                while samperr > 0.1 and j < sampmax:
                    self.y_sample = gmm.sample_gaussian_mixture(self.cen_con, self.cov_con, self.new_p_k, samples = 1)
                    self.y_samples_[j,i] = self.y_sample
                    samperr_ = np.linalg.norm(self.y_sample - X[(i+1) % numsamplesteps,:odim], 2)
                    if samperr_ < samperr:
                        samperr = samperr_
                        self.y_sample_ = self.y_sample
                    j += 1
                    # print "sample/real err", samperr
                print("sampled", j, "times")
            else:
                # retain samples from last sampling interval boundary
                self.y_samples_[:,i] = self.y_samples_[:,i-1]
            # return sample array
            self.y_samples[i] = self.y_sample_
            
        return self.y_samples, self.y_samples_
