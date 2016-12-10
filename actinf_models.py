"""Active inference project code

This file contains the learners which can be used as adaptive models of
sensorimotor contexts. For forward models there are
 - nearest neighbour
 - sparse online gaussian process models powered by Harold Soh's OTL library
 - gaussian mixture model

TODO: think about a common calling convention for all model types
   - including 'predict_naive' and 'predict_full' methods that would capture
     returning confidences about the current prediction
   - other variables that might be used by the context to modulate
     exploration, learning and behaviour
   - disambiguate static and dynamic (conditional inference types) idim/odim
TODO: consistency problem when sampling from probabilistic models (gmm, hebbsom, ...)

issues:
 - som track residual error from map training
 - som use residual for adjusting rbf width
 - som extend sampling to sample actual prediction from gaussian with unit's mu and sigma
 - plot current / final som configuration
 - plot densities
 - 

"""


from __future__ import print_function

import numpy as np
import pylab as pl
import cPickle

# KNN
from sklearn.neighbors import KNeighborsRegressor

# Online Gaussian Processes
try:
    from otl_oesgp import OESGP
    from otl_storkgp import STORKGP
    HAVE_SOESGP = True
except ImportError, e:
    print("couldn't import online GP models:", e)
    HAVE_SOESGP = False

# Gaussian mixtures
try:
    import pypr.clustering.gmm as gmm
except ImportError, e:
    print("Couldn't import pypr.clustering.gmm", e)

# hebbsom
try:
    from kohonen.kohonen import Map, Parameters, ExponentialTimeseries, ConstantTimeseries
    from kohonen.kohonen import Gas, GrowingGas, GrowingGasParameters, Filter
except ImportError, e:
    print("Couldn't import lmjohns3's kohonon SOM lib", e)

model_classes = ["KNN", "SOESGP", "STORKGP", "GMM", "HebbSOM"]
        
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

    def save(self, filename):
        cPickle.dump(self, open(filename, "wb"))

    @classmethod
    def load(cls, filename):
        return cPickle.load(open(filename, "rb"))

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
        print("%s.bootstrap'ping" % (self.__class__.__name__))
        for i in range(10):
            self.X_.append(np.random.uniform(-0.1, 0.1, (self.idim,)))
            self.y_.append(np.random.uniform(-0.1, 0.1, (self.odim,)))
        # print(self.X_, self.y_)
        self.fwd.fit(self.X_, self.y_)

    def predict(self, X):
        return self.fwd.predict(X)

    def fit(self, X, y):
        if X.shape[0] > 1: # batch of data
            return self.fit_batch(X, y)
        
        self.X_.append(X[0,:])
        # self.y_.append(self.m[0,:])
        # self.y_.append(self.goal[0,:])
        self.y_.append(y[0,:])

        # print("len(X_), len(y_)", len(self.X_), len(self.y_))
        
        self.fwd.fit(self.X_, self.y_)

    def fit_batch(self, X, y):
        self.X_ = X.tolist()
        self.y_ = y.tolist()
        self.fwd.fit(self.X_, self.y_)
        
################################################################################
# ActiveInference OTL library based model, base class implementing predict,
# predict_step (otl can't handle batches), fit, save and load methods
class ActInfOTLModel(ActInfModel):
    """sparse online echo state gaussian process function approximator
    for active inference"""
    def __init__(self, idim = 1, odim = 1):
        ActInfModel.__init__(self, idim, odim)

        self.otlmodel_type = "soesgp"
        self.otlmodel = None

    def predict(self, X):
        if X.shape[0] > 1: # batch input
            ret = np.zeros((X.shape[0], self.odim))
            for i in range(X.shape[0]):
                ret[i] = self.predict_step(X[i].flatten().tolist())
            return ret
        else:
            X_ = X.flatten().tolist()
            return self.predict_step(X_)

    def predict_step(self, X_):
        self.otlmodel.update(X_)
        pred = []
        var  = []
        self.otlmodel.predict(pred, var)
        # return np.zeros((1, self.odim))
        return np.array(pred).reshape((1, self.odim))
        
    def fit(self, X, y):
        if X.shape[0] > 1: # batch of data
            return self.fit_batch(X, y)
        
        X_ = X.flatten().tolist()
        # print("X.shape", X.shape, len(X_), X_)
        self.otlmodel.update(X_)
        # copy state into predefined structure
        # self.otlmodel.getState(self.r)

        pred = []
        var  = []
        self.otlmodel.predict(pred, var)

        y_ = y.flatten().tolist()
        self.otlmodel.train(y_)
        
        # self.otlmodel.predict(pred, var)
        # print(pred, var)
        # return np.array(pred).reshape((1, self.odim))

    def fit_batch(self, X, y):
        for i in range(X.shape[0]):
            self.fit(X[i], y[i])
        
    def save(self, filename):
        otlmodel_ = self.otlmodel
        self.otlmodel.save(filename + "_%s_model" % self.otlmodel_type)
        print("otlmodel", otlmodel_)
        self.otlmodel = None
        print("otlmodel", otlmodel_)       
        cPickle.dump(self, open(filename, "wb"))
        self.otlmodel = otlmodel_
        print("otlmodel", self.otlmodel)

    @classmethod
    def load(cls, filename):
        # otlmodel_ = cls.otlmodel
        otlmodel_wrap = cPickle.load(open(filename, "rb"))
        print("%s.load cls.otlmodel filename = %s, otlmodel_wrap.otlmodel_type = %s" % (cls.__name__, filename, otlmodel_wrap.otlmodel_type))
        if otlmodel_wrap.otlmodel_type == "soesgp":
            otlmodel_cls = OESGP
        elif otlmodel_wrap.otlmodel_type == "storkgp":
            otlmodel_cls = STORKGP
        else:
            otlmodel_cls = OESGP
            
        otlmodel_wrap.otlmodel = otlmodel_cls()
        print("otlmodel_wrap.otlmodel", otlmodel_wrap.otlmodel)
        otlmodel_wrap.otlmodel.load(filename + "_%s_model" % otlmodel_wrap.otlmodel_type)
        # print("otlmodel_wrap.otlmodel", dir(otlmodel_wrap.otlmodel))
        # cls.bootstrap(otlmodel_wrap)
        # otlmodel_wrap.otlmodel = otlmodel_
        return otlmodel_wrap

################################################################################
# Sparse Online Echo State Gaussian Process (SOESGP) OTL library model
class ActInfSOESGP(ActInfOTLModel):
    """sparse online echo state gaussian process function approximator
    for active inference"""
    def __init__(self, idim = 1, odim = 1):
        ActInfOTLModel.__init__(self, idim, odim)
        
        self.otlmodel_type = "soesgp"
        self.otlmodel = OESGP()

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
        self.otlmodel.init(self.idim, self.odim, self.res_size, self.input_weight,
                    self.output_feedback_weight, self.activation_function,
                    self.leak_rate, self.connectivity, self.spectral_radius,
                    False, self.kernel_params, self.noise, self.epsilon,
                    self.capacity, self.random_seed)
        im = res_input_matrix_random_sparse(self.idim, self.res_size, 0.2)
        print("im", type(im))
        self.otlmodel.setInputWeights(im.tolist())

################################################################################
# StorkGP OTL based model
class ActInfSTORKGP(ActInfOTLModel):
    """sparse online echo state gaussian process function approximator
    for active inference"""
    def __init__(self, idim = 1, odim = 1):
        ActInfModel.__init__(self, idim, odim)
        
        self.otlmodel_type = "storkgp"
        self.otlmodel = STORKGP()

        self.res_size = 100 # 20
        
        self.bootstrap()
    
    def bootstrap(self):
        self.otlmodel.init(self.idim, self.odim,
                          self.res_size, # window size
                          0, # kernel type
                          [0.5, 0.99, 1.0, self.idim],
                          1e-4,
                          1e-4,
                          100
                          )

################################################################################
# inference type multivalued models: GMM, SOMHebb, MDN
# these are somewhat different in operation than the models above
# - fit vs. fit_batch
# - can create conditional submodels

# GMM - gaussian mixture model
class ActInfGMM(ActInfModel):
    def __init__(self, idim = 1, odim = 1, K = 10):
        """ActInfGMM"""
        ActInfModel.__init__(self, idim, odim)

        # number of mixture components
        self.K = K
        # list of K component idim x 1    centroid vectors
        self.cen_lst = []
        # list of K component idim x idim covariances
        self.cov_lst = []
        # K mixture coeffs
        self.p_k = None
        # log loss after training
        self.logL = 0

        self.cdim = self.idim + self.odim

        # data
        self.Xy_ = []
        self.X_  = []
        self.y_  = []
        self.Xy = np.zeros((1, self.cdim))
        # fitting configuration
        self.fit_interval = 100
        self.fitted =  False

        print("%s.__init__, idim = %d, odim = %d" % (self.__class__.__name__, self.idim, self.odim))

    def fit(self, X, y):
        """ActInfGMM single step fit: X, y are single patterns"""
        # print("%s.fit" % (self.__class__.__name__), X.shape, y.shape)
        if X.shape[0] == 1:
            # single step update, add to internal data and refit if length matches update intervale
            self.Xy_.append(np.hstack((X[0], y[0])))
            self.X_.append(X[0])
            self.y_.append(y[0])
            if len(self.Xy_) % self.fit_interval == 0:
                # print("len(Xy_)", len(self.Xy_), self.Xy_[99])
                # pl.plot(self.Xy_)
                # pl.show()
                # self.fit_batch(self.Xy)
                self.fit_batch(self.X_, self.y_)
        else:
            # batch fit, just fit model to the input data batch
            self.Xy_ += np.hstack((X, y)).tolist()
            # self.X_  += X.tolist()
            # self.y_  += y.tolist()
            # self.Xy = np.hstack((X, y))
            # self.Xy  = np.asarray(self.Xy_)
            # print("X_, y_", self.X_, self.y_)
            self.fit_batch(X, y)
        
    def fit_batch(self, X, y):
        """ActInfGMM Fit the GMM model with batch data"""
        # print("%s.fit X.shape = %s, y.shape = %s" % (self.__class__.__name__, X.shape, y.shape))
        # self.Xy = np.hstack((X[:,3:], y[:,:]))
        # self.Xy = np.hstack((X, y))
        # self.Xy = np.asarray(self.Xy_)
        # self.Xy = Xy
        # X = np.asarray(X_)
        # y = np.asarray(y_)
        self.Xy = np.hstack((X, y))
        # self.Xy  = np.asarray(self.Xy_)
        print("%s.fit_batch self.Xy.shape = %s" % (self.__class__.__name__, self.Xy.shape))
        # fit gmm
        self.cen_lst, self.cov_lst, self.p_k, self.logL = gmm.em_gm(self.Xy, K = 10, max_iter = 1000,
                                                                    verbose = False, iter_call = None)
        self.fitted =  True
        print("%s.fit_batch Log likelihood (how well the data fits the model) = %f" % (self.__class__.__name__, self.logL))

    def predict(self, X):
        """ActInfGMM predict: forward to default sample call"""
        return self.sample(X)

    def sample(self, X):
        """ActInfGMM default sample function

        assumes the input is X with dims = idim located in
        the first part of the conditional inference combined input vector

        this method construct the corresponding conditioning input from the reduced input
        """
        # print("%s.sample: X.shape = %s, idim = %d" % (self.__class__.__name__, X.shape, self.idim))
        assert X.shape[1] == self.idim

        # cond = np.zeros((, self.cdim))
        uncond    = np.empty((X.shape[0], self.odim))
        uncond[:] = np.nan
        # print("%s.sample: uncond.shape = %s" % (self.__class__.__name__, uncond.shape))
        # np.array([np.nan for i in range(self.odim)])
        cond = np.hstack((X, uncond))
        # cond[:self.idim] = X.copy()
        # cond[self.idim:] = np.nan
        # print("%s.sample: cond.shape = %s" % (self.__class__.__name__, cond.shape))
        if X.shape[0] > 1: # batch
            return self.sample_batch(cond)
        return self.sample_cond(cond)
    
    def sample_cond(self, X):
        """ActInfGMM single sample from the GMM model with conditioning single input pattern X
        TODO: function conditional_dist, make predict/sample comply with sklearn and use the lowlevel
              cond_dist for advanced uses like dynamic conditioning
        """
        if not self.fitted:
            # return np.zeros((3,1))
            # model has not been bootstrapped, return random goal
            return np.random.uniform(-0.1, 0.1, (1, self.odim)) # FIXME hardcoded shape
    
        # gmm.cond_dist want's a (n, ) shape, not (1, n)
        if len(X.shape) > 1:
            cond = X[0]
        else:
            cond = X

        # print("%s.sample_cond: cond.shape = %s" % (self.__class__.__name__, cond.shape))
        (cen_con, cov_con, new_p_k) = gmm.cond_dist(cond, self.cen_lst, self.cov_lst, self.p_k)
        cond_sample = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
        # print("%s.sample_cond: cond_sample.shape = %s" % (self.__class__.__name__, cond_sample.shape))
        return cond_sample

    def sample_batch(self, X):
        """ActInfGMM.sample_batch: If X has more than one rows, return batch of samples for
        every condition row in X"""
        samples = np.zeros((X.shape[0], self.odim))
        for i in range(X.shape[0]):
            samples[i] = self.sample_cond(X[i])
        return samples
    
    def sample_batch_legacy(self, X, cond_dims = [0], out_dims = [1], resample_interval = 1):
        """ActInfGMM sample from gmm model with conditioning batch input X"""
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
                # print("%s.sample_batch: sampling gmm cond prob at step %d" % (self.__class__.__name__, i))
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

################################################################################
# Hebbian SOM model: connect to SOMs with hebbian links
class ActInfHebbianSOM(ActInfModel):
    def __init__(self, idim = 1, odim = 1):
        """ActInfHebbianSOM"""
        ActInfModel.__init__(self, idim, odim)

        # SOMs trained?
        self.soms_fitted = False
        self.fitted = False
        
        # learning rate proxy
        self.ET = ExponentialTimeseries
        self.CT = ConstantTimeseries
        
        self.mapsize = 5
        self.numepisodes_som  = 20
        self.numepisodes_hebb = 40
        # FIXME: make neighborhood_size decrease with time

        som_lr = 1e-2
        
        # SOM exteroceptive stimuli 2D input
        self.kw_e = self.kwargs(shape = (self.mapsize, self.mapsize), dimension = self.idim, lr_init = som_lr, neighborhood_size = 0.1)
        # self.kw_e = self.kwargs(shape = (self.mapsize, self.mapsize), dimension = self.idim, lr_init = 0.5, neighborhood_size = 0.6)
        self.som_e = Map(Parameters(**self.kw_e))

        # SOM proprioceptive stimuli 3D input
        self.kw_p = self.kwargs(shape = (int(self.mapsize * 1.5), int(self.mapsize * 1.5)), dimension = self.odim, lr_init = som_lr, neighborhood_size = 0.1)
        # self.kw_p = self.kwargs(shape = (int(self.mapsize * 1.5), int(self.mapsize * 1.5)), dimension = self.odim, lr_init = 0.5, neighborhood_size = 0.7)
        self.som_p = Map(Parameters(**self.kw_p))

        # FIXME: there was a nice trick for node distribution init in _some_ recently added paper

        # create "filter" using existing SOM_e, filter computes activation on distance
        self.filter_e = Filter(self.som_e, history=lambda: 0.0)
        self.filter_e.reset()
        self.filter_e_lr = self.filter_e.map._learning_rate

        # kw_f_p = kwargs(shape = (mapsize * 3, mapsize * 3), dimension = 3, neighborhood_size = 0.5, lr_init = 0.1)
        # filter_p = Filter(Map(Parameters(**kw_f_p)), history=lambda: 0.01)
        
        # create "filter" using existing SOM_p, filter computes activation on distance
        self.filter_p = Filter(self.som_p, history=lambda: 0.0)
        self.filter_p.reset()
        self.filter_p_lr = self.filter_p.map._learning_rate

        # Hebbian links
        # hebblink_som    = np.random.uniform(-1e-4, 1e-4, (np.prod(som_e._shape), np.prod(som_p._shape)))
        # hebblink_filter = np.random.uniform(-1e-4, 1e-4, (np.prod(filter_e.map._shape), np.prod(filter_p.map._shape)))
        self.hebblink_som    = np.zeros((np.prod(self.som_e._shape), np.prod(self.som_p._shape)))
        self.hebblink_filter = np.zeros((np.prod(self.filter_e.map._shape), np.prod(self.filter_p.map._shape)))
        self.hebblink_use_activity = True # use activation or distance
        
        # Hebbian learning rate
        if self.hebblink_use_activity:
            # self.hebblink_et = ExponentialTimeseries(-1e-4, 1e-1, 1e-1)
            self.hebblink_et = ConstantTimeseries(5e-3)
            # et = ConstantTimeseries(0.5)
        else:
            self.hebblink_et = ConstantTimeseries(1e-12)

    # SOM argument dict
    def kwargs(self, shape=(10, 10), z=0.001, dimension=2, lr_init = 1.0, neighborhood_size = 1):
        """ActInfHebbianSOM"""
        return dict(dimension=dimension,
                    shape=shape,
                    neighborhood_size = neighborhood_size,
                    learning_rate=self.ET(-1e-4, lr_init, 0.01),
                    noise_variance=z)

    def fit_soms(self, X, y):
        """ActInfHebbianSOM"""
        # print("%s.fit_soms fitting X = %s, y = %s" % (self.__class__.__name__, X.shape, y.shape))
        # if X.shape[0] != 1, r
        # e = EP[i,:dim_e]
        # p = EP[i,dim_e:]
        
        self.filter_e.map._learning_rate = self.filter_e_lr
        self.filter_p.map._learning_rate = self.filter_p_lr

        # don't learn twice
        # som_e.learn(e)
        # som_p.learn(p)
        # TODO for j in numepisodes
        if X.shape[0] > 1:
            numepisodes = self.numepisodes_som
        else:
            numepisodes = 1
        if X.shape[0] > 100:
            print("%s.fit_soms batch fitting of size %d" % (self.__class__.__name__, X.shape[0]))
        i = 0
        for j in range(numepisodes):
            if j > 0 and j % 10 == 0:
                print("%s.fit_soms episode %d / %d" % (self.__class__.__name__, j, numepisodes))
            for i in range(X.shape[0]):
                self.filter_e.learn(X[i])
                self.filter_p.learn(y[i])
            print("%s.fit_soms batch e mean error = %f" % (self.__class__.__name__, np.asarray(self.filter_e.distances_).mean() ))
            print("%s.fit_soms batch p mean error = %f" % (self.__class__.__name__, np.asarray(self.filter_p.distances_).mean() ))
        # print np.argmin(som_e.distances(e)) # , som_e.distances(e)

    def fit_hebb(self, X, y):
        """ActInfHebbianSOM"""
        # print("%s.fit_hebb fitting X = %s, y = %s" % (self.__class__.__name__, X.shape, y.shape))
        # numepisodes_hebb = 1
        if X.shape[0] > 100:
            print("%s.fit_hebb batch fitting of size %d" % (self.__class__.__name__, X.shape[0]))
        numsteps = X.shape[0]
        ################################################################################
        # fix the SOMs with learning rate constant 0
        self.filter_e_lr = self.filter_e.map._learning_rate
        self.filter_p_lr = self.filter_p.map._learning_rate
        # print("fit_hebb", self.filter_e.map._learning_rate)
        self.filter_e.map._learning_rate = self.CT(0.0)
        self.filter_p.map._learning_rate = self.CT(0.0)

        e_shape = (np.prod(self.filter_e.map._shape), 1)
        p_shape = (np.prod(self.filter_p.map._shape), 1)

        z_err_coef = 0.99
        z_err_norm_ = 1
        Z_err_norm  = np.zeros((self.numepisodes_hebb*numsteps,1))
        Z_err_norm_ = np.zeros((self.numepisodes_hebb*numsteps,1))
        W_norm      = np.zeros((self.numepisodes_hebb*numsteps,1))

        # # plotting
        # pl.ion()
        # fig = pl.figure()
        # fig2 = pl.figure()
                    
        # TODO for j in numepisodes
        # j = 0
        if X.shape[0] > 1:
            numepisodes = self.numepisodes_hebb
        else:
            numepisodes = 1
        i = 0
        for j in range(numepisodes):
            if j > 0 and j % 10 == 0:
                print("%s.fit_hebb episode %d / %d" % (self.__class__.__name__, j, numepisodes))
            for i in range(X.shape[0]):
                # just activate
                self.filter_e.learn(X[i])
                self.filter_p.learn(y[i])
        
                # fetch data induced activity
                if self.hebblink_use_activity:
                    p_    = self.filter_p.activity.reshape(p_shape)
                    # print(p_.shape)
                else:
                    p_    = self.filter_p.distances(p).flatten().reshape(p_shape)
        
                e_ = self.filter_e.activity.flatten()
                e__ = e_.copy()
                e_ = (e_ == np.max(e_))
                
                # compute prediction for p using e activation and hebbian weights
                if self.hebblink_use_activity:
                    # print(self.hebblink_filter.T.shape, self.filter_e.activity.reshape(e_shape).shape)
                    # p_bar = np.dot(self.hebblink_filter.T, self.filter_e.activity.reshape(e_shape))
                    # e_act = e_.reshape(e_shape)
                    # e_act
                    p_bar = np.dot(self.hebblink_filter.T, e_.reshape(e_shape))
                else:
                    p_bar = np.dot(self.hebblink_filter.T, self.filter_e.distances(e).flatten().reshape(e_shape))

                p__ = p_.copy()
                p_ = (p_ == np.max(p_)) * 1.0
                p_bar_ = p_bar.copy()
                p_bar = (p_bar == np.max(p_bar)) * 1.0

                # # plotting
                # ax1 = fig.add_subplot(411)
                # ax1.cla()
                # ax1.plot(e_ * np.max(e__))
                # ax1.plot(e__)
                # ax2 = fig.add_subplot(412)
                # ax2.cla()
                # ax2.plot(p_ * np.max(p_bar_))
                # ax2.plot(p__)
                # ax2.plot(p_bar * np.max(p_bar_))
                # ax2.plot(p_bar_)
                # ax3 = fig.add_subplot(413)
                # ax3.cla()
                # ax3.plot(self.filter_e.distances_[-1])
                # ax4 = fig.add_subplot(414)
                # ax4.cla()
                # ax4.plot(self.filter_p.distances_[-1])
                # pl.pause(0.001)
                # pl.draw()
                    
                # inject activity prediction
                p_bar_sum = p_bar.sum()
                if p_bar_sum > 0:
                    p_bar_normed = p_bar / p_bar_sum
                else:
                    p_bar_normed = np.zeros(p_bar.shape)
            
                # compute prediction error: data induced activity - prediction
                # print("p_", np.linalg.norm(p_))
                # print("p_bar", np.linalg.norm(p_bar))
                z_err = p_ - p_bar
                idx = np.argmax(p_)
                # print("idx", idx, z_err[idx])
                # z_err = (p_[idx] - p_bar[idx]) * np.ones_like(p_)
                # z_err = np.ones_like(p_) * 
                # print("z_err", z_err)
                # z_err = p_bar - p_
                z_err_norm = np.linalg.norm(z_err, 2)
                if j == 0 and i == 0:
                    z_err_norm_ = z_err_norm
                else:
                    z_err_norm_ = z_err_coef * z_err_norm_ + (1 - z_err_coef) * z_err_norm
                w_norm = np.linalg.norm(self.hebblink_filter)
        
                logidx = (j*numsteps) + i
                Z_err_norm [logidx] = z_err_norm
                Z_err_norm_[logidx] = z_err_norm_
                W_norm     [logidx] = w_norm
            
                # z_err = p_bar - self.filter_p.activity.reshape(p_bar.shape)
                # print "p_bar.shape", p_bar.shape
                # print "self.filter_p.activity.flatten().shape", self.filter_p.activity.flatten().shape
                
                # if i % 100 == 0:
                #     print("%s.fit_hebb: iter %d/%d: z_err.shape = %s, |z_err| = %f, |W| = %f, |p_bar_normed| = %f" % (self.__class__.__name__, logidx, (self.numepisodes_hebb*numsteps), z_err.shape, z_err_norm_, w_norm, np.linalg.norm(p_bar_normed)))
            
                # d_hebblink_filter = et() * np.outer(self.filter_e.activity.flatten(), self.filter_p.activity.flatten())
                if self.hebblink_use_activity:
                    eta = 5e-3 # self.hebblink_et()
                    # outer = np.outer(self.filter_e.activity.flatten(), np.clip(z_err, 0, 1))
                    # outer = np.outer(e_, np.clip(z_err, 0, 1))
                    # outer = np.outer(e_, p_)
                    # outer = np.outer(e_, p__ * np.clip(z_err, 0, 1))
                    outer = np.outer(e_, p_)
                    
                    # print(outer.shape, self.hebblink_filter.shape)
                    d_hebblink_filter = eta * outer * (1e-3 + z_err[idx])
                    # d_hebblink_filter = eta * np.outer(z_err, self.filter_e.activity.flatten()).T

                    # # plotting
                    # f2ax1 = fig2.add_subplot(111)
                    # f2ax1.imshow(self.hebblink_filter.T, interpolation="none")
                    # # im = f2ax1.imshow(outer, interpolation="none")
                    # # f2ax2 = pl.colorbar(im, ax=f2ax1)
                    # pl.pause(1e-5)
                    # pl.draw()
                      
                else:
                    d_hebblink_filter = self.hebblink_et() * np.outer(self.filter_e.distances(e), z_err)
                self.hebblink_filter += d_hebblink_filter
            # print(Z_err_norm)
            print("%s.fit_hebb error p/p_bar %f" % (self.__class__.__name__, np.array(Z_err_norm)[:logidx].mean()))
            print("%s.fit_hebb w_norm %f" % (self.__class__.__name__, w_norm))
            
            
    def fit(self, X, y):
        """ActInfHebbianSOM"""
        # print("%s.fit fitting X = %s, y = %s" % (self.__class__.__name__, X, y))
        # if X,y have more than one row, train do batch training on SOMs and links
        # otherwise do single step update on both or just the latter?
        self.fit_soms(X, y)
        self.fit_hebb(X, y)
        self.fitted = True

    def predict(self, X):
        """ActInfHebbianSOM"""
        return self.sample(X)

    def sample(self, X):
        """ActInfHebbianSOM.sample"""
        if X.shape[0] > 1: # batch
            return self.sample_batch(X)
        return self.sample_cond(X)

    def sample_cond(self, X):
        """ActInfHebbianSOM.sample_cond: draw single sample from model conditioned on X"""

        e_shape = (np.prod(self.filter_e.map._shape), 1)
        p_shape = (np.prod(self.filter_p.map._shape), 1)

        # activate input network
        self.filter_e.learn(X)

        # propagate activation via hebbian associative links
        if self.hebblink_use_activity:
            e_ = self.filter_e.activity.reshape((np.prod(self.filter_e.map._shape), 1))
            e_ = (e_ == np.max(e_)) * 1.0
            e2p_activation = np.dot(self.hebblink_filter.T, e_)
            # print(e2p_activation)
            self.filter_p.activity = np.clip((e2p_activation / (np.sum(e2p_activation) + 1e-9)).reshape(self.filter_p.map._shape), 0, np.inf)
        else:
            e2p_activation = np.dot(self.hebblink_filter.T, self.filter_e.distances(e).flatten().reshape(e_shape))

        # sample the output network with 
        e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(self.filter_p.sample(1)[0]))
        
        return e2p_w_p_weights.reshape((1, self.odim))
        

    
    def sample_cond_legacy(self, X):
        """ActInfHebbianSOM.sample_cond: sample from model conditioned on X"""
        sampling_search_num = 100

        e_shape = (np.prod(self.filter_e.map._shape), 1)
        p_shape = (np.prod(self.filter_p.map._shape), 1)

        # P_ = np.zeros((X.shape[0], self.odim))
        # E_ = np.zeros((X.shape[0], self.idim))
        
        e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(self.filter_p.sample(1)[0]))
        for i in range(X.shape[0]):
            # e = EP[i,:dim_e]
            # p = EP[i,dim_e:]
            e = X[i]
            # print np.argmin(som_e.distances(e)), som_e.distances(e)
            self.filter_e.learn(e)
            # print "self.filter_e.winner(e)", self.filter_e.winner(e)
            # filter_p.learn(p)
            # print "self.filter_e.activity.shape", self.filter_e.activity.shape
            # import pdb; pdb.set_trace()
            if self.hebblink_use_activity:
                e2p_activation = np.dot(self.hebblink_filter.T, self.filter_e.activity.reshape((np.prod(self.filter_e.map._shape), 1)))
                self.filter_p.activity = np.clip((e2p_activation / np.sum(e2p_activation)).reshape(self.filter_p.map._shape), 0, np.inf)
            else:
                e2p_activation = np.dot(self.hebblink_filter.T, self.filter_e.distances(e).flatten().reshape(e_shape))
            # print "e2p_activation.shape, np.sum(e2p_activation)", e2p_activation.shape, np.sum(e2p_activation)
            # print "self.filter_p.activity.shape", self.filter_p.activity.shape
            # print "np.sum(self.filter_p.activity)", np.sum(self.filter_p.activity), (self.filter_p.activity >= 0).all()
        
            # self.filter_p.learn(p)
            # emodes: 0, 1, 2
            emode = 0 #
            if i % 1 == 0:
                if emode == 0:
                    e2p_w_p_weights_ = []
                    for k in range(sampling_search_num):
                        # filter.sample return the index of the sampled unit
                        e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(self.filter_p.sample(1)[0]))
                        e2p_w_p_weights_.append(e2p_w_p_weights)
                    pred = np.array(e2p_w_p_weights_)
                    # print "pred", pred

                    # # if we can compare against something
                    # pred_err = np.linalg.norm(pred - p, 2, axis=1)
                    # # print "np.linalg.norm(e2p_w_p_weights - p, 2)", np.linalg.norm(e2p_w_p_weights - p, 2)
                    # e2p_w_p = np.argmin(pred_err)

                    # if not pick any
                    e2p_w_p = np.random.choice(pred.shape[0])
                    
                    # print("pred_err", e2p_w_p, pred_err[e2p_w_p])
                    e2p_w_p_weights = e2p_w_p_weights_[e2p_w_p]
                elif emode == 1:
                    if self.hebblink_use_activity:
                        e2p_w_p = np.argmax(e2p_activation)
                    else:
                        e2p_w_p = np.argmin(e2p_activation)
                    e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(e2p_w_p))
                        
                elif emode == 2:
                    e2p_w_p = self.filter_p.winner(p)
                    e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(e2p_w_p))
            # P_[i] = e2p_w_p_weights
            # E_[i] = environment.compute_sensori_effect(P_[i])
            # print("e2p shape", e2p_w_p_weights.shape)
            return e2p_w_p_weights.reshape((1, self.odim))
        
        
    def sample_batch(self, X):
        """ActInfHebbianSOM.sample_batch: If X has more than one rows, return batch of samples for
        every condition row in X"""
        samples = np.zeros((X.shape[0], self.odim))
        for i in range(X.shape[0]):
            samples[i] = self.sample_cond(X[i])
        return samples
    
    def sample_batch_legacy(self, X, cond_dims = [0], out_dims = [1], resample_interval = 1):
        """ActInfHebbianSOM"""
        print("%s.sample_batch_legacy data X = %s" % (self.__class__.__name__, X))
        sampmax = 20
        numsamplesteps = X.shape[0]
        odim = len(out_dims) # self.idim - X.shape[1]
        self.y_sample_  = np.zeros((odim,))
        self.y_sample   = np.zeros((odim,))
        self.y_samples_ = np.zeros((sampmax, numsamplesteps, odim))
        self.y_samples  = np.zeros((numsamplesteps, odim))
        self.cond       = np.zeros_like(X[0])
        
        return self.y_samples, self.y_samples_
    
# MDN model: karpathy, hardmaru, amjad, cbonnett, edward



def get_class_from_name(name = "KNN"):
    if name == "KNN":
        cls = ActInfKNN
    elif name == "SOESGP":
        cls = ActInfSOESGP
    elif name == "STORKGP":
        cls = ActInfSTORKGP
    elif name == "GMM":
        cls = ActInfGMM
    elif name == "HebbSOM":
        cls = ActInfHebbianSOM
    else:
        cls = ActInfKNN
    return cls

def generate_inverted_sinewave_dataset(N = 1000):
    X = np.linspace(0,1,N)
    Y = X + 0.3 * np.sin(2*3.1415926*X) + np.random.uniform(-0.1, 0.1, N)
    X,Y = Y[:,np.newaxis],X[:,np.newaxis]
    return X,Y

def test_model(args):
    import pylab as pl
    # get last component of datafile, the actual filename
    datafilepath_comps = args.datafile.split("/")
    if datafilepath_comps[-1].startswith("EP"):
        idim = 2
        odim = 3
        EP = np.load(args.datafile)
        sl = slice(0, args.numsteps)
        X = EP[sl,:idim]
        Y = EP[sl,idim:]
    elif args.datafile.startswith("inverted"):
        idim = 1
        odim = 1
        X,Y = generate_inverted_sinewave_dataset(N = 1000)
    else:
        idim = 1
        odim = 1

    if args.modelclass == "GMM":
        dim = idim + odim
        
    # diagnostics
    print("X.shape = %s, idim = %d, Y.shape = %s, odim = %d" % (X.shape, idim, Y.shape, odim))

    # pl.subplot(211)
    # pl.plot(X)
    # pl.subplot(212)
    # pl.plot(Y)
    # pl.show()

    mdlcls = get_class_from_name(args.modelclass)
    mdl = mdlcls(idim = idim, odim = odim)

    print("Testing model class %s, %s" % (mdlcls, mdl))

    # print("X.shape = %s, Y.shape = %s" % (X.shape, Y.shape))
    
    mdl.fit(X, Y)

    # plot prediction
    numsamples = 20
    Y_samples = []
    for i in range(numsamples):
        Y_samples.append(mdl.predict(X))
    # print("Y_samples[0]", Y_samples[0])

    for i in range(odim):
        pl.subplot(odim, 1, i+1)
        target     = Y[:,i]
        # prediction = Y_[:,i]
        
        pl.plot(target, "k.", label="Y")
        for j in range(numsamples):
            prediction = Y_samples[j][:,i]
            pl.plot(prediction, "r.", label="Y_", alpha=0.25)
        # get limits
        xlim = pl.xlim()
        ylim = pl.ylim()
        error = target - prediction
        mse   = np.mean(np.square(error))
        mae   = np.mean(np.abs(error))
        xran = xlim[1] - xlim[0]
        yran = ylim[1] - ylim[0]
        pl.text(xlim[0] + xran * 0.1, ylim[0] + yran * 0.3, "mse = %f" % mse)
        pl.text(xlim[0] + xran * 0.1, ylim[0] + yran * 0.5, "mae = %f" % mae)
    pl.show()

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile",  type=str, help="datafile containing t x (dim_extero + dim_proprio) matrix ", default="data/simplearm_n1000/EP_1000.npy")
    parser.add_argument("-m", "--modelclass", type=str, help="Which model class to test [KNN], " + ", ".join(model_classes), default="KNN")
    parser.add_argument("-n", "--numsteps",  type=int, help="Number of datapoints [1000]", default=1000)
    args = parser.parse_args()
    
    test_model(args)
