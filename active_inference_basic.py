import argparse
import numpy as np
import pylab as pl

from explauto import Environment

# from sklearn.neighbors import KNeighborsRegressor

from utils.functions import gaussian

from actinf_models import ActInfKNN

try:
    from actinf_models import ActInfSOESGP, ActInfSTORKGP
    HAVE_SOESGP = True
except ImportError, e:
    print "couldn't import online GP models", e
    HAVE_SOESGP = False

# TODO: make custom models: do incremental fit and store history for
#                     learners that need it: knn, soesgp, FORCE, ...
# TODO: watch pred_error, if keeps increasing invert (or model) sign relation
# TODO: how does it react to changing transfer function, or rather, how
#       irregularly can the transfer function be changed for a given learner
# TODO: compute PI,AIS,TE for goal->state, input->pred_error, s_pred->s to
#       answer qu's like: how much is there to learn, how much is learned

# # import numpy as np
# from sklearn import linear_model
# n_samples, n_features = 10, 5
# np.random.seed(0)
# y = np.random.randn(n_samples)
# X = np.random.randn(n_samples, n_features)
# clf = linear_model.SGDRegressor()
# clf.fit(X, y)


class ActiveInference(object):
    def __init__(self, mode = "type01_state_prediction_error",
                 model = "knn", numsteps = 1000):
        self.mode = mode
        self.model = model
        
        self.environment = Environment.from_configuration('simple_arm', 'low_dimensional')
        self.environment.noise = 0.

        self.idim = self.environment.conf.m_ndims * 2
        self.odim = self.environment.conf.m_ndims
        
        # experiment settings
        self.numsteps = numsteps
        # logging
        # X
        self.S_pred = np.zeros((self.numsteps, self.environment.conf.m_ndims))
        self.E_pred = np.zeros((self.numsteps, self.environment.conf.m_ndims))
        self.M      = np.zeros((self.numsteps, self.environment.conf.m_ndims)) # == S

        ################################################################################
        # initialize vars
        # goal = np.ones((1, environment.conf.m_ndims))
        self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
        self.e_pred = np.zeros((1, self.environment.conf.m_ndims))
        self.j = np.zeros((1, self.environment.conf.m_ndims))
        self.m = np.zeros((1, self.environment.conf.m_ndims))


        # random_motors = environment.random_motors(n=100)

        # self.dataset
        self.X_ = [] # (goal, error)
        self.y_ = [] # (state_pred)

        self.init_wiring(self.mode)
        self.init_model (self.model)

    def init_wiring(self, mode):
        if mode == "type01_state_prediction_error":
            self.run = self.run_type01_state_prediction_error
        elif mode == "type02_state":
            self.run = self.run_type02_state
        elif mode == "type03_goal_prediction_error":
            self.run = self.run_type03_goal_prediction_error
        elif mode == "type04_ext_prop":
            self.run = self.run_type04_ext_prop
        else:
            print "unknown mode, FAIL"
            
    def init_model(self, model):
        if not HAVE_SOESGP:
            model = "knn"
            print "Sorry, SOESGP/STORKGP not available, defaulting to knn"
            
        if model == "knn":
            # self.mdl = KNeighborsRegressor(n_neighbors=5)
            self.mdl = ActInfKNN(self.idim, self.odim)
        elif model == "soesgp":
            self.mdl = ActInfSOESGP(self.idim, self.odim)
        elif model == "storkgp":
            self.mdl = ActInfSTORKGP(self.idim, self.odim)
        else:
            print "unknown model, FAIL"

        # self.bootstrap_model()

    # bootstrap model
    def bootstrap_model(self):
        for i in range(10):
            self.X_.append(np.random.uniform(-0.1, 0.1, (self.environment.conf.m_ndims * 2,)))
            self.y_.append(np.random.uniform(-0.1, 0.1, (self.environment.conf.m_ndims,)))
        # print X_, y_
        self.mdl.fit(self.X_, self.y_)


    def run_type01_state_prediction_error(self):
        """active inference / predictive coding: early test, defunct"""
        for i in range(self.numsteps):
            X = np.hstack((self.goal, self.e_pred)) # model input: goal and prediction error
            # print X
            self.s_pred = self.mdl.predict(X) # state prediction
            self.e_pred = self.s_pred - self.m
            # e_pred = s_pred - goal

            # hack motor primitive
            self.m = self.e_pred + self.m
            # m = e_pred # + m
            self.m += np.random.normal(0, 0.01, self.m.shape)

            # execute command
            s_ext = self.environment.compute_sensori_effect(self.m.T)
            # print s_ext
    
            self.X_.append(X[0,:])
            self.y_.append(self.m[0,:])
            # y_.append(goal[0,:])
        
            self.mdl.fit(self.X_, self.y_)
    
            # print s_pred
            print "X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape", X.shape, self.s_pred.shape, self.e_pred.shape, self.m.shape, s_ext.shape

            self.S_pred[i] = self.s_pred
            self.E_pred[i] = self.e_pred
            self.M[i]      = self.m
    
            if i % 10 == 0:
                self.goal = np.random.uniform(-np.pi/2, np.pi/2, (1, self.environment.conf.m_ndims))

    def run_type02_state(self):
        """active inference / predictive coding: another early test, defunct too"""
        
        for i in range(self.numsteps):
            X = np.hstack((self.goal, self.e_pred)) # model input: goal and prediction error

            self.s_pred = self.mdl.predict(X) # state prediction

            # hack motor primitive
            self.m = self.environment.compute_motor_command(self.s_pred) #
            # m1 = np.sin(m * 2) # * 1.333
            # m = m1.copy()
            # print m1 - m
            # m += np.random.normal(0, 0.01, m.shape)

            self.e_pred = self.s_pred - self.m # preliminary prediction error == command
            # tgt = s_pred - e_pred
            # m = e_pred # + m

            # self.e_pred = self.m - self.goal # real prediction error
            
            # execute command
            s_ext = self.environment.compute_sensori_effect(self.m.T)
            # print s_ext

            self.X_.append(X[0,:])
            self.y_.append(self.m[0,:])
            # y_.append(goal[0,:])
            # y_.append(tgt[0,:])

            self.mdl.fit(self.X_, self.y_)
            
            # print s_pred
            # print "X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape", X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape

            self.S_pred[i] = self.s_pred
            self.E_pred[i] = self.e_pred
            self.M[i]      = self.m
            
            if i % 50 == 0:
                # goal = np.random.uniform(-np.pi/2, np.pi/2, (1, environment.conf.m_ndims))
                self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))

    ################################################################################
    def run_type03_goal_prediction_error(self):
        """active inference / predictive coding: first working, most basic version,
        proprioceptive only
        
        goal -> goal state prediction -> goal/state error -> update forward model"""
        
        for i in range(self.numsteps):
            X = np.hstack((self.goal, self.e_pred)) # model input: goal and prediction error
            # print "X.shape", X.shape
            # X = np.hstack((self.m, self.e_pred)) # model input: goal and prediction error
            self.s_pred = self.mdl.predict(X) # state prediction

            # inverse model / motor primitive / reflex arc / ...
            self.m = self.environment.compute_motor_command(self.s_pred) #
            # distort response
            self.m = np.sin(self.m * np.pi) # * 1.333
            # self.m = np.exp(self.m) - 1.0 # * 1.333
            # self.m = (gaussian(0, 0.5, self.m) - 0.4) * 5
            # add noise
            self.m += np.random.normal(0, 0.01, self.m.shape)

            # prediction error's
            # self.e_pred = np.zeros(self.m.shape)
            self.e_pred_goal  = self.m - self.goal
            self.e_pred = self.e_pred_goal
            # self.e_pred_state = self.s_pred - self.m
            # self.e_pred = self.e_pred_state
            
            # execute command
            s_ext = self.environment.compute_sensori_effect(self.m.T)
            # self.environment.plot_arm()
            # print s_ext

            # if i % 10 == 0: # play with decreased update rates
            tgt = self.s_pred - (self.e_pred * 0.5)
            # FIXME: what is the target if there is no trivial mapping of the error?
            # print "tgt", tgt
                
            self.X_.append(X[0,:])
            # self.y_.append(self.m[0,:])
            # self.y_.append(self.goal[0,:])
            self.y_.append(tgt[0,:])

            # self.mdl.fit(self.X_, self.y_)
            # if i < 300:
            self.mdl.fit(X, tgt)
            
            # print s_pred
            # print "X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape", X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape

            self.S_pred[i] = self.s_pred
            self.E_pred[i] = self.e_pred
            self.M[i]      = self.m
            
            if i % 50 == 0:
                # # continuous goal
                # w = float(i)/self.numsteps
                # f1 = 0.05 # float(i)/10000 + 0.01
                # f2 = 0.08 # float(i)/10000 + 0.02
                # f3 = 0.1 # float(i)/10000 + 0.03
                # self.goal = np.sin(i * np.array([f1, f2, f3])).reshape((1, self.environment.conf.m_ndims))
                # discrete goal
                self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
                print "new goal[%d] = %s" % (i, self.goal)
                print "e_pred = %f" % (np.linalg.norm(self.e_pred, 2))

            pl.ioff()

    ################################################################################
    def run_type04_ext_prop(self):
        """active inference / predictive coding: version including extero to proprio mapping
        
        goal -> goal state prediction -> goal/state error -> update forward model"""
        ext_dim = 2 # cartesian
        
        self.e2p = ActInfKNN(ext_dim, self.odim)
        self.p2e = ActInfKNN(self.odim, ext_dim)

        self.S_pred = np.zeros((self.numsteps*2, self.environment.conf.m_ndims))
        self.E_pred = np.zeros((self.numsteps*2, self.environment.conf.m_ndims))
        self.M      = np.zeros((self.numsteps*2, self.environment.conf.m_ndims))
        self.S_ext_ = np.zeros((self.numsteps*2, ext_dim))
        self.E2P    = np.zeros((self.numsteps*2, self.odim))
        self.P2E    = np.zeros((self.numsteps*2, ext_dim))
        self.G_e      = np.zeros((self.numsteps*2, 2))
        self.E_e_pred = np.zeros((self.numsteps*2, 2))
        # self.S_ext_ = np.zeros((self.numsteps, ext_dim))


        # 1. we learn stuff in proprioceptive state                
        for i in range(0, self.numsteps):
            X = np.hstack((self.goal, self.e_pred)) # model input: goal and prediction error
            # print "X.shape", X.shape
            # X = np.hstack((self.m, self.e_pred)) # model input: goal and prediction error
            self.s_pred = self.mdl.predict(X) # state prediction

            # inverse model / motor primitive / reflex arc / ...
            self.m = self.environment.compute_motor_command(self.s_pred) #
            # distort response
            self.m = np.sin(self.m * np.pi) # * 1.333
            # self.m = np.exp(self.m) - 1.0 # * 1.333
            # self.m = (gaussian(0, 0.5, self.m) - 0.4) * 5
            # add noise
            # self.m += np.random.normal(0, 0.01, self.m.shape)

            # prediction error's
            # self.e_pred = np.zeros(self.m.shape)
            self.e_pred_goal  = self.m - self.goal
            self.e_pred = self.e_pred_goal
            # self.e_pred_state = self.s_pred - self.m
            # self.e_pred = self.e_pred_state
            
            # execute command
            s_ext = self.environment.compute_sensori_effect(self.m.T).reshape((1, ext_dim))

            self.E2P[i] = self.e2p.predict(s_ext)
            self.P2E[i] = self.p2e.predict(self.m)

            self.e2p.fit(s_ext, self.m)
            self.p2e.fit(self.m.reshape((1, self.odim)), s_ext)
            # self.environment.plot_arm()
            # print s_ext

            # if i % 10 == 0: # play with decreased update rates
            tgt = self.s_pred - (self.e_pred * 0.8)
            # FIXME: what is the target if there is no trivial mapping of the error?
            # print "tgt", tgt
                
            self.X_.append(X[0,:])
            # self.y_.append(self.m[0,:])
            # self.y_.append(self.goal[0,:])
            self.y_.append(tgt[0,:])

            # self.mdl.fit(self.X_, self.y_)
            # if i < 300:
            self.mdl.fit(X, tgt)
            
            # print s_pred
            # print "X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape", X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape

            self.S_pred[i] = self.s_pred
            self.E_pred[i] = self.e_pred
            self.M[i]      = self.m
            self.S_ext_[i] = s_ext

            goal_sample_interval = 10
                        
            if i % goal_sample_interval == 0:
                # # continuous goal
                # w = float(i)/self.numsteps
                # f1 = 0.05 # float(i)/10000 + 0.01
                # f2 = 0.08 # float(i)/10000 + 0.02
                # f3 = 0.1 # float(i)/10000 + 0.03
                # self.goal = np.sin(i * np.array([f1, f2, f3])).reshape((1, self.environment.conf.m_ndims))
                # discrete goal
                self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
                print "new goal[%d] = %s" % (i, self.goal)
                print "e_pred = %f" % (np.linalg.norm(self.e_pred, 2))

        pl.ioff()

        # try Kernel Density Estimation
        # from sklearn.datasets import load_digits
        # from sklearn.neighbors import KernelDensity
        # from sklearn.decomposition import PCA
        # from sklearn.grid_search import GridSearchCV

        # 2. now we learn e2p mapping (conditional joint density model for dealing with ambiguity)
        X_ = np.asarray(self.X_)
        EP = np.hstack((np.asarray(self.e2p.X_), np.asarray(self.e2p.y_)))
        EP = EP[10:]
        print EP.shape

        np.save("EP.npy", EP)

        # pl.plot(EP[:,:2])
        # pl.show()

        import pypr.clustering.gmm as gmm
        # fit gmm
        cen_lst, cov_lst, p_k, logL = gmm.em_gm(EP, K = 10, max_iter = 1000,\
            verbose = False, iter_call = None)
        print "Log likelihood (how well the data fits the model) = ", logL
        # compute conditional
        sampmax = 20
        y_sample = np.zeros((3,))
        y_samples_ = np.zeros((sampmax, EP.shape[0], 3))
        y_samples = np.zeros((EP.shape[0], 3))
        for i in range(EP.shape[0]):
            if i % 100 == 0: print "sampling gmm cond prob at step %d" % i
            if i % goal_sample_interval == 0:
                ref_interval = 1
                cond = EP[(i+ref_interval)%EP.shape[0]] # X_[i,:3]
                # cond = np.array()
                # cond[:2] = X_
                cond[2:] = np.nan
                (cen_con, cov_con, new_p_k) = gmm.cond_dist(cond, cen_lst, cov_lst, p_k)
                # print cond.shape
                samperr = 1e6
                j = 0
                while samperr > 0.1 and j < sampmax:
                    y_sample = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
                    y_samples_[j,i] = y_sample
                    samperr_ = np.linalg.norm(y_sample - X_[(i+1)%EP.shape[0],:3], 2)
                    if samperr_ < samperr:
                        samperr = samperr_
                        y_sample_ = y_sample
                    j += 1
                    # print "sample/real err", samperr
                print "sampled", j, "times"
            else:
                y_samples_[:,i] = y_samples_[:,i-1]
            y_samples[i] = y_sample_

        # kde = KernelDensity(bandwidth=0.2).fit(EP)
        # print kde.predict()
            
        ax = pl.subplot(211)
        pl.title("Exteroceptive state S_e, extero to proprio mapping p2e")
        s_ext = ax.plot(self.S_ext_, "k-", alpha=0.8, label="S_e")
        p2e   = ax.plot(self.P2E, "r-", alpha=0.8, label="p2e")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[handles[i] for i in [0, 2]],
                  labels=[labels[i] for i in [0, 2]])
        ax2 = pl.subplot(212)
        pl.title("Proprioceptive state, S_p")
        ax2.plot(self.M, "k-", label="S_p")
        # pl.plot(self.E2P, "y-", label="E2P knn")
        ax2.plot(y_samples, "g-", label="E2P gmm cond", alpha=0.8, linewidth=2)
        for _ in y_samples_:
            # print "_", _
            if np.sum(_) > 1.0:
                ax2.plot(_, "b.", label="E2P gmm samples", alpha=0.2)
        ax2.plot(X_[:,:3], "r-", label="goal goal")
        handles, labels = ax2.get_legend_handles_labels()
        print "handls, labels", handles, labels
        legidx = slice(0, 9, 3)
        ax2.legend(handles[legidx], labels[legidx])
        # ax.legend(handles=[handles[i] for i in [0, 2]],
        #           labels=[labels[i] for i in [0, 2]])
        pl.show()

        # now drive goal from exteroceptive state

        # sample first goal
        self.goal_e = cond[:2]
        self.goal = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
        # goal_sample = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_l)
        
        for i in range(self.numsteps, self.numsteps*2):
            X = np.hstack((self.goal, self.e_pred)) # model input: goal and prediction error
            # print "X.shape", X.shape
            # X = np.hstack((self.m, self.e_pred)) # model input: goal and prediction error
            self.s_pred = self.mdl.predict(X) # state prediction

            # inverse model / motor primitive / reflex arc / ...
            self.m = self.environment.compute_motor_command(self.s_pred) #
            # distort response
            self.m = np.sin(self.m * np.pi) # * 1.333
            # self.m = np.exp(self.m) - 1.0 # * 1.333
            # self.m = (gaussian(0, 0.5, self.m) - 0.4) * 5
            # add noise
            # self.m += np.random.normal(0, 0.01, self.m.shape)

            # prediction error's
            # self.e_pred = np.zeros(self.m.shape)
            self.e_pred_goal  = self.m - self.goal
            self.e_pred = self.e_pred_goal
            # self.e_pred_state = self.s_pred - self.m
            # self.e_pred = self.e_pred_state
            
            # execute command
            s_ext = self.environment.compute_sensori_effect(self.m.T).reshape((1, ext_dim))

            self.E2P[i] = self.e2p.predict(s_ext)
            self.P2E[i] = self.p2e.predict(self.m)

            self.e2p.fit(s_ext, self.m)
            self.p2e.fit(self.m.reshape((1, self.odim)), s_ext)
            # self.environment.plot_arm()
            # print s_ext

            # if i % 10 == 0: # play with decreased update rates
            tgt = self.s_pred - (self.e_pred * 0.8)
            # FIXME: what is the target if there is no trivial mapping of the error?
            # print "tgt", tgt
                
            self.X_.append(X[0,:])
            # self.y_.append(self.m[0,:])
            # self.y_.append(self.goal[0,:])
            self.y_.append(tgt[0,:])

            # self.mdl.fit(self.X_, self.y_)
            # if i < 300:
            self.mdl.fit(X, tgt)
            
            # print s_pred
            # print "X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape", X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape

            self.S_pred[i] = self.s_pred
            self.E_pred[i] = self.e_pred
            self.M[i]      = self.m
            self.S_ext_[i] = s_ext
            
            self.G_e[i] = self.goal_e
            self.E_e_pred[i] = s_ext - self.goal_e

            goal_sample_interval = 10
                        
            if i % goal_sample_interval == 0:
                # update e2p
                EP = np.hstack((np.asarray(self.e2p.X_), np.asarray(self.e2p.y_)))
                EP = EP[10:]
                if i % 100 == 0:
                    cen_lst, cov_lst, p_k, logL = gmm.em_gm(EP, K = 10, max_iter = 1000,
                                                        verbose = False, iter_call = None)
                print "EP, cen_lst, cov_lst, p_k, logL", EP, cen_lst, cov_lst, p_k, logL
                ref_interval = 1
                cond = EP[(i+ref_interval)%EP.shape[0]] # X_[i,:3]
                cond[2:] = np.nan
                cond_ = np.random.uniform(-1, 1, (5, ))
                self.goal_e = EP[np.random.choice(range(self.numsteps)),:2]
                cond_[:2] = self.goal_e
                cond_[2:] = np.nan
                print "cond", cond, "cond_", cond_
                (cen_con, cov_con, new_p_k) = gmm.cond_dist(cond_, cen_lst, cov_lst, p_k)
                self.goal = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
                # # # continuous goal
                # # w = float(i)/self.numsteps
                # # f1 = 0.05 # float(i)/10000 + 0.01
                # # f2 = 0.08 # float(i)/10000 + 0.02
                # # f3 = 0.1 # float(i)/10000 + 0.03
                # # self.goal = np.sin(i * np.array([f1, f2, f3])).reshape((1, self.environment.conf.m_ndims))
                # # discrete goal
                # self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
                print "new goal[%d] = %s" % (i, self.goal)
                # print "e_pred = %f" % (np.linalg.norm(self.e_pred, 2))

        e2pidx = slice(self.numsteps,self.numsteps*2)
        pl.suptitle("top: extero goal and extero state, bottom: error_e = |g_e - s_e|^2")
        pl.subplot(211)
        pl.plot(self.G_e[e2pidx])
        pl.plot(self.S_ext_[e2pidx])
        pl.subplot(212)
        pl.plot(np.linalg.norm(self.E_e_pred[e2pidx], 2, axis=1))
        pl.show()
        
    def plot_experiment(self):
        # convert list to array
        self.X__ = np.array(self.X_)
        # start

        err1 = self.M - self.X__[:,:3]
        print err1
        err1 = np.sqrt(np.mean(err1**2))
        err2 = self.E_pred
        print err2
        err2 = np.sqrt(np.mean(err2**2))
        print "errors: e1 = %f, e2 = %f" % (err1, err2)
                
        pl.ioff()
        pl.suptitle("mode: %s using %s (X: FM input, state pred: FM output)" % (self.mode, self.model))
        pl.subplot(511)
        pl.title("X[goals]")
        # pl.plot(self.X__[10:,0:3], "-x")
        pl.plot(self.X__[:,0:3], "-x")
        pl.subplot(512)
        pl.title("X[prediction error]")
        # pl.plot(self.X__[10:,3:], "-x")
        pl.plot(self.X__[:,3:], "-x")
        pl.subplot(513)
        pl.title("state pred")
        pl.plot(self.S_pred)
        pl.subplot(514)
        pl.title("error state - goal")
        pl.plot(self.E_pred)
        pl.subplot(515)
        pl.title("state")
        pl.plot(self.M)
        pl.show()
        
    
# pl.ioff()
# ax = pl.axes()

# for m in random_motors:
#     self.mdl.fit(X, y) 

#     environment.plot_arm(ax, m)
# pl.show()

# pl.wait-for-wm
# pl.ioff()


# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]
# KSelf.MdlborsRegressor(...)
# print(self.mdl.predict([[1.5]]))
# [ 0.5]

def test_models(args):
    from actinf_models import ActInfModel
    from actinf_models import ActInfKNN
    from actinf_models import ActInfSOESGP

    idim = 4
    odim = 2
    numdatapoints = 10
    
    for aimclass in [ActInfModel, ActInfKNN, ActInfSOESGP]:
        print("aimclass", aimclass)
        aim = aimclass(idim = idim, odim = odim)

        X = np.random.uniform(-0.1, 0.1, (numdatapoints, 1, idim))
        y = np.random.uniform(-0.1, 0.1, (numdatapoints, 1, odim))

        for i in range(numdatapoints-1):
            aim.fit(X[i], y[i])
        y_ = aim.predict(X[i+1])
        print("prediction error = %s" % (y_ - y[i+1]))

    
def main(args):
    if args.mode.startswith("test_"):
        test_models(args)
    else:
        inf = ActiveInference(args.mode, args.model, args.numsteps)

        inf.run()

        inf.plot_experiment()

if __name__ == "__main__":
    modes = [
        "test_models",
        "type01_state_prediction_error",
        "type02_state",
        "type03_goal_prediction_error",
        "type04_ext_prop",
    ]
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="program execution mode, one of " + ", ".join(modes) + " [type01_state_prediction_error]", default="type04_ext_prop")
    parser.add_argument("-md", "--model", type=str, help="learning machine [knn]", default="knn")
    parser.add_argument("-n", "--numsteps", type=int, help="number of learning steps [1000]", default=1000)
    args = parser.parse_args()

    main(args)