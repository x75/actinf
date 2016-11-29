
import argparse, cPickle, os
from collections import OrderedDict
from functools   import partial

import numpy as np
import pylab as pl

from explauto import Environment
from explauto.environment import environments
from explauto.environment.pointmass import PointmassEnvironment

# from sklearn.neighbors import KNeighborsRegressor

from utils.functions import gaussian

from actinf_models import ActInfKNN

try:
    from actinf_models import ActInfSOESGP, ActInfSTORKGP
    HAVE_SOESGP = True
except ImportError, e:
    print "couldn't import online GP models", e
    HAVE_SOESGP = False

try:
    import pypr.clustering.gmm as gmm
except ImportError, e:
    print "Couldn't import pypr.clustering.gmm", e
    
# TODO: make custom models: do incremental fit and store history for
#                     learners that need it: knn, soesgp, FORCE, ...
# TODO: watch pred_error, if keeps increasing invert (or model) sign relation
# TODO: how does it react to changing transfer function, or rather, how
#       irregularly can the transfer function be changed for a given learner
# TODO: compute PI,AIS,TE for goal->state, input->pred_error, s_pred->s to
#       answer qu's like: how much is there to learn, how much is learned
# TODO: pass pre-generated system into actinf machine, so we can use random robots with parameterized DoF, stiffness, force budget, DoF coupling coefficient

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
                 model = "knn", numsteps = 1000, idim = None,
                 goal_sample_interval = 50):
        self.mode = mode
        self.model = model
        self.mdl_pkl = "mdl.bin"

        # experiment settings
        self.numsteps = numsteps
        
        # intialize robot environment        
        self.environment = Environment.from_configuration('simple_arm', 'low_dimensional')
        # self.environment = Environment.from_configuration('pointmass', 'low_dim_vel')
        self.environment.noise = 0.

        if idim is None:
            self.idim = self.environment.conf.m_ndims * 2
        else:
            self.idim = idim
        self.odim = self.environment.conf.m_ndims
        # exteroceptive dimensionality
        self.ext_dim = 2 # cartesian

        # prepare run_hooks
        self.run_hooks = OrderedDict()
        print "self.run_hooks", self.run_hooks
        
        # initialize run method and model
        self.init_wiring(self.mode)
        self.init_model (self.model)

        # sensory space mappings: this are used as data store for X,Y
        self.e2p = ActInfKNN(self.ext_dim, self.odim)
        self.p2e = ActInfKNN(self.odim, self.ext_dim)

        ################################################################################
        # logging
        self.logs = {}
        logging_vars = {
            "S_pred": {"col": self.environment.conf.m_ndims},
            "E_pred": {"col": self.environment.conf.m_ndims},
            "M_pred": {"col": self.environment.conf.m_ndims},
            "S_ext":  {"col": self.ext_dim},
            "E2P_pred": {"col": self.odim},
            "P2E_pred": {"col": self.ext_dim},
            "goal_ext": {"col": self.ext_dim},
            "E_pred_e": {"col": self.ext_dim},
            "X_": {"col": -1},
            "X__": {"col": -1},
            "y_": {"col": -1}
        }
        for k, v in logging_vars.items():
            if v["col"] > 0:
                setattr(self, k, np.zeros((self.numsteps, v["col"])))
                self.logs[k] = np.zeros((self.numsteps, v["col"]))
            else:
                self.logs[k] = []
                # setattr(self, k, [])
            
        # self.logs["S_pred"] = np.zeros((self.numsteps, self.environment.conf.m_ndims))
        # self.logs["E_pred"] = np.zeros((self.numsteps, self.environment.conf.m_ndims))
        # self.logs["M_pred"]      = np.zeros((self.numsteps, self.environment.conf.m_ndims)) # == S

        # self.logs["S_ext"] = np.zeros((self.numsteps, self.ext_dim))
        # self.logs["E2P_pred"]    = np.zeros((self.numsteps, self.odim))
        # self.logs["P2E_pred"]    = np.zeros((self.numsteps, self.ext_dim))
        # self.logs["goal_ext"]      = np.zeros((self.numsteps, 2))
        # self.logs["E_pred_e"] = np.zeros((self.numsteps, 2))
        
        # # self.dataset
        # self.logs["X_"]  = [] # (goal, error) appendable list
        # self.logs["X__"] = [] # (goal, error) converted to ndarray
        # self.logs["y_"]  = [] # (state_pred)

        ################################################################################
        # initialize vars with special needs
        self.goal_sample_interval = goal_sample_interval
        self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
        self.goal_tm1 = np.zeros_like(self.goal)
        # self.j = np.zeros((1, self.environment.conf.m_ndims))
        self.M_pred = np.zeros((1, self.environment.conf.m_ndims))
        self.E_pred = self.M_pred - self.goal
        self.S_pred = np.random.normal(0, 0.01, (1, self.environment.conf.m_ndims))

    def init_wiring(self, mode):
        if mode == "type01_state_prediction_error":
            self.run_hooks["hook01"] = self.run_hook_state_prediction_error
            
        elif mode == "type02_state":
            self.run = self.run_type02_state
            
        elif mode == "type03_goal_prediction_error":
            # self.run = self.run_type03_goal_prediction_error
            self.run_hooks["hook01"] = partial(self.run_hook_learn_proprio, iter_start = 0, iter_end = self.numsteps)
            self.run_hooks["hook02"] = self.run_hook_learn_proprio_save
            self.run_hooks["hook03"] = self.run_hook_check_for_model_and_map
            
        elif mode == "type03_1_prediction_error":
            self.run = self.run_type03_1_prediction_error
            
        elif mode == "type04_ext_prop":
            # self.run = self.run_type04_ext_prop
            self.run_hooks["hook01"] = partial(self.run_hook_learn_proprio, iter_start = 0, iter_end = self.numsteps/2)
            self.run_hooks["hook02"] = self.run_hook_learn_proprio_save
            self.run_hooks["hook03"] = self.run_hook_e2p_fit
            self.run_hooks["hook04"] = self.run_hook_e2p_sample
            self.run_hooks["hook05"] = self.run_hook_e2p_sample_plot
            self.run_hooks["hook06"] = partial(self.run_hook_e2p_sample_and_drive, iter_start = self.numsteps/2, iter_end = self.numsteps)
            self.run_hooks["hook07"] = self.run_hook_e2p_sample_and_drive_plot
            
        elif mode == "type05_multiple_models":
            self.run = self.run_type05_multiple_models
            
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
            self.logs["X_"].append(np.random.uniform(-0.1, 0.1, (self.environment.conf.m_ndims * 2,)))
            self.logs["y_"].append(np.random.uniform(-0.1, 0.1, (self.environment.conf.m_ndims,)))
        # print X_, y_
        self.mdl.fit(self.logs["X_"], self.logs["y_"])

    def attr_check(self, attrs):
        """check if object has all attributes in attrs"""
        # check = True
        uncheck = []
        for attr in attrs:
            if not hasattr(self, attr):
                # check = False
                # return check
                uncheck.append(attr)
        if len(uncheck) > 0:
            print "missing attributes %s" % (uncheck)
            return False
        return True

    def load_run(self):
        self.mdl  = cPickle.load(open(self.mdl_pkl, "rb"))
        self.logs = cPickle.load(open("logs.bin", "rb"))
        self.e2p  = cPickle.load(open("e2p.bin", "rb"))
        self.p2e  = cPickle.load(open("p2e.bin", "rb"))
    
    ################################################################################
    # hooks

    # map a model
    def run_hook_check_for_model_and_map(self):
        if os.path.exists(self.mdl_pkl):
            self.load_run()
            self.map_model_type03_goal_prediction_error()
            return

    def run_hook_learn_proprio(self, iter_start = 0, iter_end = 1000):
        """learn control proprioceptive state"""
        if os.path.exists(self.mdl_pkl):
            print "found trained model at %s, skipping learning and using that" % self.mdl_pkl
            # load data from previous run
            self.load_run()
            return

        for i in range(iter_start, iter_end):
            # prepare model input X as goal and prediction error
            self.X_ = np.hstack((self.goal, self.E_pred))

            # predict next state in proprioceptive space
            self.S_pred = self.mdl.predict(self.X_)

            # inverse model / motor primitive / reflex arc
            self.M_pred = self.environment.compute_motor_command(self.S_pred)
            # distort response
            self.M_pred = np.sin(self.M_pred * np.pi) # * 1.333
            # self.M_pred = np.exp(self.M_pred) - 1.0 # * 1.333
            # self.M_pred = (gaussian(0, 0.5, self.M_pred) - 0.4) * 5
            
            # add noise
            self.M_pred += np.random.normal(0, 0.01, self.M_pred.shape)

            # prediction error's
            self.E_pred_goal  = self.M_pred - self.goal
            self.E_pred = self.E_pred_goal
            # # prediction error's variant
            # self.E_pred_state = self.S_pred - self.M_pred
            # self.E_pred = self.E_pred_state
            
            # execute command propagating effect through system, body + environment
            self.S_ext = self.environment.compute_sensori_effect(self.M_pred.T).reshape((1, self.ext_dim))
            # self.environment.plot_arm()
            
            # compute target for the prediction error driven forward model
            # if i % 10 == 0: # play with decreased update rates
            self.y_ = self.S_pred - (self.E_pred * 0.5)
            # FIXME: what is the target if there is no trivial mapping of the error?
            # print "self.y_", self.y_

            # fit the model
            self.mdl.fit(self.X_, self.y_)
            
            # print s_pred
            # print "self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape", self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape

            # extero/proprio mapping predict / fit
            self.E2P_pred = self.e2p.predict(self.S_ext)
            self.P2E_pred = self.p2e.predict(self.M_pred)

            self.e2p.fit(self.S_ext, self.M_pred)
            self.p2e.fit(self.M_pred.reshape((1, self.odim)), self.S_ext)
            
            # logging                
            self.logs["X_"].append(self.X_[0,:])
            # self.logs["y_"].append(self.M_pred[0,:])
            # self.logs["y_"].append(self.goal[0,:])
            self.logs["y_"].append(self.y_[0,:])

            self.logs["S_pred"][i] = self.S_pred
            self.logs["E_pred"][i] = self.E_pred
            self.logs["M_pred"][i]      = self.M_pred
            self.logs["S_ext"][i] = self.S_ext

            self.logs["E2P_pred"][i] = self.E2P_pred
            self.logs["P2E_pred"][i] = self.P2E_pred
            
            if i % self.goal_sample_interval == 0:
                # # continuous goal
                # w = float(i)/self.numsteps
                # f1 = 0.05 # float(i)/10000 + 0.01
                # f2 = 0.08 # float(i)/10000 + 0.02
                # f3 = 0.1 # float(i)/10000 + 0.03
                # self.goal = np.sin(i * np.array([f1, f2, f3])).reshape((1, self.environment.conf.m_ndims))
                # discrete goal
                self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
                print "new goal[%d] = %s" % (i, self.goal)
                print "e_pred = %f" % (np.linalg.norm(self.E_pred, 2))

    def run_hook_learn_proprio_save(self):
        """save data from proprio learning"""
        if not self.attr_check(["logs", "mdl", "mdl_pkl"]):
            return

        # already loaded all data
        if os.path.exists(self.mdl_pkl):
            return
            
        cPickle.dump(self.mdl, open(self.mdl_pkl, "wb"))

        # convert to numpy array
        self.logs["X__"] = np.asarray(self.logs["X_"])
        # np.save("X_.npy", self.logs["X_"])
        
        self.logs["EP"] = np.hstack((np.asarray(self.e2p.X_), np.asarray(self.e2p.y_)))
        # if mdl is type knn?
        self.logs["EP"] = self.logs["EP"][10:]
        print "self.logs[\"EP\"]", type(self.logs["EP"]), self.logs["EP"].shape, self.logs["EP"]
        print "self.logs[\"EP\"].shape = %s, %s" % (self.logs["EP"].shape, self.logs["X__"].shape)
        # print "%d self.logs["EP"].shape = %s".format((0, self.logs["EP"].shape))

        # np.save("EP.npy", self.logs["EP"])

        cPickle.dump(self.logs, open("logs.bin", "wb"))

        cPickle.dump(self.e2p, open("e2p.bin", "wb"))
        cPickle.dump(self.p2e, open("p2e.bin", "wb"))

        # pl.plot(EP[:,:2])
        # pl.show()

    def run_hook_e2p_fit(self):
        # 2. now we learn e2p mapping (conditional joint density model for dealing with ambiguity)
        # ## prepare data
        if not self.attr_check(["logs", "e2p"]):
            return

        # print self.logs["EP"].shape, self.logs["X_"].shape
        # pl.ioff()
        # pl.plot(self.logs["X_"])
        # pl.show()
                
        # fit gmm
        self.cen_lst, self.cov_lst, self.p_k, self.logL = gmm.em_gm(self.logs["EP"], K = 10, max_iter = 1000,\
            verbose = False, iter_call = None)
        print "Log likelihood (how well the data fits the model) = ", self.logL

    def run_hook_e2p_sample(self):
        # intro checks
        if not self.attr_check(["cen_lst", "cov_lst", "p_k", "logL", "logs"]):
            return
        
        # compute conditional
        sampmax = 20
        self.y_sample_   = np.zeros((3,))
        self.y_sample   = np.zeros((3,))
        self.y_samples_ = np.zeros((sampmax, self.logs["EP"].shape[0], 3))
        self.y_samples  = np.zeros((self.logs["EP"].shape[0], 3))
        self.cond       = np.zeros_like(self.logs["EP"][0])
        for i in range(self.logs["EP"].shape[0]):
            # if i % 100 == 0: 
            if i % self.goal_sample_interval == 0:
                print "sampling gmm cond prob at step %d" % i
                ref_interval = 1
                self.cond = self.logs["EP"][(i+ref_interval) % self.logs["EP"].shape[0]] # self.X__[i,:3]
                # self.cond = np.array()
                # self.cond[:2] = X_
                self.cond[2:] = np.nan
                (self.cen_con, self.cov_con, self.new_p_k) = gmm.cond_dist(self.cond, self.cen_lst, self.cov_lst, self.p_k)
                # print self.cond.shape
                samperr = 1e6
                j = 0
                while samperr > 0.1 and j < sampmax:
                    self.y_sample = gmm.sample_gaussian_mixture(self.cen_con, self.cov_con, self.new_p_k, samples = 1)
                    self.y_samples_[j,i] = self.y_sample
                    samperr_ = np.linalg.norm(self.y_sample - self.logs["X__"][(i+1) % self.logs["EP"].shape[0],:3], 2)
                    if samperr_ < samperr:
                        samperr = samperr_
                        self.y_sample_ = self.y_sample
                    j += 1
                    # print "sample/real err", samperr
                print "sampled", j, "times"
            else:
                self.y_samples_[:,i] = self.y_samples_[:,i-1]
            self.y_samples[i] = self.y_sample_

    def run_hook_e2p_sample_plot(self):
        # intro checks
        if not self.attr_check(["y_samples"]):
            return
        
        pl.ioff()
        # 2a. plot sampling results
        pl.suptitle("type04: step 1 + 2: learning proprio, then learning e2p")
        ax = pl.subplot(211)
        pl.title("Exteroceptive state S_e, extero to proprio mapping p2e")
        self.S_ext = ax.plot(self.logs["S_ext"], "k-", alpha=0.8, label="S_e")
        p2e   = ax.plot(self.logs["P2E_pred"], "r-", alpha=0.8, label="p2e")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[handles[i] for i in [0, 2]],
                  labels=[labels[i] for i in [0, 2]])
        ax2 = pl.subplot(212)
        pl.title("Proprioceptive state S_p, proprio to extero mapping e2p")
        ax2.plot(self.logs["M_pred"], "k-", label="S_p")
        # pl.plot(self.logs["E2P_pred"], "y-", label="E2P knn")
        ax2.plot(self.y_samples, "g-", label="E2P gmm cond", alpha=0.8, linewidth=2)
        ax2.plot(self.logs["X__"][:,:3], "r-", label="goal goal")
        for _ in self.y_samples_:
            # print "_", np.sum(_), _
            if np.sum(np.abs(_)) > 1.0: # FIXME: what is that for, for thinning out the number of samples?
                ax2.plot(_, "b.", label="E2P gmm samples", alpha=0.2)
        handles, labels = ax2.get_legend_handles_labels()
        print "handles, labels", handles, labels
        legidx = slice(0, 12, 3)
        ax2.legend(handles[legidx], labels[legidx])
        # ax.legend(handles=[handles[i] for i in [0, 2]],
        #           labels=[labels[i] for i in [0, 2]])
        pl.show()

    def run_hook_e2p_sample_and_drive(self, iter_start = 0, iter_end = 1000):
        # 3. now drive goal from exteroceptive state using e2p mapping

        # # prepare from loaded data
        # print "self.EP.shape", self.logs["EP"].shape
        # self.logs["EP"] = list(self.logs["EP"])
        # print "self.EP len", len(self.logs["EP"]), self.logs["EP"][0]

        # sample first goal
        self.goal_e = self.cond[:2]
        self.goal = gmm.sample_gaussian_mixture(self.cen_con, self.cov_con, self.new_p_k, samples = 1)
        # goal_sample = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_l)

        # FIXME: this is the same as proprio learning, only with e2p sample hook included, fix that
        # - main learning loop needs hooks
        # - how will hooks transfer to smq?
        # - pack e2p into generic model frame
        # - integrate hebbsom
        for i in range(iter_start, iter_end):
            # prepare model input X as goal and prediction error
            self.X_ = np.hstack((self.goal, self.E_pred))

            # predict next state in proprioceptive space
            self.S_pred = self.mdl.predict(self.X_)

            # inverse model / motor primitive / reflex arc
            self.M_pred = self.environment.compute_motor_command(self.S_pred)
            # distort response
            self.M_pred = np.sin(self.M_pred * np.pi) # * 1.333
            # self.M_pred = np.exp(self.M_pred) - 1.0 # * 1.333
            # self.M_pred = (gaussian(0, 0.5, self.M_pred) - 0.4) * 5
            
            # add noise
            self.M_pred += np.random.normal(0, 0.01, self.M_pred.shape)

            # prediction error's
            # self.E_pred = np.zeros(self.M_pred.shape)
            self.E_pred_goal  = self.M_pred - self.goal
            self.E_pred = self.E_pred_goal
            # # prediction error's variant
            # self.E_pred_state = self.S_pred - self.M_pred
            # self.E_pred = self.E_pred_state
            
            # execute command propagating effect through system, body + environment
            self.S_ext = self.environment.compute_sensori_effect(self.M_pred.T).reshape((1, self.ext_dim))
            # self.environment.plot_arm()

            # compute target for the prediction error driven forward model
            # if i % 10 == 0: # play with decreased update rates
            self.y_ = self.S_pred - (self.E_pred * 0.5)
            # FIXME: what is the target if there is no trivial mapping of the error?
            # print "self.y_", self.y_

            # fit the model
            self.mdl.fit(self.X_, self.y_)
            
            # print s_pred
            # print "self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape", self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape

            # extero/proprio mapping predict / fit
            self.logs["E2P_pred"][i] = self.e2p.predict(self.S_ext)
            self.logs["P2E_pred"][i] = self.p2e.predict(self.M_pred)

            self.e2p.fit(self.S_ext, self.M_pred)
            self.p2e.fit(self.M_pred.reshape((1, self.odim)), self.S_ext)
            
            # logging
            self.logs["X_"].append(self.X_[0,:])
            # self.logs["y_"].append(self.M_pred[0,:])
            # self.logs["y_"].append(self.goal[0,:])
            self.logs["y_"].append(self.y_[0,:])

            self.logs["S_pred"][i] = self.S_pred
            self.logs["E_pred"][i] = self.E_pred
            self.logs["M_pred"][i]      = self.M_pred
            self.logs["S_ext"][i] = self.S_ext
            
            self.logs["goal_ext"][i] = self.goal_e
            self.logs["E_pred_e"][i] = self.S_ext - self.goal_e
            
            if i % self.goal_sample_interval == 0:
                # update e2p
                EP = np.hstack((np.asarray(self.e2p.X_), np.asarray(self.e2p.y_)))
                print "EP", EP
                EP = EP[10:]
                if i % 100 == 0:
                    cen_lst, cov_lst, p_k, logL = gmm.em_gm(EP, K = 10, max_iter = 1000,
                                                        verbose = False, iter_call = None)
                print "EP, cen_lst, cov_lst, p_k, logL", EP, cen_lst, cov_lst, p_k, logL
                ref_interval = 1
                self.cond = EP[(i+ref_interval)%EP.shape[0]] # X_[i,:3]
                self.cond[2:] = np.nan
                self.cond_ = np.random.uniform(-1, 1, (5, ))
                self.goal_e = EP[np.random.choice(range(self.numsteps/2)),:2]
                self.cond_[:2] = self.goal_e
                self.cond_[2:] = np.nan
                print "self.cond", self.cond, "self.cond_", self.cond_
                (cen_con, cov_con, new_p_k) = gmm.cond_dist(self.cond_, cen_lst, cov_lst, p_k)
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
                # print "e_pred = %f" % (np.linalg.norm(self.E_pred, 2))

    def run_hook_e2p_sample_and_drive_plot(self):
        # e2pidx = slice(self.numsteps,self.numsteps*2)
        e2pidx = slice(0, self.numsteps)
        pl.suptitle("top: extero goal and extero state, bottom: error_e = |g_e - s_e|^2")
        pl.subplot(211)
        pl.plot(self.logs["goal_ext"][e2pidx])
        pl.plot(self.logs["S_ext"][e2pidx])
        pl.subplot(212)
        pl.plot(np.linalg.norm(self.logs["E_pred_e"][e2pidx], 2, axis=1))
        pl.show()
        
    def run(self):
        """active inference run method definition, iterate dictionary of hooks and execute each
        """
        for k, v in self.run_hooks.items():
            print "key = %s, value = %s" % (k, v)
            # execute value which is a function pointer
            v()
        
    def run_hook_state_prediction_error(self): # type01_
        """active inference / predictive coding: early test, defunct"""
        for i in range(self.numsteps):
            self.X_ = np.hstack((self.goal, self.E_pred)) # model input: goal and prediction error
            # print self.X_
            self.S_pred = self.mdl.predict(self.X_) # state prediction
            self.E_pred = self.S_pred - self.M_pred
            # e_pred = s_pred - goal

            # hack motor primitive
            self.M_pred = self.E_pred + self.M_pred
            # m = e_pred # + m
            self.M_pred += np.random.normal(0, 0.01, self.M_pred.shape)

            # execute command
            self.S_ext = self.environment.compute_sensori_effect(self.M_pred.T)
            # print self.S_ext

            self.y_ = self.M_pred.copy()
            
            self.logs["X_"].append(self.X_[0,:])
            self.logs["y_"].append(self.y_[0,:])
            # y_.append(goal[0,:])
        
            self.mdl.fit(self.X_, self.y_)
    
            # print s_pred
            print "self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape", self.X_.shape, self.S_pred.shape, self.E_pred.shape, self.M_pred.shape, self.S_ext.shape

            self.logs["S_pred"][i] = self.S_pred
            self.logs["E_pred"][i] = self.E_pred
            self.logs["M_pred"][i]      = self.M_pred
    
            if i % 10 == 0:
                self.goal = np.random.uniform(-np.pi/2, np.pi/2, (1, self.environment.conf.m_ndims))

    def run_type02_state(self):
        """active inference / predictive coding: another early test, defunct too"""
        
        for i in range(self.numsteps):
            self.X_ = np.hstack((self.goal, self.E_pred)) # model input: goal and prediction error

            self.S_pred = self.mdl.predict(self.X_) # state prediction

            # hack motor primitive
            self.M_pred = self.environment.compute_motor_command(self.S_pred) #
            # m1 = np.sin(m * 2) # * 1.333
            # m = m1.copy()
            # print m1 - m
            # m += np.random.normal(0, 0.01, m.shape)

            self.E_pred = self.S_pred - self.M_pred # preliminary prediction error == command
            # self.y_ = s_pred - e_pred
            # m = e_pred # + m

            # self.E_pred = self.M_pred - self.goal # real prediction error
            
            # execute command
            self.S_ext = self.environment.compute_sensori_effect(self.M_pred.T)
            # print self.S_ext

            self.logs["X_"].append(self.X_[0,:])
            self.logs["y_"].append(self.y_[0,:])
            # y_.append(goal[0,:])
            # y_.append(self.y_[0,:])

            self.mdl.fit(self.X_, self.M_pred)
            
            # print s_pred
            # print "self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape", self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape

            self.logs["S_pred"][i] = self.S_pred
            self.logs["E_pred"][i] = self.E_pred
            self.logs["M_pred"][i]      = self.M_pred
            
            if i % 50 == 0:
                # goal = np.random.uniform(-np.pi/2, np.pi/2, (1, environment.conf.m_ndims))
                self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))

    ################################################################################
    def map_model_type03_goal_prediction_error(self):
        """plot model output over model input sweep"""
        from mpl_toolkits.mplot3d import Axes3D
        doplot_scattermatrix = False
        
        # turn off interactive mode from explauto
        pl.ioff()

        # sweep error and record output
        # ERROR =
        # meshgrid resolution
        numgrid = 5

        # linear axes
        x_ = np.linspace(-1., 1., numgrid)
        y_ = np.linspace(-1., 1., numgrid)
        z_ = np.linspace(-1., 1., numgrid)

        # meshgrid from axes and resolution
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        # print "x.shape", x.shape
        # print "x", x.flatten()
        # print "y", y.flatten()
        # print "z", z.flatten()
        # combined grid
        error_grid = np.vstack((x.flatten(), y.flatten(), z.flatten()))

        # draw 5 different current state / goal configurations
        X_accum = []
        for i in range(numgrid):
            # randomize initial position
            self.M_pred = self.environment.compute_motor_command(np.random.uniform(-1.0, 1.0, (1, self.environment.conf.m_ndims)))
            # draw random goal and keep it fixed
            self.goal = self.environment.compute_motor_command(np.random.uniform(-1.0, 1.0, (1, self.environment.conf.m_ndims)))
            # self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
            GOALS = np.repeat(self.goal, error_grid.shape[1], axis = 0) # as many goals as error components
            X = np.hstack((GOALS, error_grid.T))
            X_accum.append(X)

        X_accum = np.array(X_accum)
        # ref1 = X_accum[0].copy()
        # print "ref1.shape", ref1.shape
        X_accum = X_accum.reshape((X_accum.shape[0] * X_accum.shape[1], X_accum.shape[2]))
        # ref2 = X_accum[:125].copy()
        # print "ref2.shape", ref2.shape
        # print "ref1 == ref2?", np.all(ref1 == ref2)
        print "X_accum.shape", X_accum.shape
        X = X_accum
        pred = self.mdl.predict(X)
        print "pred.shape", pred.shape
        # X's and pred's indices now mean: slowest: goal, e1, e2, fastest: e3

        ############################################################
        # quiver matrix
        pl.ioff()
        numout = 3
        numoutkombi = [[0, 1], [0, 2], [1, 2]]
        numoutmultf = [25, 5, 1]
        numoutrange = [125, 25, 5]
        # dim x dim matrix: looping over combinations of input and combinations of output
        for i in range(numout): # looping over first error component
        # for i in range(numgrid): # looping over first error component
            dimsel = numoutkombi[i]
            i1 = numoutkombi[i][0]
            i2 = numoutkombi[i][1]
            xidx = range(0, numoutrange[dimsel[0]], numoutmultf[dimsel[0]])
            yidx = range(0, numoutrange[dimsel[1]], numoutmultf[dimsel[1]])
            # curdata  = pred[i*25:(i+1)*25].reshape((numgrid, numgrid, -1))
            curdata  = pred[xidx]
            curgoal  = X_accum[0,:3]
            # curgoal  = X_accum[i*25:(i+1)*25].reshape((numgrid, numgrid, -1))
            print "curgoal.shape", curgoal.shape
            curerror = X_accum[i*25:(i+1)*25].reshape((numgrid, numgrid, -1))
            print "curerror.shape", curerror.shape
            # for j in range(numout): # loop remaining error comps
            # pl.subplot(numgrid, numout, (i*numout) + j + 1)
            pl.subplot(numgrid, 1, i + 1)
            # i1 = numoutkombi[j][0]
            # i2 = numoutkombi[j][1]
            # print "curerror[:,:,i1+3]", curerror[:,:,i1+3].shape
            # pl.plot(curerror[:,:,i1+3], curerror[:,:,i2+3], "ko")
            # pl.show()
            # xidx = np.range(0, )
            # yidx
            print curerror.shape, curdata.shape
            pl.quiver(
                # curerror[:,:,i1+3], curerror[:,:,i2+3],
                curerror[:,:,4], curerror[:,:,5],
                # curdata[:,:,i1],  curdata[:,:,i2]
                curdata[:,1],  curdata[:,2]
                )
            pl.plot([curgoal[i1]], [curgoal[i2]], "ro")
        pl.show()
        # sys.exit()
        
        ############################################################
        # 3D scatter
        pl.ioff()
        fig = pl.figure()
        cols = ["k", "r", "b", "g", "y", "c", "m"]
        axs = [None for i in range(numgrid)]
        for i in range(numgrid):
            print "grid #%d" % i
            sl = slice(i * (numgrid**3), (i+1) * (numgrid**3))
            of = 0 # (i*2)
            axs[i] = fig.add_subplot(1, numgrid, i+1, projection='3d')
            print "sl", sl, "of", of, "ax", axs[i], error_grid.shape
            # axs[i].scatter3D(error_grid.T[sl,0] + of, error_grid.T[sl,1], error_grid.T[sl,2], c=cols[i])
            # axs[i].scatter3D(X_accum[sl,0] + of, X_accum[sl,1], X_accum[sl,2], c=cols[i])
            # axs[i].set_title("GOAL = %s, state = %s" % (X[i*125,:3], X[i*125,3:]))
            # axs[i].set_title("state/GOAL error = %s" % (X[i*125,3:] - X[i*125,:3]))
            axs[i].set_title("GOAL = %s" % (X[i*125,:3]), fontsize=8)
            axs[i].scatter3D(pred[sl,0] + of, pred[sl,1], pred[sl,2], c=cols[i])
            # axs[i].scatter(error_grid.T[sl,0] + of, error_grid.T[sl,1], c=cols[i])
            axs[i].set_aspect(1.0)
            axs[i].set_xlabel("d1")
            axs[i].set_ylabel("d2")
            axs[i].set_zlabel("d3")
            
            # pl.subplot(1, numgrid, i+1)
            # pl.scatter(error_grid.T[sl,0] + of, error_grid.T[sl,1], c=cols[i])
            
        pl.show()

        # scatterplot nd
        # historgram  nd
        # draw a vectorfield?

        # # error_grid = np.concatenate((x, y, z)) # .reshape((numgrid, numgrid, numgrid))
        # print "error_grid", error_grid.shape, error_grid[:]
                
        # # print "x[0,0,0]", x[0,0,[0]],y[0,0,[0]],z[0,0,:]
        # X_ = np.vstack((x[0,0,:],y[0,0,:],z[0,0,:]))
        # print "X_.shape", X_.shape, X_[:,0].shape, X_
        # GOALS = np.repeat(self.goal, numgrid, axis = 0)
        # print "self.goal.shape", self.goal.shape
        # print "GOALS.shape", GOALS.shape
        # X = np.hstack((GOALS, X_.T))
        
        self.E_pred = self.M_pred - self.goal
        # X_s = []
        # X_s.append(np.hstack((self.goal, self.E_pred))) # model input: goal and prediction error
        # X_s.append(np.hstack((self.goal, X_[:,0].reshape((1, 3))))) # model input: goal and prediction error
        # # X_s = X_.tolist()
        # # GOALS = np.repeat
        # X = np.vstack(X_s)
        print "X.shape", X.shape
        # print "pred", self.mdl.predict(X)
        pred = self.mdl.predict(X)
        print "pred.shape", pred.shape

        from sklearn.decomposition import PCA

        X_pca = PCA(n_components = 2)
        # X_pca.fit(X)
        X_pca.fit(error_grid.T)
        pred_pca = PCA(n_components = 1)
        pred_pca.fit(pred)

        # X_red = X_pca.transform(X)
        X_red = X_pca.transform(error_grid.T)
        print "X_red.shape", X_red.shape, np.min(X_red, axis=0), np.max(X_red, axis=0)
        pred_red = pred_pca.transform(pred)
        print "pred_red", pred_red.shape
        
        pl.ioff()

        # ################################################################################
        # # plot type 1
        # import matplotlib.colors as mcol
        # import matplotlib.cm as cm

        # cmap = pl.get_cmap("gray")

        # # m = cm.ScalarMappable(norm=None, cmap=cmap)
        # from matplotlib.colors import colorConverter
        # # X_grid, Y_grid = np.meshgrid(X_red[:,0], X_red[:,1])
        # # print X_grid.shape
        # # pl.pcolormesh(X_red[:,0], X_red[:,1], pred_red)
        # # pl.pcolormesh(X_grid, Y_grid, pred_red)
        # # cc = colorConverter
        # pred_red -= np.min(pred_red) 
        # pred_red /= np.max(pred_red)
        # colorsss = [colorConverter.to_rgb(str(_)) for _ in pred_red.flatten()]
        # print colorsss
        
        # # pl.scatter(X_red[:,0], X_red[:,1], color = colorsss)
        # pl.scatter(X[:,3], X[:,4], color = colorsss, s = 100)
        # pl.scatter(X[:,3] + 2.5, X[:,5], color = colorsss, s = 100)
        # pl.scatter(X[:,4] + 0.0, X[:,5] + 2.5, color = colorsss, s = 100)
        
        # # pl.scatter(X_red[:,0], X_red[:,1], color=pred_red)
        # pl.show()

        ############################################################################
        # plot type 2: hexbin

        ############################################################################
        # plot type 3: pcolormesh, using dimstack
        from smp.dimstack import dimensional_stacking
        from smp.smp_plot import resize_panel_vert, resize_panel_horiz, put_legend_out_right, put_legend_out_top
        # we have 4 axes with 5 steps of variation: goal, e1, e2, e3
        # we have 3 axes with 5**4 responses (function values)
        
        # pl.pcolormesh(X_red[:,0], X_red[:,1], pred_red)
        # A = np.hstack((X, pred[:,[0, 1]]))
        # (5, 5, 5 Goals, 5, 5, 5, errors, 5, 5, 5, preds) z.shape = (5, 5, 5, 5)
        vmin, vmax = np.min(pred), np.max(pred)
        for i in range(pred.shape[1]):
            pl.subplot(1, 3, i+1)
            # p1 = pred[:,i].reshape((numgrid, numgrid, numgrid, numgrid))
            p1 = pred[:,i].reshape((numgrid, numgrid, numgrid, numgrid))
            d1_stacked = dimensional_stacking(p1, [1,0 ], [3, 2])
            pl.pcolormesh(d1_stacked, vmin=vmin, vmax=vmax)
            pl.gca().set_aspect(1.0)
            # if i == (pred.shape[1]/2):
            pl.colorbar(orientation="horizontal") #, use_gridspec=True)
            # resize_panel_horiz(resize_by = 0.8)
            # resize_panel_vert(resize_by = 0.8)
        pl.show()

        ############################################################################
        # plot type 4: scattermatrix
        if doplot_scattermatrix:
            X_grid_margin = np.linspace(np.min(X_red[:,0]), np.max(X_red[:,0]), numgrid)
            Y_grid_margin = np.linspace(np.min(X_red[:,1]), np.max(X_red[:,1]), numgrid)
            X_grid, Y_grid = np.meshgrid(X_grid_margin, Y_grid_margin)

            print "grid shapes", X_grid_margin.shape, Y_grid_margin.shape, X_grid.shape, Y_grid.shape

            import pandas as pd
            from pandas.tools.plotting import scatter_matrix
            # df = pd.DataFrame(X, columns=['x1_t', 'x2_t', 'x1_tptau', 'x2_tptau', 'u_t'])
            scatter_data_raw = np.hstack((error_grid.T, pred))
            print "scatter_data_raw", scatter_data_raw.shape
            
            df = pd.DataFrame(scatter_data_raw, columns=["x_%d" % i for i in range(scatter_data_raw.shape[1])])
            scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')
            pl.show()

        # pl.subplot(121)
        # # pl.scatter(X_red[:,0], X_red[:,1])
        # pl.scatter(X_red[:,0], X_red[:,1])
        # pl.subplot(122)
        # pl.pcolormesh(pred_red)
        # # pl.plot(pred_red)
        # pl.show()
                
                                
        # print "x, y, z", x, y, z
        # print "x, y, z", x.shape, y.shape, z.shape
        # pick given goal transition and corresponding errors, sweep errors

        # try and make it unstable?

        
        
        # self.idim
        # self.odim
        
        
    def run_type03_goal_prediction_error(self):
        """active inference / predictive coding: first working, most basic version,
        proprioceptive only
        
        goal -> goal state prediction -> goal/state error -> update forward model"""
        
        pass
    
    ################################################################################
    def run_type03_1_prediction_error(self):
        """active inference / predictive coding: most basic version (?), proprioceptive only
        
        just goal/state error -> mdl -> goal/state error prediction -> goal/state error -> update forward model"""

        # FIXME: merge this and goal pred error into common sm-loop structure with different hooks coding for
        # the marco-model itself
        # FIXME: this and goal pred error: multiply error with (the sign of) local gradient of the function you're modelling?
        
        # some init foo
        self.X_6 = np.hstack((self.goal, self.E_pred)) # model input: goal and prediction error
        self.X_ = np.hstack((self.E_pred)).reshape((1, self.idim)) # model input: just prediction error

        # if draw:      
        # ax = pl.subplot(111)
        # ax.set_aspect(1)
        
        for i in range(self.numsteps):
            # rearrange to comply with tapping
            # t has just become t+1

            # safe last goal g_{t-1}
            self.goal_tm1 = self.goal.copy()
                        
            # 1. pick a goal g_t
            if False and i % 1 == 0:
                # continuous goal
                w = float(i)/self.numsteps
                f1 = 0.05 # float(i)/10000 + 0.01
                f2 = 0.08 # float(i)/10000 + 0.02
                f3 = 0.1 # float(i)/10000 + 0.03
                self.goal = np.sin(i * np.array([f1, f2, f3])).reshape((1, self.environment.conf.m_ndims))
                
            if i % 50 == 0:
                # discrete goal
                self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
                print "g_[%d] = %s" % (i, self.goal)
                print "e_pred[%d] = %f" % (i-1, np.linalg.norm(self.E_pred, 2))

            # print "goal equal?",

            # 2. new information available / new measurement, s_t
            # inverse model / motor primitive / reflex arc / ...
            self.M_pred = self.environment.compute_motor_command(self.M_pred + self.S_pred) #
            
            # execute command with exteroceptive effect
            self.S_ext = self.environment.compute_sensori_effect(self.M_pred.T)

            # if draw:
            # self.environment.plot_arm(ax, self.M_pred.T)
            # pl.pause(0.001)

            # print self.S_ext
            # self.M_pred +=

            # 2a. optionally distort response
            self.M_pred = np.sin(self.M_pred * np.pi/1.95) # * 1.333
            # self.M_pred = np.exp(self.M_pred) - 1.0 # * 1.333
            # self.M_pred = (gaussian(0, 0.5, self.M_pred) - 0.4) * 5

            # 2b. add noise
            self.M_pred += np.random.normal(0, 0.01, self.M_pred.shape)

            # 3. compute error of measured state s_t with respect to current goal g_t
            #    s_t is called m here for some reason ;)

            # self.E_pred = np.zeros(self.M_pred.shape)
            # self.E_pred_goal  = 
            self.E_pred = self.M_pred - self.goal_tm1 # self.E_pred_goal
            # self.E_pred_state = self.S_pred - self.M_pred
            # self.E_pred = self.E_pred_state
            

            # 4. compute target
            # if i % 10 == 0: # play with decreased update rates
            # self.y_ = self.S_pred - (self.E_pred * 0.02) # error-only
            self.y_ = -self.E_pred * 1.0 # i am amazed this works
            # FIXME: what is the target if there is no trivial mapping of the error?
            # print "self.y_", self.y_

            # 5. fit model
            # self.mdl.fit(self.logs["X_"], self.logs["y_"])
            # if i < 300:
            self.mdl.fit(self.X_, self.y_)

            # debug            
            # print s_pred
            # print "self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape", self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape
                
            # prepare new model inputs
            self.X_6 = np.hstack((self.goal, self.E_pred)) # model input: goal and prediction error

            if np.sum(np.abs(self.goal - self.goal_tm1)) > 1e-6:
                self.X_ = np.hstack((self.M_pred - self.goal)).reshape((1, self.idim)) # model input: just prediction error
            else:
                self.X_ = np.hstack((self.E_pred)).reshape((1, self.idim)) # model input: just prediction error
            # print "self.X_.shape", self.X_.shape
            # self.X_ = np.hstack((self.M_pred, self.E_pred)) # model input: goal and prediction error

            # 6. compute prediction
            self.S_pred = self.mdl.predict(self.X_) # state prediction
            # self.S_pred = self.X_.copy() # identity doesn't do the job

            # logging
            self.logs["X_"].append(self.X_6[0,:]) # also track goal here, self.X_6
            # self.logs["y_"].append(self.M_pred[0,:])
            # self.logs["y_"].append(self.goal[0,:])
            self.logs["y_"].append(self.y_[0,:])

            self.logs["S_pred"][i] = self.S_pred
            self.logs["E_pred"][i] = self.E_pred
            self.logs["M_pred"][i]      = self.M_pred
            

            pl.ioff()
            

    ################################################################################
    def run_type04_ext_prop(self):
        """active inference / predictive coding: version including extero to proprio mapping
        
        goal -> goal state prediction -> goal/state error -> update forward model"""
        # this is for logging and plotting
        # self.logs["S_pred"] = np.zeros((self.numsteps*2, self.environment.conf.m_ndims))
        # self.logs["E_pred"] = np.zeros((self.numsteps*2, self.environment.conf.m_ndims))
        # self.logs["M_pred"]      = np.zeros((self.numsteps*2, self.environment.conf.m_ndims))

        # ################################################################################        
        # # 1. we learn stuff in proprioceptive state
        # for i in range(0, self.numsteps/2):
        #     # prepare model input X as goal and prediction error
        #     self.X_ = np.hstack((self.goal, self.E_pred))
            
        #     # predict next state in proprioceptive space
        #     self.S_pred = self.mdl.predict(self.X_) # state prediction

        #     # inverse model / motor primitive / reflex arc / ...
        #     self.M_pred = self.environment.compute_motor_command(self.S_pred) #
        #     # distort response
        #     self.M_pred = np.sin(self.M_pred * np.pi) # * 1.333
        #     # self.M_pred = np.exp(self.M_pred) - 1.0 # * 1.333
        #     # self.M_pred = (gaussian(0, 0.5, self.M_pred) - 0.4) * 5
            
        #     # add noise
        #     # self.M_pred += np.random.normal(0, 0.01, self.M_pred.shape)

        #     # prediction error's
        #     self.E_pred_goal  = self.M_pred - self.goal
        #     self.E_pred = self.E_pred_goal
        #     # prediction error's variant
        #     # self.E_pred_state = self.S_pred - self.M_pred
        #     # self.E_pred = self.E_pred_state
            
        #     # execute command propagating effect through system, body + environment
        #     self.S_ext = self.environment.compute_sensori_effect(self.M_pred.T).reshape((1, self.ext_dim))
        #     # self.environment.plot_arm()

        #     # compute target for the prediction error driven forward model
        #     # if i % 10 == 0: # play with decreased update rates
        #     self.y_ = self.S_pred - (self.E_pred * 0.8)
        #     # FIXME: what is the target if there is no trivial mapping of the error?
        #     # print "self.y_", self.y_

        #     # fit the model
        #     self.mdl.fit(self.X_, self.y_)
            
        #     # print s_pred
        #     # print "self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape", self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape

        #     # extero/proprio mapping predict / fit
        #     self.logs["E2P_pred"][i] = self.e2p.predict(self.S_ext)
        #     self.logs["P2E_pred"][i] = self.p2e.predict(self.M_pred)

        #     self.e2p.fit(self.S_ext, self.M_pred)
        #     self.p2e.fit(self.M_pred.reshape((1, self.odim)), self.S_ext)
            
        #     # logging                
        #     self.logs["X_"].append(self.X_[0,:])
        #     # self.logs["y_"].append(self.M_pred[0,:])
        #     # self.logs["y_"].append(self.goal[0,:])
        #     self.logs["y_"].append(self.y_[0,:])

        #     self.logs["S_pred"][i] = self.S_pred
        #     self.logs["E_pred"][i] = self.E_pred
        #     self.logs["M_pred"][i]      = self.M_pred
        #     self.logs["S_ext"][i] = self.S_ext

        #     if i % self.goal_sample_interval == 0:
        #         # # continuous goal
        #         # w = float(i)/self.numsteps
        #         # f1 = 0.05 # float(i)/10000 + 0.01
        #         # f2 = 0.08 # float(i)/10000 + 0.02
        #         # f3 = 0.1 # float(i)/10000 + 0.03
        #         # self.goal = np.sin(i * np.array([f1, f2, f3])).reshape((1, self.environment.conf.m_ndims))
        #         # discrete goal
        #         self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
        #         print "new goal[%d] = %s" % (i, self.goal)
        #         print "e_pred = %f" % (np.linalg.norm(self.E_pred, 2))

        # pl.ioff()

        # ################################################################################        
        # # 2. now we learn e2p mapping (conditional joint density model for dealing with ambiguity)
        # # ## prepare data
        # X_ = np.asarray(self.logs["X_"])
        # EP = np.hstack((np.asarray(self.e2p.X_), np.asarray(self.e2p.y_)))
        # EP = EP[10:]
        # print EP.shape

        # np.save("EP.npy", EP)

        # # pl.plot(EP[:,:2])
        # # pl.show()

        # import pypr.clustering.gmm as gmm
        # # fit gmm
        # cen_lst, cov_lst, p_k, logL = gmm.em_gm(EP, K = 10, max_iter = 1000,\
        #     verbose = False, iter_call = None)
        # print "Log likelihood (how well the data fits the model) = ", logL
        
        # # compute conditional
        # sampmax = 20
        # y_sample = np.zeros((3,))
        # y_samples_ = np.zeros((sampmax, EP.shape[0], 3))
        # y_samples = np.zeros((EP.shape[0], 3))
        # for i in range(EP.shape[0]):
        #     if i % 100 == 0: print "sampling gmm cond prob at step %d" % i
        #     if i % self.goal_sample_interval == 0:
        #         ref_interval = 1
        #         cond = EP[(i+ref_interval)%EP.shape[0]] # X_[i,:3]
        #         # cond = np.array()
        #         # cond[:2] = X_
        #         cond[2:] = np.nan
        #         (cen_con, cov_con, new_p_k) = gmm.cond_dist(cond, cen_lst, cov_lst, p_k)
        #         # print cond.shape
        #         samperr = 1e6
        #         j = 0
        #         while samperr > 0.1 and j < sampmax:
        #             y_sample = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
        #             y_samples_[j,i] = y_sample
        #             samperr_ = np.linalg.norm(y_sample - X_[(i+1)%EP.shape[0],:3], 2)
        #             if samperr_ < samperr:
        #                 samperr = samperr_
        #                 self.y_sample_ = y_sample
        #             j += 1
        #             # print "sample/real err", samperr
        #         print "sampled", j, "times"
        #     else:
        #         y_samples_[:,i] = y_samples_[:,i-1]
        #     y_samples[i] = self.y_sample_

        # ################################################################################
        # # 2a. plot sampling results
        # pl.suptitle("type04: step 1 + 2: learning proprio, then learning e2p")
        # ax = pl.subplot(211)
        # pl.title("Exteroceptive state S_e, extero to proprio mapping p2e")
        # self.S_ext = ax.plot(self.logs["S_ext"], "k-", alpha=0.8, label="S_e")
        # p2e   = ax.plot(self.logs["P2E_pred"], "r-", alpha=0.8, label="p2e")
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles=[handles[i] for i in [0, 2]],
        #           labels=[labels[i] for i in [0, 2]])
        # ax2 = pl.subplot(212)
        # pl.title("Proprioceptive state, S_p")
        # ax2.plot(self.logs["M_pred"], "k-", label="S_p")
        # # pl.plot(self.logs["E2P_pred"], "y-", label="E2P knn")
        # ax2.plot(y_samples, "g-", label="E2P gmm cond", alpha=0.8, linewidth=2)
        # for _ in y_samples_:
        #     # print "_", _
        #     if np.sum(_) > 1.0:
        #         ax2.plot(_, "b.", label="E2P gmm samples", alpha=0.2)
        # ax2.plot(X_[:,:3], "r-", label="goal goal")
        # handles, labels = ax2.get_legend_handles_labels()
        # print "handls, labels", handles, labels
        # legidx = slice(0, 9, 3)
        # ax2.legend(handles[legidx], labels[legidx])
        # # ax.legend(handles=[handles[i] for i in [0, 2]],
        # #           labels=[labels[i] for i in [0, 2]])
        # pl.show()

        # # 3. now drive goal from exteroceptive state using e2p mapping
        # # sample first goal
        # self.goal_e = cond[:2]
        # self.goal = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
        # # goal_sample = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_l)
        
        # for i in range(self.numsteps/2, self.numsteps):
        #     self.X_ = np.hstack((self.goal, self.E_pred)) # model input: goal and prediction error
        #     # print "self.X_.shape", self.X_.shape
        #     # self.X_ = np.hstack((self.M_pred, self.E_pred)) # model input: goal and prediction error
        #     self.S_pred = self.mdl.predict(self.X_) # state prediction

        #     # inverse model / motor primitive / reflex arc / ...
        #     self.M_pred = self.environment.compute_motor_command(self.S_pred) #
        #     # distort response
        #     self.M_pred = np.sin(self.M_pred * np.pi) # * 1.333
        #     # self.M_pred = np.exp(self.M_pred) - 1.0 # * 1.333
        #     # self.M_pred = (gaussian(0, 0.5, self.M_pred) - 0.4) * 5
        #     # add noise
        #     # self.M_pred += np.random.normal(0, 0.01, self.M_pred.shape)

        #     # prediction error's
        #     # self.E_pred = np.zeros(self.M_pred.shape)
        #     self.E_pred_goal  = self.M_pred - self.goal
        #     self.E_pred = self.E_pred_goal
        #     # self.E_pred_state = self.S_pred - self.M_pred
        #     # self.E_pred = self.E_pred_state
            
        #     # execute command
        #     self.S_ext = self.environment.compute_sensori_effect(self.M_pred.T).reshape((1, self.ext_dim))

        #     self.logs["E2P_pred"][i] = self.e2p.predict(self.S_ext)
        #     self.logs["P2E_pred"][i] = self.p2e.predict(self.M_pred)

        #     self.e2p.fit(self.S_ext, self.M_pred)
        #     self.p2e.fit(self.M_pred.reshape((1, self.odim)), self.S_ext)
        #     # self.environment.plot_arm()
        #     # print self.S_ext

        #     # if i % 10 == 0: # play with decreased update rates
        #     self.y_ = self.S_pred - (self.E_pred * 0.8)
        #     # FIXME: what is the target if there is no trivial mapping of the error?
        #     # print "self.y_", self.y_
                
        #     self.logs["X_"].append(self.X_[0,:])
        #     # self.logs["y_"].append(self.M_pred[0,:])
        #     # self.logs["y_"].append(self.goal[0,:])
        #     self.logs["y_"].append(self.y_[0,:])

        #     # self.mdl.fit(self.logs["X_"], self.logs["y_"])
        #     # if i < 300:
        #     self.mdl.fit(self.X_, self.y_)
            
        #     # print s_pred
        #     # print "self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape", self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape

        #     self.logs["S_pred"][i] = self.S_pred
        #     self.logs["E_pred"][i] = self.E_pred
        #     self.logs["M_pred"][i]      = self.M_pred
        #     self.logs["S_ext"][i] = self.S_ext
            
        #     self.logs["goal_ext"][i] = self.goal_e
        #     self.logs["E_pred_e"][i] = self.S_ext - self.goal_e

        #     if i % self.goal_sample_interval == 0:
        #         # update e2p
        #         EP = np.hstack((np.asarray(self.e2p.X_), np.asarray(self.e2p.y_)))
        #         EP = EP[10:]
        #         if i % 100 == 0:
        #             cen_lst, cov_lst, p_k, logL = gmm.em_gm(EP, K = 10, max_iter = 1000,
        #                                                 verbose = False, iter_call = None)
        #         print "EP, cen_lst, cov_lst, p_k, logL", EP, cen_lst, cov_lst, p_k, logL
        #         ref_interval = 1
        #         cond = EP[(i+ref_interval)%EP.shape[0]] # X_[i,:3]
        #         cond[2:] = np.nan
        #         cond_ = np.random.uniform(-1, 1, (5, ))
        #         self.goal_e = EP[np.random.choice(range(self.numsteps/2)),:2]
        #         cond_[:2] = self.goal_e
        #         cond_[2:] = np.nan
        #         print "cond", cond, "cond_", cond_
        #         (cen_con, cov_con, new_p_k) = gmm.cond_dist(cond_, cen_lst, cov_lst, p_k)
        #         self.goal = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
        #         # # # continuous goal
        #         # # w = float(i)/self.numsteps
        #         # # f1 = 0.05 # float(i)/10000 + 0.01
        #         # # f2 = 0.08 # float(i)/10000 + 0.02
        #         # # f3 = 0.1 # float(i)/10000 + 0.03
        #         # # self.goal = np.sin(i * np.array([f1, f2, f3])).reshape((1, self.environment.conf.m_ndims))
        #         # # discrete goal
        #         # self.goal = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.environment.conf.m_ndims))
        #         print "new goal[%d] = %s" % (i, self.goal)
        #         # print "e_pred = %f" % (np.linalg.norm(self.E_pred, 2))

        # # e2pidx = slice(self.numsteps,self.numsteps*2)
        # e2pidx = slice(0, self.numsteps)
        # pl.suptitle("top: extero goal and extero state, bottom: error_e = |g_e - s_e|^2")
        # pl.subplot(211)
        # pl.plot(self.logs["goal_ext"][e2pidx])
        # pl.plot(self.logs["S_ext"][e2pidx])
        # pl.subplot(212)
        # pl.plot(np.linalg.norm(self.logs["E_pred_e"][e2pidx], 2, axis=1))
        # pl.show()
        
    ################################################################################
    def run_type05_multiple_models(self):
        """active inference / predictive coding: first working, most basic version,
        proprioceptive only
        
        goal -> goal state prediction -> goal/state error -> update forward model"""
        
        for i in range(self.numsteps):
            # FIXME: this needs a major update: try with a pool of multiple models
            if i == 1200:
                self.environment.factor = 0.8
                
            self.X_ = np.hstack((self.goal, self.E_pred)) # model input: goal and prediction error

            
    def plot_experiment(self):
        # turn off interactive mode from explauto
        pl.ioff()

        if len(self.logs["X_"]) <= 0:
            return
        
        # convert list to array
        if not type(self.logs["X_"]) is np.ndarray:
            self.X__ = np.array(self.logs["X_"])
        else:
            self.X__ = self.logs["X_"].copy()
        # start

        # print self.logs["X_"].shape, self.X__.shape

        err1 = self.logs["M_pred"] - self.X__[:,:3]
        # print err1
        err1 = np.sqrt(np.mean(err1**2))
        err2 = self.logs["E_pred"]
        # print err2
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
        pl.plot(self.logs["S_pred"])
        pl.subplot(514)
        pl.title("error state - goal")
        pl.plot(self.logs["E_pred"])
        pl.subplot(515)
        pl.title("state")
        pl.plot(self.logs["M_pred"])
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
        idim = None
        if args.mode.startswith("type03_1"):
            idim = 3
        inf = ActiveInference(args.mode, args.model, args.numsteps, idim = idim)

        inf.run()

        inf.plot_experiment()

if __name__ == "__main__":
    modes = [
        "test_models",
#        "type01_state_prediction_error",
#        "type02_state",
        "type03_goal_prediction_error",
        "type03_1_prediction_error",
        "type04_ext_prop",
        "type05_multiple_models",
    ]
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="program execution mode, one of " + ", ".join(modes) + " [type01_state_prediction_error]", default="type04_ext_prop")
    parser.add_argument("-md", "--model", type=str, help="learning machine [knn]", default="knn")
    parser.add_argument("-n", "--numsteps", type=int, help="number of learning steps [1000]", default=1000)
    args = parser.parse_args()

    main(args)
