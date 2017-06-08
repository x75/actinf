
import argparse, cPickle, os, sys
from collections import OrderedDict
from functools   import partial # mhm :)

import numpy as np
import pylab as pl
import matplotlib.gridspec as gridspec
import pandas as pd

import explauto
from explauto import Environment
from explauto.environment import environments
from explauto.environment.pointmass import PointmassEnvironment

# from sklearn.neighbors import KNeighborsRegressor

# from utils.functions import gaussian

from actinf_models import ActInfKNN, ActInfGMM, ActInfHebbianSOM

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

SAVEPLOTS = True
    
# TODO: make fwd model aware of goal change to ignore old prediction error which is irrelevant for the new goal
# TODO: make fwd model aware of it's learning state and wether it can advantageously accomodate new data or
#       is fine to ignore it. Also: prune old data?
    
# TODO: variant 0 and 1: multiply error with (the sign of) local gradient of the function you're modelling for non-monotonic cases?
#       need to plot behaviour for non-invertible functions
# TODO: make custom models: do incremental fit and store history for
#                     learners that need it: knn, soesgp, FORCE, ...
# TODO: watch pred_error, if keeps increasing invert (or model) sign relation
# TODO: how does it react to changing transfer function, or rather, how
#       irregularly can the transfer function be changed for a given learner: monotonicity and all above
# TODO: compute PI,AIS,TE for goal->state, input->pred_error, s_pred->s to
#       answer qu's like: how much is there to learn, how much is learned
# TODO: pass pre-generated system into actinf machine, so we can use random robots with parameterized DoF, stiffness, force budget, DoF coupling coefficient

# DONE: modify this code to use the GMM model from actinf_models.py (ca. 20161210)

modes = [
    "m1_goal_error_1d",
    "m1_goal_error_nd",
    "m1_goal_error_nd_e2p",
    "m2_error_nd",
    "m2_error_nd_ext",
    "plot_system",
    "test_models", # simple model test
]

actinf_environments = [
    'simplearm',
    'pointmass1d',
    'pointmass3d',
]
    
def gaussian(m, s, x):
    """univariate gaussian"""
    return 1/(s*np.sqrt(2*np.pi)) * np.exp(-0.5*np.square((m-x)/s))

class ActiveInferenceExperiment(object):
    def __init__(self, mode = "m1_goal_error_nd",
                 model = "knn", numsteps = 1000, idim = None,
                 environment_str = "simplearm",
                 goal_sample_interval = 50, e2pmodel = None,
                 saveplots = False):
        self.mode = mode
        self.model = model

        self.mdl_pkl = "mdl.bin"

        # experiment settings
        self.numsteps = numsteps

        self.eta_fwd_mdl = 0.7
        self.coef_smooth_fast = 0.9
        self.coef_smooth_slow = 0.95
                
        self.saveplots = saveplots

        self.environment_str = environment_str
        
        # intialize robot environment
        if environment_str == "simplearm":
            self.environment = Environment.from_configuration('simple_arm', 'low_dimensional')
            self.environment.noise = 1e-9
            # dimensions
            if mode.startswith("type03_1"):
                self.idim = self.environment.conf.m_ndims
            else:
                self.idim = self.environment.conf.m_ndims * 2
            self.odim = self.environment.conf.m_ndims
            self.dim_ext  = 2 # cartesian
                
        elif environment_str == "pointmass1d":
            self.environment = Environment.from_configuration('pointmass', 'low_dim_vel')
            if mode.startswith("type03_1"):
                self.idim = self.environment.conf.m_ndims
            else:
                self.idim = self.environment.conf.m_ndims * 2
            self.odim = self.environment.conf.m_ndims
            self.dim_ext  = self.environment.conf.m_ndims # cartesian
            
        elif environment_str == "pointmass3d":
            self.environment = Environment.from_configuration('pointmass', 'mid_dim_vel')
            if mode.startswith("type03_1"):
                self.idim = self.environment.conf.m_ndims
            else:
                self.idim = self.environment.conf.m_ndims * 2
            self.odim = self.environment.conf.m_ndims
            self.dim_ext  = self.environment.conf.m_ndims # cartesian
        else:
            print "%s.__init__ unknown environment string '%s', exiting" % (self.__class__.__name__, environment_str)
            import sys
            sys.exit(1)

        print "%s init pass 1: enironment = %s / %s, idim = %d, odim = %d" % (self.__class__.__name__, environment_str, self.environment, self.idim, self.odim)

        # if idim is None:
        #     self.idim = self.environment.conf.m_ndims * 2
        # else:
        #     self.idim = idim
        # self.odim = self.environment.conf.m_ndims
        # exteroceptive dimensionality
        # self.dim_ext  = 2 # cartesian
        # self.dim_prop = 2 # cartesian

        # prepare run_hooks
        self.run_hooks = OrderedDict()
        self.rh_learn_proprio_hooks = OrderedDict()
        print "self.run_hooks", self.run_hooks, self.rh_learn_proprio_hooks
        
        # initialize run method and model
        self.init_wiring(self.mode)
        self.init_model (self.model)

        # sensory space mappings: this are KNN models and just used as data store for X,Y
        self.e2p = ActInfKNN(self.dim_ext, self.odim)
        self.p2e = ActInfKNN(self.odim, self.dim_ext)

        self.init_e2p(e2pmodel)
                
        ################################################################################
        # logging, configure logs dictionary and dict of logging variables
        self.logs = {}
        self.logging_vars = {
            "S_prop_pred":  {"col": self.odim},
            "E_prop_pred":  {"col": self.odim},
            "E_prop_pred_fast": {"col": self.odim},
            "dE_prop_pred_fast": {"col": self.odim},
            "E_prop_pred_slow": {"col": self.odim},
            "dE_prop_pred_slow": {"col": self.odim},
            "d_E_prop_pred_": {"col": self.odim},
            "M_prop_pred":  {"col": self.odim},
            "goal_prop":    {"col": self.odim},
            "S_ext":        {"col": self.dim_ext},
            "E2P_pred":     {"col": self.odim},
            "P2E_pred":     {"col": self.dim_ext},
            "goal_ext":     {"col": self.dim_ext},
            "E_pred_e":     {"col": self.dim_ext},
            "X_":           {"col": -1}, # 6
            "X__":          {"col": -1}, # 6
            "y_":           {"col": -1}  # 3 low-dim
        }
        for k, v in self.logging_vars.items():
            if v["col"] > 0:
                setattr(self, k, np.zeros((1, v["col"])))          # single timestep
                self.logs[k] = np.zeros((self.numsteps, v["col"])) # all timesteps
            else:
                self.logs[k] = []
                # setattr(self, k, [])
            
        ################################################################################
        # initialize vars with special needs
        self.goal_sample_interval = goal_sample_interval
        self.goal_sample_time     = 0
        self.goal_prop = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.odim))
        self.goal_prop_tm1 = np.zeros_like(self.goal_prop)
        # self.j = np.zeros((1, self.odim))
        self.M_prop_pred = np.zeros((1, self.odim))
        self.E_prop_pred = self.M_prop_pred - self.goal_prop
        self.E_prop_pred_fast = self.M_prop_pred.copy()
        self.dE_prop_pred_fast = self.M_prop_pred.copy()
        self.d_E_prop_pred_ = self.M_prop_pred.copy()
        self.E_prop_pred_slow = self.M_prop_pred.copy()
        self.dE_prop_pred_slow = self.M_prop_pred.copy()
        self.S_prop_pred = np.random.normal(0, 0.01, (1, self.odim))

    def init_e2p(self, e2pmodel):
        """ActiveInferenceExperiment.init_e2p

        Initialize the extero-to-proprio mapping
        """
        # sum up all variable dimensions
        mmdims = self.dim_ext + self.odim
        if e2pmodel == "gmm": # use gaussian mixture model
            self.mm = ActInfGMM(idim = self.dim_ext, odim = self.odim)
        elif e2pmodel == "som": # use hebbian SOM model
            self.mm = ActInfHebbianSOM(idim = self.dim_ext, odim = self.odim)
        else:
            print "unknown e2pmodel %s" % e2pmodel
        
    def init_wiring(self, mode):
        """ActiveInferenceExperiment.init_wiring

        Initialize the structure of the experiment specified by the 'mode'.

        The experiment consists of a 'run' on the outside that executes
        all functions registered in self.run_hooks once and in sequence.

        One special hook is the rh_learn_proprio function which consists of a
        list of hooked functions implementing the particular learning setup. The
        rh_learn_proprio function is looped over numstep times.
        """
        
        if mode == "m1_goal_error_nd":
            """active inference / predictive coding

            most basic and first working version, learning in proprioceptive
            space only

            1. sample proprio goal ->
            2. make proprio goal/state prediction ->
            3. measure the proprio goal/state error ->
            4. update the forward model / prediction towards reducing the error,
               assuming monotonic response function
            """

            # learning loop hooks
            self.rh_learn_proprio_hooks["hook01"] = self.lh_learn_proprio_base_0
            self.rh_learn_proprio_hooks["hook02"] = self.lh_learn_proprio_e2p2e
            self.rh_learn_proprio_hooks["hook03"] = self.lh_do_logging
            self.rh_learn_proprio_hooks["hook04"] = self.lh_sample_discrete_uniform_goal

            # experiment loop hooks
            # self.run_hooks["hook00"] = self.make_plot_system_function_and_exec # sweep system before learning
            self.run_hooks["hook01"] = partial(self.rh_learn_proprio, iter_start = 0, iter_end = self.numsteps)
            self.run_hooks["hook02"] = self.rh_learn_proprio_save
            # wip: make it work again
            self.run_hooks["hook03"] = self.rh_check_for_model_and_map
            self.run_hooks["hook99"] = self.experiment_plot

        elif mode == "m2_error_nd":
            """active inference / predictive coding: most basic version (?), proprioceptive only
        
            just goal/state error -> mdl -> goal/state error prediction -> goal/state error -> update forward model"""

            self.rh_learn_proprio_hooks["hook01"] = self.lh_sample_discrete_uniform_goal
            self.rh_learn_proprio_hooks["hook02"] = self.lh_learn_proprio_base_1
            self.rh_learn_proprio_hooks["hook03"] = self.lh_do_logging
            
            self.run_hooks["hook01"] = self.rh_learn_proprio_init_1
            self.run_hooks["hook02"] = partial(self.rh_learn_proprio, iter_start = 0, iter_end = self.numsteps)
            self.run_hooks["hook03"] = self.rh_learn_proprio_save
            # self.run_hooks["hook04"] = self.rh_check_for_model_and_map
            self.run_hooks["hook99"] = self.experiment_plot
                        
        elif mode == "m1_goal_error_nd_e2p":
            """run basic proprio learning, record extero/proprio data, fit probabilistic / multivalued model 'mm'
            to e/p data, then drive the trained proprio model with exteroceptive goals that
            get translated to proprio goals using 'mm'"""
            
            # learning loop hooks
            self.rh_learn_proprio_hooks["hook01"] = self.lh_learn_proprio_base_0
            self.rh_learn_proprio_hooks["hook02"] = self.lh_learn_proprio_e2p2e
            self.rh_learn_proprio_hooks["hook03"] = self.lh_do_logging
            self.rh_learn_proprio_hooks["hook04"] = self.lh_sample_discrete_uniform_goal
            
            # experiment loop hooks
            self.run_hooks["hook01"] = partial(self.rh_learn_proprio, iter_start = 0, iter_end = self.numsteps/2)
            self.run_hooks["hook02"] = self.rh_learn_proprio_save
            self.run_hooks["hook03"] = self.rh_e2p_fit
            self.run_hooks["hook04"] = self.rh_e2p_sample
            self.run_hooks["hook05"] = self.rh_e2p_sample_plot
            self.run_hooks["hook06"] = self.rh_e2p_change_goal_sampling
            # self.run_hooks["hook06"] = partial(self.rh_e2p_sample_and_drive, iter_start = self.numsteps/2, iter_end = self.numsteps)
            self.run_hooks["hook07"] = partial(self.rh_learn_proprio, iter_start = self.numsteps/2, iter_end = self.numsteps)
            self.run_hooks["hook08"] = self.rh_e2p_sample_and_drive_plot
            self.run_hooks["hook99"] = self.experiment_plot
            
        elif mode == "m2_error_nd_ext":
            """active inference / predictive coding

            most basic version, proprioceptive only

            goal -> goal state prediction -> goal/state error -> update forward model
            """

            # only works for pointmass?
            assert isinstance(self.environment, PointmassEnvironment), "Need PointmassEnvironment, not %s" % (self.environment,)
            
            # pimp environment
            self.environment.motor_aberration["type"] = "linsin"
            self.environment.motor_aberration["coef"] = 3.0 # np.random.uniform(-1.0, 1.0, self.odim) # -0.7
            
            # learning loop hooks
            self.rh_learn_proprio_hooks["hook04"] = self.lh_sample_discrete_uniform_goal
            self.rh_learn_proprio_hooks["hook01"] = self.lh_learn_proprio_base_2
            self.rh_learn_proprio_hooks["hook02"] = self.lh_learn_proprio_e2p2e
            self.rh_learn_proprio_hooks["hook03"] = self.lh_do_logging
            # self.rh_learn_proprio_hooks["hook05"] = self.lh_sample_error_gradient

            # experiment loop hooks
            self.run_hooks["hook00"] = self.make_plot_system_function_and_exec # sweep system before learning
            self.run_hooks["hook01"] = partial(self.rh_learn_proprio, iter_start = 0, iter_end = self.numsteps)
            self.run_hooks["hook02"] = self.rh_learn_proprio_save
            # self.run_hooks["hook03"] = self.rh_check_for_model_and_map
            self.run_hooks["hook98"] = self.experiment_plot
            self.run_hooks["hook99"] = self.make_plot_model_function_and_exec # sweep model after learning
            
        elif mode == "m1_goal_error_1d":
            """Experiment: Basic operation M1 on 1-dimensional data"""

            # learning loop hooks
            self.rh_learn_proprio_hooks["hook00"] = self.lh_sample_discrete_uniform_goal
            self.rh_learn_proprio_hooks["hook01"] = self.lh_learn_proprio_base_0
            self.rh_learn_proprio_hooks["hook02"] = self.lh_learn_proprio_e2p2e
            self.rh_learn_proprio_hooks["hook03"] = self.lh_do_logging

            # experiment loop hooks
            self.run_hooks["hook00"] = self.make_plot_system_function_and_exec # sweep system before learning
            self.run_hooks["hook01"] = partial(self.rh_learn_proprio, iter_start = 0, iter_end = self.numsteps/2)
            self.run_hooks["hook02"] = self.rh_learn_proprio_save
            self.run_hooks["hook03"] = self.make_plot_model_function_and_exec # sweep model after learning
            # self.run_hooks["hook03"] = self.rh_check_for_model_and_map
            self.run_hooks["hook05"] = partial(self.rh_learn_proprio, iter_start = self.numsteps/2, iter_end = self.numsteps)
            self.run_hooks["hook06"] = self.rh_learn_proprio_save
            self.run_hooks["hook07"] = self.experiment_plot_basic # plot experiment timeseries illustrative of operation
            self.run_hooks["hook99"] = self.make_plot_model_function_and_exec # sweep model after learning
                        
        elif mode == "plot_system":
            # wow, functools
            self.make_plot_system_function_and_exec()
                        
        else:
            print "FAIL: unknown mode, choose from %s" % (", ".join(modes))
            sys.exit(1)

    def make_plot_system_function_and_exec(self):
        """create specific dictionary of functions to be passed to composition"""
        funcdict = OrderedDict()
        # create sweep input data
        funcdict["hook01"] = self.rh_system_generate_sweep_input
        # sweep and create output data
        funcdict["hook02"] = self.rh_system_sweep
        # plot result
        funcdict["hook03"] = self.rh_system_plot

        f = self.make_function_from_hooks(funcdict)
        f(0)
        # return f
            
    def make_plot_model_function_and_exec(self):
        """create specific dictionary of functions to be passed to composition"""
        assert hasattr(self, "mdl")
        
        funcdict = OrderedDict()
        # create sweep input data
        funcdict["hook01"] = self.rh_model_sweep_generate_input_grid_a
        # sweep and create output data
        funcdict["hook02"] = self.rh_model_sweep
        # plot result
        funcdict["hook03"] = self.rh_model_plot

        f = self.make_function_from_hooks(funcdict)
        f(0)
        # return f
        
    def make_function_from_hooks(self, hookdict):
        """return a single function composed of hooks in the dictionary (ordered), gleaned from
        https://mathieularose.com/function-composition-in-python/"""
        # print dir(hookdict)
        # for k in hookdict.keys():
        #     v = hookdict[k]
        #     print "make_function_from_hooks k = %s, v = %s" % (k, v)
        # return functools.reduce(lambda f, g: lambda x: f(g(x)), hookdict.values(), lambda x: x)
        myfuncs = hookdict.values()
        myfuncs.reverse()
        return reduce(lambda f, g: lambda x: f(g()), myfuncs, lambda x: x)

    def rh_e2p_change_goal_sampling(self):
        self.rh_learn_proprio_hooks["hook04"] = self.sample_discrete_from_extero
        # if hasattr(self.mm, "set_learning_rate_constant"):
        #     self.mm.set_learning_rate_constant(0.0)
        
            
    def init_model(self, model):
        """initialize sensorimotor forward model"""
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
            print "unknown model, FAIL, exiting"
            import sys
            sys.exit(1)

    def attr_check(self, attrs):
        """check if object has all attributes given in attrs array"""
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

    def lh_do_logging(self, i):
        """do logging step: append single timestep versions of variable to logging array at position i"""
        for k, v in self.logging_vars.items():
            # print "key = %s" % (k)
            if k in ["X__"]:
                pass
            elif k in ["X_", "y_"]:
                self.logs[k].append(getattr(self, k)[0,:])
            else:
                self.logs[k][i] = getattr(self, k)
            
    def load_run_data(self):
        """load previous run stored as pickles"""
        # self.mdl  = cPickle.load(open(self.mdl_pkl, "rb"))
        self.mdl  = self.mdl.load(self.mdl_pkl)
        self.logs = cPickle.load(open("logs.bin", "rb"))
        self.e2p  = cPickle.load(open("e2p.bin", "rb"))
        self.p2e  = cPickle.load(open("p2e.bin", "rb"))
    
    ################################################################################
    # hooks

    # system sweep hooks
    def rh_system_generate_sweep_input(self):
        """ActiveInferenceExperiment.rh_system_generate_sweep_input

        generate system inputs on a grid to sweep the system
        """
        
        # create meshgrid over proprio dimensions
        sweepsteps = 21 # 11
        dim_axes = [np.linspace(self.environment.conf.m_mins[i], self.environment.conf.m_maxs[i], sweepsteps) for i in range(self.environment.conf.m_ndims)]
        full_axes = np.meshgrid(*tuple(dim_axes), indexing='ij')

        # print "dim_axes", dim_axes
        # print "full_axes", len(full_axes)
        # print "full_axes", full_axes

        for i in range(len(full_axes)):
            print i, full_axes[i].shape
            print i, full_axes[i].flatten()

        # return proxy
        self.X_system_sweep = np.vstack([full_axes[i].flatten() for i in range(len(full_axes))]).T

    def rh_system_sweep(self):
        """sweep the system by activating it on input grid"""
        assert hasattr(self, "X_system_sweep")
        self.Y_system_sweep = self.environment.compute_motor_command(self.X_system_sweep)
        
    def rh_system_plot(self):
        """prepare and plot system outputs over input variations from sweep"""
        assert hasattr(self, "X_system_sweep")
        assert hasattr(self, "Y_system_sweep")

        print "%s.rh_plot_system sweepsteps = %d" % (self.__class__.__name__, self.X_system_sweep.shape[0])
        print "%s.rh_plot_system environment = %s" % (self.__class__.__name__, self.environment)
        print "%s.rh_plot_system environment proprio dims = %d" % (self.__class__.__name__, self.environment.conf.m_ndims)
        
        scatter_data_raw   = np.hstack((self.X_system_sweep, self.Y_system_sweep))
        scatter_data_cols  = ["X%d" % i for i in range(self.X_system_sweep.shape[1])]
        scatter_data_cols += ["Y%d" % i for i in range(self.Y_system_sweep.shape[1])]
        print "scatter_data_raw", scatter_data_raw.shape
        # df = pd.DataFrame(scatter_data_raw, columns=["x_%d" % i for i in range(scatter_data_raw.shape[1])])
        df = pd.DataFrame(scatter_data_raw, columns=scatter_data_cols)

        title = "%s: i/o behvaiour for %s, in = X, out = Y" % (self.mode, self.environment_str,)
        
        # plot_scattermatrix(df)
        plot_scattermatrix_reduced(df, title = title)

    ################################################################################
    # model sweep hooks
    
        
    # map a model
    def rh_check_for_model_and_map(self):
        if os.path.exists(self.mdl_pkl):
            self.load_run_data()
            print "%s.rh_check_for_model_and_map\n    loaded mdl = %s with idim = %d, odim = %d" % (self.__class__.__name__, self.mdl, self.mdl.idim, self.mdl.odim)
            
            self.map_model_m1_goal_error_nd()
            return

    # proprio learning base loop
    def rh_learn_proprio(self, iter_start = 0, iter_end = 1000):
        # hook: load_run_data
        if iter_start == 0 and os.path.exists(self.mdl_pkl):
            print "found trained model at %s, skipping learning and using that" % self.mdl_pkl
            # load data from previous run
            self.load_run_data()
            return

        # current outermost experiment loop
        for i in range(iter_start, iter_end):
            for k, v in self.rh_learn_proprio_hooks.items():
                # print "k = %s, v = %s" % (k, v)
                v(i)

    ################################################################################
    # proprio learning model variant 2 using more of prediction error
    def lh_learn_proprio_base_2(self, i):
        """ActiveInferenceExperiment.lh_learn_proprio_base_2

        Modified proprio learning hook using state prediction error model M1
        with additional gradient sampling aronud the current working point
        to enable learning of non-monotonic functions
        """
        
        assert self.goal_prop is not None, "self.goal_prop at iter = %d is None, should by ndarray" % i
        assert self.goal_prop.shape == (1, self.odim), "self.goal_prop.shape is wrong, should be %s" % (1, self.odim)
                
        # prepare model input X as goal and prediction error
        self.X_ = np.hstack((self.goal_prop, self.E_prop_pred))

        # predict next state in proprioceptive space
        self.S_prop_pred = self.mdl.predict(self.X_)

        # inverse model / motor primitive / reflex arc
        self.M_prop_pred = self.environment.compute_motor_command(self.S_prop_pred)
        
        # distort response
        # self.M_prop_pred = np.sin(self.M_prop_pred * np.pi) # * 1.333
        # self.M_prop_pred = np.exp(self.M_prop_pred) - 1.0 # * 1.333
        # self.M_prop_pred = (gaussian(0, 0.5, self.M_prop_pred) - 0.4) * 5

        # add noise
        self.M_prop_pred += np.random.normal(0, 0.01, self.M_prop_pred.shape)

        # sample error gradient
        numsamples = 20
        # was @ 50
        if i % 1 == 0:
            from sklearn import linear_model
            import sklearn
            from sklearn import kernel_ridge
            
            lm = linear_model.Ridge(alpha = 0.0)
            
            S_ = []
            M_ = []
            for i in range(numsamples):
                # S_.append(np.random.normal(self.S_prop_pred, 0.01 * self.environment.conf.m_maxs, self.S_prop_pred.shape))
                # larger sampling range
                S_.append(np.random.normal(self.S_prop_pred, 0.3 * self.environment.conf.m_maxs, self.S_prop_pred.shape))
                # print "S_[-1]", S_[-1]
                M_.append(self.environment.compute_motor_command(S_[-1]))
                S_ext_ = self.environment.compute_sensori_effect(M_[-1]).reshape((1, self.dim_ext))
            S_ = np.array(S_).reshape((numsamples, self.S_prop_pred.shape[1]))
            M_ = np.array(M_).reshape((numsamples, self.S_prop_pred.shape[1]))
            print "S_", S_.shape, "M_", M_.shape
            # print "S_", S_, "M_", M_


            lm.fit(S_, M_)
            self.grad = np.diag(lm.coef_)
            print "grad", np.sign(self.grad), self.grad
            
            # pl.plot(S_, M_, "ko", alpha=0.4)
            # pl.show()

            
        
        self.prediction_errors_extended()
            
        # # prediction error's variant
        # self.E_prop_pred_state = self.S_prop_pred - self.M_prop_pred
        # self.E_prop_pred = self.E_prop_pred_state
        
        # execute command propagating effect through system, body + environment
        self.S_ext = self.environment.compute_sensori_effect(self.M_prop_pred.T).reshape((1, self.dim_ext))
        # self.environment.plot_arm()
        
        # compute target for the prediction error driven forward model
        # if i % 10 == 0: # play with decreased update rates
        # self.y_ = self.S_prop_pred - (self.E_prop_pred * self.eta_fwd_mdl) - self.E_prop_pred_state * (self.eta_fwd_mdl/2.0)
        # modulator = self.grad
        modulator = np.sign(self.grad)
        print "modulator", modulator
        self.y_ = self.S_prop_pred - (self.E_prop_pred * self.eta_fwd_mdl * modulator)
        # modulator = -np.sign(self.dE_prop_pred_fast / -E_prop_pred_tm1)
        # self.y_ = self.S_prop_pred - (self.E_prop_pred * self.eta_fwd_mdl * modulator)
        # FIXME: what is the target if there is no trivial mapping of the error?
        # FIXME: suppress update when error is small enough (< threshold)
        # print "self.y_", self.y_

        # fit the model
        self.mdl.fit(self.X_, self.y_)

        self.goal_prop_tm1 = self.goal_prop.copy()

    def prediction_errors_extended(self):
        if np.sum(np.abs(self.goal_prop - self.goal_prop_tm1)) > 1e-2:
            self.E_prop_pred_fast = np.random.uniform(-1e-5, 1e-5, self.E_prop_pred_fast.shape)
            self.E_prop_pred_slow = np.random.uniform(-1e-5, 1e-5, self.E_prop_pred_slow.shape)
            # recompute error
            # self.E_prop_pred = self.M_prop_pred - self.goal_prop
            # self.E_prop_pred[:] = np.random.uniform(-1e-5, 1e-5, self.E_prop_pred.shape)
            #else:            
                
        E_prop_pred_tm1 = self.E_prop_pred.copy()

        # prediction error's
        self.E_prop_pred_state = self.S_prop_pred - self.M_prop_pred
        self.E_prop_pred_goal  = self.M_prop_pred - self.goal_prop
        self.E_prop_pred = self.E_prop_pred_goal
        
        self.E_prop_pred__fast = self.E_prop_pred_fast.copy()
        self.E_prop_pred_fast  = self.coef_smooth_fast * self.E_prop_pred_fast + (1 - self.coef_smooth_fast) * self.E_prop_pred

        self.E_prop_pred__slow = self.E_prop_pred_slow.copy()
        self.E_prop_pred_slow  = self.coef_smooth_slow * self.E_prop_pred_slow + (1 - self.coef_smooth_slow) * self.E_prop_pred
                
        self.dE_prop_pred_fast = self.E_prop_pred_fast - self.E_prop_pred__fast
        self.d_E_prop_pred_ = self.coef_smooth_slow * self.d_E_prop_pred_ + (1 - self.coef_smooth_slow) * self.dE_prop_pred_fast
        
    ################################################################################
    # proprio learning model variant 0
    def lh_learn_proprio_base_0(self, i):
        """ActiveInferenceExperiment.lh_learn_proprio_base_0

        Basic proprio learning hook using goal prediction error model M1
        """
        
        assert self.goal_prop is not None, "self.goal_prop at iter = %d is None, should be ndarray" % i
        assert self.goal_prop.shape == (1, self.odim), "self.goal_prop.shape %s is wrong, should be %s" % (self.goal_prop.shape, (1, self.odim))

        # prepare model input X as goal and prediction error
        self.X_ = np.hstack((self.goal_prop, self.E_prop_pred))

        # predict next state in proprioceptive space
        self.S_prop_pred = self.mdl.predict(self.X_)

        # inverse model / motor primitive / reflex arc
        self.M_prop_pred = self.environment.compute_motor_command(self.S_prop_pred)
        
        # distort response
        # self.M_prop_pred = np.sin(self.M_prop_pred * np.pi) # * 1.333
        # self.M_prop_pred = np.exp(self.M_prop_pred) - 1.0 # * 1.333
        # self.M_prop_pred = (gaussian(0, 0.5, self.M_prop_pred) - 0.4) * 5
        
        # add noise
        self.M_prop_pred += np.random.normal(0, 0.01, self.M_prop_pred.shape)

        # prediction error's
        self.E_prop_pred_goal  = self.M_prop_pred - self.goal_prop
        self.E_prop_pred = self.E_prop_pred_goal

        self.prediction_errors_extended()            
        
        # # prediction error's variant
        # self.E_prop_pred_state = self.S_prop_pred - self.M_prop_pred
        # self.E_prop_pred = self.E_prop_pred_state
        
        # execute command propagating effect through system, body + environment
        self.S_ext = self.environment.compute_sensori_effect(self.M_prop_pred.T).reshape((1, self.dim_ext))
        # self.environment.plot_arm()
        
        # compute target for the prediction error driven forward model
        # if i % 10 == 0: # play with decreased update rates
        self.y_ = self.S_prop_pred - (self.E_prop_pred * self.eta_fwd_mdl)
        # FIXME: suppress update when error is small enough (< threshold)

        # fit the model
        self.mdl.fit(self.X_, self.y_)

        self.goal_prop_tm1 = self.goal_prop.copy()

    ################################################################################
    # proprio learning model variant 1
    def rh_learn_proprio_init_1(self):
        print("%s.rh_learn_proprio_init_1 self.E_prop_pred.shape = %s, self.idim = %d" % (self.__class__.__name__, self.E_prop_pred.shape, self.idim))
        if not hasattr(self, "X_"): self.X_ = np.hstack((self.E_prop_pred)).reshape((1, self.idim)) # initialize model input
        
    def lh_learn_proprio_base_1(self, i):
        """ActiveInferenceExperiment.lh_learn_proprio_base_1

        Modified proprio learning hook using state prediction error model M2
        """
        
        # create a new proprioceptive state, M_prop is the motor state, S_prop is the incremental change
        self.M_prop_pred = self.environment.compute_motor_command(self.M_prop_pred + self.S_prop_pred) #
            
        # 2a. optionally distort response
        # self.M_prop_pred = np.sin(self.M_prop_pred * np.pi/1.95) # * 1.333
        # self.M_prop_pred = np.exp(self.M_prop_pred) - 1.0 # * 1.333
        # self.M_prop_pred = (gaussian(0, 0.5, self.M_prop_pred) - 0.4) * 5

        # 2b. add noise
        self.M_prop_pred += np.random.normal(0, 0.01, self.M_prop_pred.shape)

        # execute command, compute exteroceptive sensory effect
        self.S_ext = self.environment.compute_sensori_effect(self.M_prop_pred.T)
        
        # compute error as state prediction minus last goal
        self.E_prop_pred = self.M_prop_pred - self.goal_prop_tm1 # self.E_prop_pred_goal

        # compute forward model target from error
        self.y_ = -self.E_prop_pred * self.eta_fwd_mdl # i am amazed this works
        # FIXME: what is the target if there is no trivial mapping of the error?

        # fit the forward model
        self.mdl.fit(self.X_, self.y_)

        # prepare new model input
        if np.sum(np.abs(self.goal_prop - self.goal_prop_tm1)) > 1e-6:
            # goal changed
            self.X_ = np.hstack((self.M_prop_pred - self.goal_prop)).reshape((1, self.idim)) # model input: just prediction error
        else:
            # goal unchanged
            self.X_ = np.hstack((self.E_prop_pred)).reshape((1, self.idim)) # model input: just prediction error

        self.prediction_errors_extended()            

        # compute new prediction
        self.S_prop_pred = self.mdl.predict(self.X_) # state prediction

        # store last goal g_{t-1}
        self.goal_prop_tm1 = self.goal_prop.copy()

    ################################################################################
    # some utils
    def lh_learn_proprio_e2p2e(self, i):
        # hook: learn e2p, p2e mappings
        # extero/proprio mapping predict / fit
        self.E2P_pred = self.e2p.predict(self.S_ext)
        self.P2E_pred = self.p2e.predict(self.M_prop_pred)

        self.e2p.fit(self.S_ext, self.M_prop_pred)
        self.p2e.fit(self.M_prop_pred.reshape((1, self.odim)), self.S_ext)

        self.E_pred_e = self.S_ext - self.goal_ext

    def lh_sample_discrete_uniform_goal(self, i):
        # discrete goal
        # hook: goal sampling
        if i % self.goal_sample_interval == 0:
            self.goal_prop = np.random.uniform(self.environment.conf.m_mins * 0.95, self.environment.conf.m_maxs * 0.95, (1, self.odim))
            print "new goal[%d] = %s" % (i, self.goal_prop)
            print "e_pred = %f" % (np.linalg.norm(self.E_prop_pred, 2))

    def sample_continuos_goal_sine(self, i):
        # continuous goal
        if i % self.goal_sample_interval == 0:
            w = float(i)/self.numsteps
            f1 = 0.05 # float(i)/10000 + 0.01
            f2 = 0.08 # float(i)/10000 + 0.02
            f3 = 0.1 # float(i)/10000 + 0.03
            self.goal_prop = np.sin(i * np.array([f1, f2, f3])).reshape((1, self.odim))
            print "new goal[%d] = %s" % (i, self.goal_prop)
            print "e_pred = %f" % (np.linalg.norm(self.E_prop_pred, 2))
            
    def sample_discrete_from_extero(self, i):
        self.mm.fit(self.S_ext, self.M_prop_pred)
        ext_err = np.sum(np.abs(self.goal_ext - self.S_ext))
        if i % self.goal_sample_interval == 0 or \
            ((i - self.goal_sample_time) > 5 and ext_err > 0.1):
            # update e2p
            EP = np.hstack((np.asarray(self.e2p.X_), np.asarray(self.e2p.y_)))
            # print "EP[%d] = %s" % (i, EP)
            EP = EP[10:] # knn bootstrapping creates additional datapoints
            # if i % 100 == 0:
            # re-fit gmm e2p
            # self.mm.fit(np.asarray(self.e2p.X_)[10:], np.asarray(self.e2p.y_)[10:])
            # self.mm.fit(np.asarray(self.e2p.X_)[10:], np.asarray(self.e2p.y_)[10:])
                
            # print "EP, cen_lst, cov_lst, p_k, logL", EP, self.cen_lst, self.cov_lst, self.p_k, self.logL
            ref_interval = 1
            self.cond = EP[(i+ref_interval) % EP.shape[0]] # X_[i,:3]
            self.cond[2:] = np.nan
            self.cond_ = np.random.uniform(-1, 1, (5, ))
            # randomly fetch an exteroceptive state that we have seen already (= reachable)
            self.goal_ext = EP[np.random.choice(range(self.numsteps/2)),:2].reshape((1, self.dim_ext))
            # self.cond_[:2] = self.goal_ext
            # self.cond_[2:] = np.nan
            # print "self.cond", self.cond
            # print "self.cond_", self.cond_

            # predict proprioceptive goal from exteroceptive one
            # if hasattr(self.mm, "cen_lst"):
            #     self.goal_prop = self.mm.sample(self.cond_)
            # else:
            #     self.goal_prop = self.mm.sample(self.goal_ext)
            self.goal_prop = self.mm.sample(self.goal_ext)

            self.goal_sample_time     = i
                            
            # (cen_con, cov_con, new_p_k) = gmm.cond_dist(self.cond_, self.cen_lst, self.cov_lst, self.p_k)
            # self.goal_prop = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
            
            # # discrete goal
            # self.goal_prop = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.odim))
            print "new goal_prop[%d] = %s" % (i, self.goal_prop)
            print "    goal_ext[%d] = %s" % (i, self.goal_ext)
            print "e_pred = %f" % (np.linalg.norm(self.E_prop_pred, 2))
            print "ext_er = %f" % (ext_err)

    # def lh_sample_error_gradient(self, i):
    #     """sample the local error gradient"""
    #     # hook: goal sampling
    #     if i % self.goal_sample_interval == 0:
    #         self.goal_prop = np.random.uniform(self.environment.conf.m_mins * 0.95, self.environment.conf.m_maxs * 0.95, (1, self.odim))
    #         print "new goal[%d] = %s" % (i, self.goal_prop)
    #         print "e_pred = %f" % (np.linalg.norm(self.E_prop_pred, 2))

    def rh_learn_proprio_save(self):
        """save data from proprio learning"""
        if not self.attr_check(["logs", "mdl", "mdl_pkl"]):
            return

        # already loaded all data
        if os.path.exists(self.mdl_pkl):
            return
            
        # cPickle.dump(self.mdl, open(self.mdl_pkl, "wb"))
        self.mdl.save(self.mdl_pkl)

        # convert to numpy array
        self.logs["X__"] = np.asarray(self.logs["X_"])
        # np.save("X_.npy", self.logs["X_"])
        
        self.logs["EP"] = np.hstack((np.asarray(self.e2p.X_), np.asarray(self.e2p.y_)))
        # if mdl is type knn?
        self.logs["EP"] = self.logs["EP"][10:]
        # print "self.logs[\"EP\"]", type(self.logs["EP"]), self.logs["EP"].shape, self.logs["EP"]
        print "self.logs[\"EP\"].shape = %s, %s" % (self.logs["EP"].shape, self.logs["X__"].shape)
        # print "%d self.logs["EP"].shape = %s".format((0, self.logs["EP"].shape))

        # np.save("EP.npy", self.logs["EP"])

        cPickle.dump(self.logs, open("logs.bin", "wb"))

        cPickle.dump(self.e2p, open("e2p.bin", "wb"))
        cPickle.dump(self.p2e, open("p2e.bin", "wb"))

        # pl.plot(EP[:,:2])
        # pl.show()

    def rh_e2p_fit(self):
        """Initial fit of e2p map with a batch of data"""
        
        # 2. now we learn e2p mapping (conditional joint density model for dealing with ambiguity)
        # ## prepare data
        if not self.attr_check(["logs", "e2p"]):
            return

        # print self.logs["EP"].shape, self.logs["X_"].shape
        # pl.ioff()
        # pl.plot(self.logs["X_"])
        # pl.show()

        # print "self.logs['X_']", self.logs["X_"]

        print("%s.rh_e2p_fit batch fitting of e2p (%s)" % (self.__class__.__name__, self.mm.__class__.__name__))
        self.mm.fit(np.asarray(self.e2p.X_)[10:], np.asarray(self.e2p.y_)[10:])
        
        # # fit gmm
        # self.cen_lst, self.cov_lst, self.p_k, self.logL = gmm.em_gm(self.logs["EP"], K = 10, max_iter = 1000,\
        #     verbose = False, iter_call = None)
        # print "rh_e2p_fit gmm: Log likelihood (how well the data fits the model) = ", self.logL
        # # print "rh_e2p_fit gmm:", np.array(self.cen_lst).shape, np.array(self.cov_lst).shape, self.p_k.shape

    def rh_e2p_sample(self):
        """sample the probabilistic e2p model for the entire dataset"""
        # intro checks
        if not self.attr_check(["logs"]):
            return

        self.y_samples, self.y_samples_ = self.mm.sample_batch_legacy(X = self.logs["EP"], cond_dims = [0, 1], out_dims = [2,3,4], resample_interval = self.goal_sample_interval)
                
    def rh_e2p_sample_plot(self):
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
        ax2.plot(self.logs["M_prop_pred"], "k-", label="S_p")
        # pl.plot(self.logs["E2P_pred"], "y-", label="E2P knn")
        ax2.plot(self.y_samples, "g-", label="E2P gmm cond", alpha=0.8, linewidth=2)
        ax2.plot(self.logs["X__"][:,:3], "r-", label="goal goal")
        for _ in self.y_samples_:
            plausibility = _ - self.logs["X__"][:,:3]
            # print "_.shape = %s, plausibility.shape = %s, %d" % (_.shape, plausibility.shape, 0)
            # print "_", np.sum(_), _ - self.logs["X__"][:,:3]
            plausibility_norm = np.linalg.norm(plausibility, 2, axis=1)
            print "plausibility = %f" % (np.mean(plausibility_norm))
            if np.mean(plausibility_norm) < 0.8: # FIXME: what is that for, for thinning out the number of samples?
                ax2.plot(_, "b.", label="E2P gmm samples", alpha=0.2)
        handles, labels = ax2.get_legend_handles_labels()
        print "handles, labels", handles, labels
        legidx = slice(0, 12, 3)
        ax2.legend(handles[legidx], labels[legidx])
        # ax.legend(handles=[handles[i] for i in [0, 2]],
        #           labels=[labels[i] for i in [0, 2]])
        pl.show()

    def rh_e2p_sample_and_drive_plot(self):
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
        """ActiveInferenceExperiment.run

        run method iterates the dictionary of hooks and executes each
        """
        for k, v in self.run_hooks.items():
            print "key = %s, value = %s" % (k, v)
            # execute value which is a function pointer
            v()
        
    def rh_model_sweep_generate_input_random(self):
        return None

    def rh_model_sweep_generate_input_grid_a(self):
        sweepsteps = 11 # 21
        # extero config
        dim_axes = [np.linspace(self.environment.conf.s_mins[i], self.environment.conf.s_maxs[i], sweepsteps) for i in range(self.environment.conf.s_ndims)]
        # dim_axes = [np.linspace(self.environment.conf.s_mins[i], self.environment.conf.s_maxs[i], sweepsteps) for i in range(self.mdl.idim)]
        print "rh_model_sweep_generate_input_grid: s_ndims = %d, dim_axes = %s" % (self.environment.conf.s_ndims, dim_axes,)
        full_axes = np.meshgrid(*tuple(dim_axes), indexing='ij')
        print "rh_model_sweep_generate_input_grid: full_axes = %s, %s" % (len(full_axes), full_axes,)

        for i in range(len(full_axes)):
            print i, full_axes[i].shape
            print i, full_axes[i].flatten()

        # return proxy
        error_grid = np.vstack([full_axes[i].flatten() for i in range(len(full_axes))])
        print "error_grid", error_grid.shape

        # draw state / goal configurations
        X_accum = []
        states = np.linspace(-1, 1, sweepsteps)
        # for state in range(1): # sweepsteps):
        for state in states:
            # randomize initial position
            # self.M_prop_pred = self.environment.compute_motor_command(np.random.uniform(-1.0, 1.0, (1, self.odim)))
            # draw random goal and keep it fixed
            # self.goal_prop = self.environment.compute_motor_command(np.random.uniform(-1.0, 1.0, (1, self.odim)))
            self.goal_prop = self.environment.compute_motor_command(np.ones((1, self.odim)) * state)
            # self.goal_prop = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.odim))
            GOALS = np.repeat(self.goal_prop, error_grid.shape[1], axis = 0) # as many goals as error components
            # FIXME: hacks for M1/M2
            if self.mdl.idim == 3:
                X = GOALS
            elif self.mdl.idim == 6:
                X = np.hstack((GOALS, error_grid.T))
            else:
                X = np.hstack((GOALS, error_grid.T))
            X_accum.append(X)

        X_accum = np.array(X_accum)

        # don't need this?
        # X_accum = X_accum.reshape((X_accum.shape[0] * X_accum.shape[1], X_accum.shape[2]))
        
        print "X_accum.shape = %s, mdl.idim = %d, mdl.odim = %d" % (X_accum.shape, self.mdl.idim, self.mdl.odim)
        # print X_accum
        X = X_accum
        # X's and pred's indices now mean: slowest: goal, e1, e2, fastest: e3
        self.X_model_sweep = X.copy()
        return X
        # print "self.X_model_sweep.shape", self.X_model_sweep.shape

        
    def rh_model_sweep_generate_input_grid(self):
        """ActiveInferenceExperiment.rh_model_sweep_generate_input_grid

        generate a meshgrid that sweeps a model's input space

        See also: rh_model_sweep_generate_input_random"""
        
        # grid resolution
        # sweepsteps = 5 # 11 # 21
        sweepsteps = 11 # 21

        print "%s.map_model_m1_goal_error_nd self.mdl.idim = %d, self.mdl.odim = %s" % (self.__class__.__name__, self.mdl.idim, self.mdl.odim)

        # construct linear axes for input sweep, goal assumed equal odim / motor dim
        # x_ = [np.linspace(-1., 1., sweepsteps) for i in range(self.mdl.odim))

        # get meshgrid from linear axes
        # x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        
        x_ = np.linspace(-1., 1., sweepsteps)
        y_ = np.linspace(-1., 1., sweepsteps)
        z_ = np.linspace(-1., 1., sweepsteps)

        # meshgrid from axes and resolution
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        # print "x.shape", x.shape
        # print "x", x.flatten()
        # print "y", y.flatten()
        # print "z", z.flatten()
        
        # combined grid
        error_grid = np.vstack((x.flatten(), y.flatten(), z.flatten()))

        # draw state / goal configurations
        X_accum = []
        states = np.linspace(-1, 1, sweepsteps)
        # for state in range(1): # sweepsteps):
        for state in states:
            # randomize initial position
            # self.M_prop_pred = self.environment.compute_motor_command(np.random.uniform(-1.0, 1.0, (1, self.odim)))
            # draw random goal and keep it fixed
            # self.goal_prop = self.environment.compute_motor_command(np.random.uniform(-1.0, 1.0, (1, self.odim)))
            self.goal_prop = self.environment.compute_motor_command(np.ones((1, self.odim)) * state)
            # self.goal_prop = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.odim))
            GOALS = np.repeat(self.goal_prop, error_grid.shape[1], axis = 0) # as many goals as error components
            # FIXME: hacks for M1/M2
            if self.mdl.idim == 3:
                X = GOALS
            elif self.mdl.idim == 6:
                X = np.hstack((GOALS, error_grid.T))
            else:
                X = np.hstack((GOALS, error_grid.T))
            X_accum.append(X)

        X_accum = np.array(X_accum)

        # don't need this?
        # X_accum = X_accum.reshape((X_accum.shape[0] * X_accum.shape[1], X_accum.shape[2]))
        
        print "X_accum.shape = %s, mdl.idim = %d, mdl.odim = %d" % (X_accum.shape, self.mdl.idim, self.mdl.odim)
        # print X_accum
        X = X_accum
        # X's and pred's indices now mean: slowest: goal, e1, e2, fastest: e3
        self.X_model_sweep = X.copy()
        return X
        # print "self.X_model_sweep.shape", self.X_model_sweep.shape

    def rh_model_sweep(self):
        assert hasattr(self, "X_model_sweep")
        pred = []
        for i in range(self.X_model_sweep.shape[0]): # loop over start states
            print "trying a predict with X[%d] = %s on model %s with idim = %d / odim %d" % (i, self.X_model_sweep.shape, self.mdl, self.mdl.idim, self.mdl.odim)
            # this might go wrong with different models
            pred.append(self.mdl.predict(self.X_model_sweep[i]))
        pred = np.array(pred)
        print "pred.shape", pred.shape
        self.Y_model_sweep = pred
        return pred
                
    def rh_model_plot(self):
        """prepare and plot model outputs over input variations from sweep"""
        assert hasattr(self, "X_model_sweep")
        assert hasattr(self, "Y_model_sweep")

        print "%s.rh_plot_model sweepsteps = %d" % (self.__class__.__name__, self.X_model_sweep.shape[0])
        print "%s.rh_plot_model environment = %s" % (self.__class__.__name__, self.environment)
        print "%s.rh_plot_model environment proprio dims = %d" % (self.__class__.__name__, self.environment.conf.m_ndims)
        
        # scatter_data_raw   = np.hstack((self.X_model_sweep[:,1:], self.Y_model_sweep))
        # scatter_data_cols  = ["X%d" % i for i in range(1, self.X_model_sweep.shape[1])]
        # scatter_data_cols += ["Y%d" % i for i in range(self.Y_model_sweep.shape[1])]
        # print "scatter_data_raw", scatter_data_raw.shape
        # # df = pd.DataFrame(scatter_data_raw, columns=["x_%d" % i for i in range(scatter_data_raw.shape[1])])
        # df = pd.DataFrame(scatter_data_raw, columns=scatter_data_cols)
 
        # plot_scattermatrix(df)
        # plot_scattermatrix_reduced(df)
        plot_colormeshmatrix_reduced(self.X_model_sweep, self.Y_model_sweep, ymin = -1.0, ymax = 1.0)
        
    ################################################################################
    def map_model_m1_goal_error_nd(self):
        """plot model output over model input sweep"""
        from mpl_toolkits.mplot3d import Axes3D
        doplot_scattermatrix = False
        
        # turn off interactive mode from explauto
        pl.ioff()

        # generate model input sweep as meshgrid
        X_accum = self.rh_model_sweep_generate_input_grid()
        print "map_model_m1_goal_error_nd self.X_model_sweep = %s" % (self.X_model_sweep.shape,)

        # execute the model on the sweep meshgrid
        pred = self.rh_model_sweep()
        print "map_model_m1_goal_error_nd self.Y_model_sweep = %s" % (self.Y_model_sweep.shape,)

        # X_accum = self.X_model_sweep
        # pred = self.Y_model_sweep
        numgrid = self.Y_model_sweep.shape[0]
        
        # ############################################################
        # # quiver matrix
        # pl.ioff()
        # numout = 3
        # numoutkombi = [[0, 1], [0, 2], [1, 2]]
        # # numoutmultf = [25, 5, 1]
        # # numoutrange = [125, 25, 5]
        # numoutmultf = [1, 1, 1]
        # numoutrange = [11, 121, 3]
        # # dim x dim matrix: looping over combinations of input and combinations of output
        # for i in range(numout): # looping over first error component
        # # for i in range(numgrid): # looping over first error component
        #     dimsel = numoutkombi[i]
        #     i1 = numoutkombi[i][0]
        #     i2 = numoutkombi[i][1]
        #     xidx = range(0, numoutrange[dimsel[0]], numoutmultf[dimsel[0]])
        #     yidx = range(0, numoutrange[dimsel[1]], numoutmultf[dimsel[1]])
        #     # curdata  = pred[i*25:(i+1)*25].reshape((numgrid, numgrid, -1))
        #     print "pred", pred.shape, "xidx", xidx, "yidx", yidx
        #     curdata  = pred[xidx]
        #     curgoal  = X_accum[0,0,:3]
        #     # curgoal  = X_accum[0,:3]
        #     # curgoal  = X_accum[i*25:(i+1)*25].reshape((numgrid, numgrid, -1))
        #     print "curgoal.shape", curgoal.shape
        #     print "curdata.shape", curdata.shape
        #     curerror = X_accum[i*25:(i+1)*25].reshape((numgrid, numgrid, -1))
        #     print "curerror.shape", curerror.shape
        #     # for j in range(numout): # loop remaining error comps
        #     # pl.subplot(numgrid, numout, (i*numout) + j + 1)
        #     pl.subplot(numgrid, 1, i + 1)
        #     # i1 = numoutkombi[j][0]
        #     # i2 = numoutkombi[j][1]
        #     # print "curerror[:,:,i1+3]", curerror[:,:,i1+3].shape
        #     # pl.plot(curerror[:,:,i1+3], curerror[:,:,i2+3], "ko")
        #     # pl.show()
        #     # xidx = np.range(0, )
        #     # yidx
        #     print "curdata.shape = %s, %d" % (curdata.shape, 0)
        #     if self.mdl.idim == 3:
        #         curerror_idx = [1,2]
        #     else:
        #         curerror_idx = [4,5]
        #     pl.quiver(
        #         # curerror[:,:,i1+3], curerror[:,:,i2+3],
        #         curerror[:,:,curerror_idx[0]], curerror[:,:,curerror_idx[1]],
        #         # curdata[:,:,i1],  curdata[:,:,i2]
        #         curdata[:,1],  curdata[:,2]
        #         )
        #     pl.plot([curgoal[i1]], [curgoal[i2]], "ro")
        # pl.show()
        # # sys.exit()
        
        ############################################################
        # 3D scatter
        pl.ioff()
        fig = pl.figure()
        cols = ["k", "r", "b", "g", "y", "c", "m"] * 10
        axs = [None for i in range(numgrid)]
        for i in range(numgrid):
            print "grid #%d" % i
            # sl = slice(i * (numgrid**3), (i+1) * (numgrid**3))
            sl = slice(i, i+1, None)
            of = 0 # (i*2)
            axs[i] = fig.add_subplot(1, numgrid, i+1, projection='3d')
            # print "sl", sl, "of", of, "ax", axs[i] #, error_grid.shape
            # # axs[i].scatter3D(error_grid.T[sl,0] + of, error_grid.T[sl,1], error_grid.T[sl,2], c=cols[i])
            # # axs[i].scatter3D(X_accum[sl,0] + of, X_accum[sl,1], X_accum[sl,2], c=cols[i])
            # # axs[i].set_title("GOAL = %s, state = %s" % (X[i*125,:3], X[i*125,3:]))
            # # axs[i].set_title("state/GOAL error = %s" % (X[i*125,3:] - X[i*125,:3]))
            axs[i].set_title("GOAL = %s" % (self.X_model_sweep[i,0,:3]), fontsize=8)
            axs[i].scatter3D(pred[sl,:,0] + of, pred[sl,:,1], pred[sl,:,2], c=cols[i], alpha = 0.33)
            # # axs[i].scatter(error_grid.T[sl,0] + of, error_grid.T[sl,1], c=cols[i])
            # axs[i].set_aspect(1.0)
            # axs[i].set_xlabel("d1")
            # axs[i].set_ylabel("d2")
            # axs[i].set_zlabel("d3")
            
            # # pl.subplot(1, numgrid, i+1)
            # # pl.scatter(error_grid.T[sl,0] + of, error_grid.T[sl,1], c=cols[i])
            
        pl.show()

        # scatterplot nd
        # historgram  nd
        # draw a vectorfield?

        # # error_grid = np.concatenate((x, y, z)) # .reshape((numgrid, numgrid, numgrid))
        # print "error_grid", error_grid.shape, error_grid[:]
                
        # # print "x[0,0,0]", x[0,0,[0]],y[0,0,[0]],z[0,0,:]
        # X_ = np.vstack((x[0,0,:],y[0,0,:],z[0,0,:]))
        # print "X_.shape", X_.shape, X_[:,0].shape, X_
        # GOALS = np.repeat(self.goal_prop, numgrid, axis = 0)
        # print "self.goal_prop.shape", self.goal_prop.shape
        # print "GOALS.shape", GOALS.shape
        # X = np.hstack((GOALS, X_.T))
        
        self.E_prop_pred = self.M_prop_pred - self.goal_prop
        # X_s = []
        # X_s.append(np.hstack((self.goal_prop, self.E_prop_pred))) # model input: goal and prediction error
        # X_s.append(np.hstack((self.goal_prop, X_[:,0].reshape((1, 3))))) # model input: goal and prediction error
        # # X_s = X_.tolist()
        # # GOALS = np.repeat
        # X = np.vstack(X_s)

        # # PCA
        # print "self.X_model_sweep.shape", self.X_model_sweep.shape
        # # print "pred", self.mdl.predict(X)
        # pred = self.mdl.predict(self.X_model_sweep)
        # print "pred.shape", pred.shape

        # from sklearn.decomposition import PCA

        # X_pca = PCA(n_components = 2)
        # # X_pca.fit(X)
        # X_pca.fit(error_grid.T)
        # pred_pca = PCA(n_components = 1)
        # pred_pca.fit(pred)

        # # X_red = X_pca.transform(X)
        # X_red = X_pca.transform(error_grid.T)
        # print "X_red.shape", X_red.shape, np.min(X_red, axis=0), np.max(X_red, axis=0)
        # pred_red = pred_pca.transform(pred)
        # print "pred_red", pred_red.shape
        
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
        for i in range(pred.shape[2]):
            pl.subplot(1, pred.shape[2], i+1)
            # p1 = pred[:,i].reshape((numgrid, numgrid, numgrid, numgrid))
            print i, pred.shape, pred[...,i].shape
            p1 = pred[...,i].reshape((numgrid, numgrid, numgrid, numgrid))
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
        
    ################################################################################
    def run_m2_error_nd(self):
        """active inference / predictive coding: most basic version (?), proprioceptive only
        
        just goal/state error -> mdl -> goal/state error prediction -> goal/state error -> update forward model"""

        # FIXME: merge this and goal pred error into common sm-loop structure with different hooks coding for
        # the marco-model itself
                
        for i in range(self.numsteps):
            
            # # 1. pick a goal g_t
            # if True and i % 1 == 0:
            #     # continuous goal
            #     w = float(i)/self.numsteps
            #     f1 = 0.05 # float(i)/10000 + 0.01
            #     f2 = 0.08 # float(i)/10000 + 0.02
            #     f3 = 0.1 # float(i)/10000 + 0.03
            #     self.goal_prop = np.sin(i * np.array([f1, f2, f3])).reshape((1, self.odim))
                
            # if i % 50 == 0:
            #     # discrete goal
            #     self.goal_prop = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.odim))
            #     print "g_[%d] = %s" % (i, self.goal_prop)
            #     print "e_pred[%d] = %f" % (i-1, np.linalg.norm(self.E_prop_pred, 2))

            # rearrange to comply with tapping
            # t has just become t+1

            # 2. new information available / new measurement, s_t
            # inverse model / motor primitive / reflex arc / ...
            self.M_prop_pred = self.environment.compute_motor_command(self.M_prop_pred + self.S_prop_pred) #
            
            # execute command with exteroceptive effect
            # self.S_ext = self.environment.compute_sensori_effect(self.M_prop_pred.T)

            # if draw:
            # self.environment.plot_arm(ax, self.M_prop_pred.T)
            # pl.pause(0.001)

            # print self.S_ext
            # self.M_prop_pred +=

            # 2a. optionally distort response
            self.M_prop_pred = np.sin(self.M_prop_pred * np.pi/1.95) # * 1.333
            # self.M_prop_pred = np.exp(self.M_prop_pred) - 1.0 # * 1.333
            # self.M_prop_pred = (gaussian(0, 0.5, self.M_prop_pred) - 0.4) * 5

            # 2b. add noise
            self.M_prop_pred += np.random.normal(0, 0.01, self.M_prop_pred.shape)

            # execute command with exteroceptive effect
            self.S_ext = self.environment.compute_sensori_effect(self.M_prop_pred.T)
            
            # 3. compute error of measured state s_t with respect to current goal g_t
            #    s_t is called m here for some reason ;)

            # self.E_prop_pred = np.zeros(self.M_prop_pred.shape)
            # self.E_prop_pred_goal  = 
            self.E_prop_pred = self.M_prop_pred - self.goal_prop_tm1 # self.E_prop_pred_goal
            # self.E_prop_pred_state = self.S_prop_pred - self.M_prop_pred
            # self.E_prop_pred = self.E_prop_pred_state
            

            # 4. compute target
            # if i % 10 == 0: # play with decreased update rates
            # self.y_ = self.S_prop_pred - (self.E_prop_pred * 0.02) # error-only
            self.y_ = -self.E_prop_pred * 1.0 # i am amazed this works
            # FIXME: what is the target if there is no trivial mapping of the error?
            # print "self.y_", self.y_

            # 5. fit model
            # self.mdl.fit(self.logs["X_"], self.logs["y_"])
            # if i < 300:
            self.mdl.fit(self.X_, self.y_)

            # debug            
            # print s_pred
            # print "self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape", self.X_.shape, s_pred.shape, e_pred.shape, m.shape, self.S_ext.shape

            # attheend1?
            if np.sum(np.abs(self.goal_prop - self.goal_prop_tm1)) > 1e-6:
                # goal changed
                self.X_ = np.hstack((self.M_prop_pred - self.goal_prop)).reshape((1, self.idim)) # model input: just prediction error
            else:
                # goal unchanged
                self.X_ = np.hstack((self.E_prop_pred)).reshape((1, self.idim)) # model input: just prediction error

            # 6. compute prediction
            self.S_prop_pred = self.mdl.predict(self.X_) # state prediction
            # self.S_prop_pred = self.X_.copy() # identity doesn't do the job
            # attheend1?

            # safe last goal g_{t-1}
            self.goal_prop_tm1 = self.goal_prop.copy()
                        
            
            self.lh_do_logging(i)
            
    ################################################################################
    # plotting
    def experiment_plot(self):
        """main plot of experiment multivariate timeseries"""
        # turn off interactive mode from explauto
        # pl.ioff()

        # no data?
        if len(self.logs["X_"]) <= 0:
            return # nothing to plot
        
        # convert list to array
        if not type(self.logs["X_"]) is np.ndarray:
            self.X__ = np.array(self.logs["X_"])
        else:
            self.X__ = self.logs["X_"].copy()
        # print self.logs["X_"].shape, self.X__.shape

        # prop_pred minus prop
        err1 = self.logs["M_prop_pred"] - self.X__[:,:3]
        # print err1
        err1 = np.sqrt(np.mean(err1**2))
        err2 = self.logs["E_prop_pred"]
        # print err2
        err2 = np.sqrt(np.mean(err2**2))
        print "errors: e1 = %f, e2 = %f" % (err1, err2)
                
        pl.ioff()

        fig = pl.figure()
        fig.suptitle("Mode: %s using %s (X: FM input, state pred: FM output)" % (self.mode, self.model))
        
        ax = fig.add_subplot(511)
        ax.set_title("Proprioceptive goal")
        ax.plot(self.logs["goal_prop"], "-x")
        # ax.plot(self.logs["E2P_pred"], "-x")
        
        # ax = fig.add_subplot(512)
        # ax.set_title("Proprioceptive prediction error")
        # # ax.plot(self.X__[10:,3:], "-x")
        # # ax.plot(self.X__[:,3:], "-x")
        # ax.plot(self.logs["E_prop_pred"], "-x")
        
        ax = fig.add_subplot(512)
        ax.set_title("Proprioceptive state prediction")
        ax.plot(self.logs["S_prop_pred"])
        
        ax = fig.add_subplot(513)
        ax.set_title("Proprioceptive state measurement")
        ax.plot(self.logs["M_prop_pred"])

        ax = fig.add_subplot(514)
        ax.set_title("Proprioceptive prediction error (state - goal)")
        ax.plot(self.logs["E_prop_pred"])
        
        ax = fig.add_subplot(515)
        ax.set_title("Exteroceptive state and goal")
        ax.plot(self.logs["S_ext"], label="S_ext")
        ax.plot(self.logs["goal_ext"], label="goal_ext")
        ax.legend()

        # pl.show()
        if self.saveplots:
            fig.savefig("fig_%03d_aie_experiment_plot.pdf" % (fig.number), dpi=300)

        fig.show()
        
    def experiment_plot_basic(self):
        """plot experiment timeseries to illustrate basic operation"""
        # pl.ioff()

        # no data?
        if len(self.logs["X_"]) <= 0:
            return # nothing to plot
        
        # convert list to array
        if not type(self.logs["X_"]) is np.ndarray:
            self.X__ = np.array(self.logs["X_"])
        else:
            self.X__ = self.logs["X_"].copy()
        # print self.logs["X_"].shape, self.X__.shape

        # prop_pred minus prop
        err1 = self.logs["M_prop_pred"] - self.X__[:,:3]
        # print err1
        err1 = np.sqrt(np.mean(err1**2))
        err2 = self.logs["E_prop_pred"]
        # print err2
        err2 = np.sqrt(np.mean(err2**2))
        print "errors: e1 = %f, e2 = %f" % (err1, err2)
                
        pl.ioff()

        fig = pl.figure()
        fig.suptitle("Mode: %s using %s (X: FM input, state pred: FM output)" % (self.mode, self.model))
        
        ax = fig.add_subplot(311)
        ax.set_title("Proprioceptive goal")
        ax.plot(self.logs["goal_prop"], "-x", label="goal_p")
        ax.plot(self.logs["S_prop_pred"], label="pred_p")
        ax.plot(self.logs["M_prop_pred"], label="state_p")
        ax.legend()
        # ax.plot(self.logs["E2P_pred"], "-x")
        
        ax = fig.add_subplot(312)
        ax.set_title("Proprioceptive prediction error")
        # ax.plot(np.abs(self.logs["E_prop_pred"]), "-x")
        ax.plot(self.logs["E_prop_pred"],  label="err_p")
        ax.plot(self.logs["E_prop_pred_fast"], label="err_p lp")
        ax.plot(self.logs["E_prop_pred_slow"], label="err_p llp")
        ax.legend()
        
        ax = fig.add_subplot(313)
        ax.set_title("Derivative of proprioceptive prediction error")
        ax.plot(self.logs["dE_prop_pred_fast"], label="derr_p")
        # ax.plot(self.logs["d_E_prop_pred_"], "-x", label="derr_p/dt lp")
        # ax.plot(self.logs["d_E_prop_pred_"] * self.logs["E_prop_pred_fast"], "-x", label="(derr_p/dt lp) * (err_p lp)")
        # x = np.diff(self.logs["d_E_prop_pred_"], axis=0)
        # print "x", x
        # ax.plot(x, "-x", label="d^2E")
        # ax.plot(self.logs["dE_prop_pred_fast"] * self.logs["E_prop_pred_fast"], label="d_err_fast * (err_p lp)")
        # ax.plot(np.abs(self.logs["E_prop_pred_fast"]), label="err_p lp")
        # ax.plot((self.logs["E_prop_pred_fast"] - self.logs["E_prop_pred_slow"]) * np.abs(self.logs["E_prop_pred_fast"]), label="err_p lp - err_p llp")
        ax.legend()
        # ax.set_yscale("log")
        
        # ax = fig.add_subplot(613)
        # ax.set_title("Proprioceptive state prediction")
        # ax.plot(self.logs["S_prop_pred"])
        
        # ax = fig.add_subplot(614)
        # ax.set_title("Proprioceptive prediction error (state - goal)")
        # ax.plot(self.logs["E_prop_pred"])
        
        # ax = fig.add_subplot(615)
        # ax.set_title("Proprioceptive state")
        # ax.plot(self.logs["M_prop_pred"])

        # ax = fig.add_subplot(616)
        # ax.set_title("Exteroceptive state and goal")
        # ax.plot(self.logs["S_ext"], label="S_ext")
        # ax.plot(self.logs["goal_ext"], label="goal_ext")
        # ax.legend()
        if self.saveplots:
            fig.savefig("fig_%03d_aie_experiment_plot_basic.pdf" % (fig.number), dpi=300)
        fig.show()
        
def plot_scattermatrix(df, title = "plot_scattermatrix"):
    """plot a scattermatrix of dataframe df"""
    if df is None:
        print "plot_scattermatrix: no data passed"
        return
        
    from pandas.tools.plotting import scatter_matrix
    # df = pd.DataFrame(X, columns=['x1_t', 'x2_t', 'x1_tptau', 'x2_tptau', 'u_t'])
    # scatter_data_raw = np.hstack((np.array(Xs), np.array(Ys)))
    # scatter_data_raw = np.hstack((Xs, Ys))
    # print "scatter_data_raw", scatter_data_raw.shape

    pl.ioff()
    # df = pd.DataFrame(scatter_data_raw, columns=["x_%d" % i for i in range(scatter_data_raw.shape[1])])
    sm = scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='hist')
    fig = sm[0,0].get_figure()
    fig.suptitle(title)
    if SAVEPLOTS:
        fig.savefig("fig_%03d_scattermatrix.pdf" % (fig.number), dpi=300)
    fig.show()
    # pl.show()

def plot_scattermatrix_reduced(df, title = "plot_scattermatrix_reduced"):
    input_cols  = [i for i in df.columns if i.startswith("X")]
    output_cols = [i for i in df.columns if i.startswith("Y")]
    Xs = df[input_cols]
    Ys = df[output_cols]

    numsamples = df.shape[0]
    print "plot_scattermatrix_reduced: numsamples = %d" % numsamples
    
    # numplots = Xs.shape[1] * Ys.shape[1]
    # print "numplots = %d" % numplots

    gs = gridspec.GridSpec(Ys.shape[1], Xs.shape[1])
    pl.ioff()
    fig = pl.figure()
    fig.suptitle(title)
    # alpha = 1.0 / np.power(numsamples, 1.0/(Xs.shape[1] - 0))
    alpha = 0.2
    print "alpha", alpha
    cols = ["k", "b", "r", "g", "c", "m", "y"]
    for i in range(Xs.shape[1]):
        for j in range(Ys.shape[1]):
            # print "i, j", i, j, Xs, Ys
            ax = fig.add_subplot(gs[j, i])
            ax.plot(Xs.as_matrix()[:,i], Ys.as_matrix()[:,j], "ko", alpha = alpha)
            ax.set_xlabel(input_cols[i])
            ax.set_ylabel(output_cols[j])
    if SAVEPLOTS:
        fig.savefig("fig_%03d_scattermatrix_reduced.pdf" % (fig.number), dpi=300)
    fig.show()
            
def plot_colormeshmatrix_reduced(X, Y, ymin = None, ymax = None):
    print "X.shape", X.shape, "Y.shape", Y.shape
    # input_cols  = [i for i in df.columns if i.startswith("X")]
    # output_cols = [i for i in df.columns if i.startswith("Y")]
    # Xs = df[input_cols]
    # Ys = df[output_cols]

    # numsamples = df.shape[0]
    # print "plot_scattermatrix_reduced: numsamples = %d" % numsamples
    
    # # numplots = Xs.shape[1] * Ys.shape[1]
    # # print "numplots = %d" % numplots
    
    gs = gridspec.GridSpec(Y.shape[2], X.shape[2]/2)
    pl.ioff()
    fig = pl.figure()
    # # alpha = 1.0 / np.power(numsamples, 1.0/(Xs.shape[1] - 0))
    # alpha = 0.2
    # print "alpha", alpha
    # cols = ["k", "b", "r", "g", "c", "m", "y"]
    for i in range(X.shape[2]/2):
        for j in range(Y.shape[2]):
            # print "i, j", i, j, Xs, Ys
            ax = fig.add_subplot(gs[j, i])
            pcm = ax.pcolormesh(X[:,:,i], X[:,:,X.shape[2]/2+i], Y[:,:,j], vmin = ymin, vmax = ymax)
            # ax.plot(Xs.as_matrix()[:,i], Ys.as_matrix()[:,j], "ko", alpha = alpha)
            ax.set_xlabel("goal")
            ax.set_ylabel("error")
            cbar = fig.colorbar(mappable = pcm, ax=ax, orientation="horizontal")
            ax.set_aspect(1)
    if SAVEPLOTS:
        fig.savefig("fig_%03d_colormeshmatrix_reduced.pdf" % (fig.number), dpi=300)
    fig.show()

def test_models(args):
    from actinf_models import ActInfModel
    from actinf_models import ActInfKNN
    from actinf_models import ActInfSOESGP

    idim = 4
    odim = 2
    numdatapoints = 10
    
    for aimclass in [ActInfModel, ActInfKNN, ActInfSOESGP, ActInfSTORKGP, ActInfGMM, ActInfHebbianSOM]:
        print("Testing aimclass = %s" % (aimclass,))
        aim = aimclass(idim = idim, odim = odim)

        X = np.random.uniform(-0.1, 0.1, (numdatapoints, 1, idim))
        y = np.random.uniform(-0.1, 0.1, (numdatapoints, 1, odim))

        for i in range(numdatapoints-1):
            print("Fitting model with X = %s, Y = %s" % (X[i].shape, y[i].shape))
            aim.fit(X[i], y[i])
        y_ = aim.predict(X[i+1])
        print("Prediction error = %s" % (y_ - y[i+1]))
    
def main(args):

    # seed PRNG
    np.random.seed(args.seed)
        
    if args.mode.startswith("test_"):
        test_models(args)
    else:
        idim = None
        # if args.mode.startswith("type03_1"):
        #     idim = 3
        # print "args.goal_sample_interval", args.goal_sample_interval
        inf = ActiveInferenceExperiment(args.mode, args.model, args.numsteps,
                                        idim = idim,
                                        environment_str = args.environment,
                                        goal_sample_interval = args.goal_sample_interval,
                                        e2pmodel = args.e2pmodel,
                                        saveplots = SAVEPLOTS)

        inf.run()

        # inf.experiment_plot()
        pl.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e2p", "--e2pmodel",             type=str, help="extero to proprio mapping [gmm]", default="gmm")
    parser.add_argument("-e",   "--environment",          type=str, help="which environment to use [simplearm] one of " + ", ".join(actinf_environments), default="simplearm")
    parser.add_argument("-gsi", "--goal_sample_interval", type=int, help="Interval at which to sample goals [50]", default=50)
    parser.add_argument("-m",   "--mode",                 type=str, help="program execution mode, one of " + ", ".join(modes) + " [m1_goal_error_nd_e2p]", default="m1_goal_error_nd_e2p")
    parser.add_argument("-md",  "--model",                type=str, help="learning machine [knn]", default="knn")
    parser.add_argument("-n",   "--numsteps",             type=int, help="number of learning steps [1000]", default=1000)
    parser.add_argument("-s",  "--seed",                  type=int, help="seed for RNG [0]",        default=0)
    args = parser.parse_args()

    main(args)
