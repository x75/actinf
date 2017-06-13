# Documentation of active inference (actinf) experiments

## actinf files ##

* actinf_models.py: basic models with fit(X,Y)/predict(X) style interface
* active_inference_basic.py: active inference experiments in proprio
  and exteroceptive space
* active_inference_hebbsom.py: devel testbed of the hebbsom model
* active_inference_naoqi.py: interfacing original actinf code to naoqi

## actinf experiments ##

### actinf_models ###

`python actinf_models.py`

runs a fit/predict test on all base function approximators (models)
implemented in the file: knn, sparse online echo state gaussian
process (soesgp), storkgp, gaussian mixture model, hebbsom. Running it
with no arguments tests all the models and plots the results. Setting
the modelclass argument tests a specific model.

### actinf_inference_basic test_models ###

`python active_inference_basic.py --mode test_models`

runs another more basic test on knn, soesgp, storkgp models

### actinf_inference_basic plot_system ###

```
python active_inference_basic.py --mode plot_system --environment simplearm
python active_inference_basic.py --mode plot_system --environment pointmass1d
python active_inference_basic.py --mode plot_system --environment pointmass3d
```

Plots the system input/output behaviour in proprio space by an input
sweep over a grid, see Fig. 1 - 3.

![simplearm env](doc/img/actinf_plot_system_simplearm.jpg)
![pointmass1d env](doc/img/actinf_plot_system_pointmass1d.jpg)
![pointmass3d env](doc/img/actinf_plot_system_pointmass3d.jpg)

### actinf_inference_basic m1_goal_error_1d ###

```
python active_inference_basic.py --mode m1_goal_error_1d --environment
pointmass1d --model knn
```

Runs the actinf learning variant 1 (M1) which uses both goal and
prediction error as input to the forward model. This is the minimal
version for a one-dimensional idenity system.

![simplearm system](doc/img/basic_op_id_1000steps_knn/fig_001_scattermatrix_reduced.pdf)

The state of the model at half and full number of episode steps.


![simplearm model early](doc/img/basic_op_id_1000steps_knn/fig_002_colormeshmatrix_reduced.pdf)

![simplearm model final](doc/img/basic_op_id_1000steps_knn/fig_004_colormeshmatrix_reduced.pdf)

Variable timeseries plot.

![simplearm timeseries](doc/img/basic_op_id_1000steps_knn/fig_003_aie_experiment_plot_basic.pdf)

This can be repeated for other models


```
python active_inference_basic.py --mode m1_goal_error_1d --environment pointmass1d --model soesgp
python active_inference_basic.py --mode m1_goal_error_1d --environment pointmass1d --model storkgp
```

### Help ###


```
$ python active_inference_basic.py --help
usage: active_inference_basic.py [-h] [-e2p E2PMODEL] [-e ENVIRONMENT]
                                 [-gsi GOAL_SAMPLE_INTERVAL] [-m MODE]
                                 [-md MODEL] [-n NUMSTEPS] [-s SEED]

optional arguments:
  -h, --help            show this help message and exit
  -e2p E2PMODEL, --e2pmodel E2PMODEL
                        extero to proprio mapping [gmm]
  -e ENVIRONMENT, --environment ENVIRONMENT
                        which environment to use [simplearm] one of simplearm,
                        pointmass1d, pointmass3d
  -gsi GOAL_SAMPLE_INTERVAL, --goal_sample_interval GOAL_SAMPLE_INTERVAL
                        Interval at which to sample goals [50]
  -m MODE, --mode MODE  program execution mode, one of test_models,
                        type01_state_prediction_error, type02_state,
                        type03_1_prediction_error,
                        type03_goal_prediction_error, type04_ext_prop,
                        type05_multiple_models, m2_prediction_errors,
                        basic_operation_1D_M1, plot_system [type04_ext_prop]
  -md MODEL, --model MODEL
                        learning machine [knn]
  -n NUMSTEPS, --numsteps NUMSTEPS
                        number of learning steps [1000]
  -s SEED, --seed SEED  seed for RNG [0]
```
