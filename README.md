actinf - Active Inference
=========================

A set of sensorimotor learning experiments using the active inference and predictive coding approach. Inspired by the input from Bruno Lara and Guido Schillaci.

This just came out of my main work repo as a submodule for easier sharing. This module is work in progress so please expect the interface to be changing while integration is happening. If you want to make changes, fork and send a pull request.

**Update 20170607** I'm freezing this repository in favor of integrating the experiments and models into [smp\_graphs](https://github.com/x75/smp\_graphs).

Dependencies
------------

The base dependencies numpy, scipy, sklearn, and matplotlib which you can usually be installed from your distribution with

``` example
sudo apt install python-numpy python-scipy python-sklearn
```

python-matplotlib

or alternatively through pip with

``` example
pip install numpy scipy sklearn matplotlib
```

In addition, the following packages are needed, some of which need additional setup.

**explauto** is a sensorimotor exploration library by the INRIA Flowers team. We use their simple\_arm environment, and have added the pointmass as an explauto environment. The project is forked on <https://github.com/x75/explauto> and has an 'smp' branch

``` example
git clone git@github.com:x75/explauto.git
cd explauto
git checkout smp
```

**kohonen** is a SOM library used in 'active\_inference\_hebbsom.py', also forked on <https://github.com/x75/kohonen>

``` example
git clone git@github.com:x75/kohonen.git
```

**otl** is Harold Soh's online temporal learning library which is forked from the original mercurial repository on <https://github.com/x75/otl>

``` example
git clone git@github.com:x75/otl.git
cd otl
```

then follow build instructions in README.txt and make sure to enable Python bindings, in brief

``` example
sudo apt install libeigen3-dev cmake swig
cd otl
mkdir build
cd build
cmake ../ -DBUILD_PYTHON_BINDINGS=ON -DBUILD_DOCS=ON
make
```

**PyPR** is a pattern recognition library providing conditional inference on gaussian mixture models

``` example
pip install pypr
```

**Finish** the setup by appending these lines to your .bashrc file

``` example
export PYTHONPATH=/path/to/explauto:$PYTHONPATH
export PYTHONPATH=/path/to/rlspy:$PYTHONPATH
export PYTHONPATH=/path/to/kohonen:$PYTHONPATH
```

Files
-----

-   actinf\_models.py: knn, soesgp and storkgp based online learning models
-   active\_inference\_basic.py: basic experiments, here we use explauto's simple\_arm in low-dimensional configuration (3 joints) to learn to control the arm under dynamic online goals. Several program execution modes are available:
-   m1\_goal\_error\_nd: most basic proprioceptive only model
-   m2\_error\_nd: most basic proprioceptive only model
-   m1\_goal\_error\_nd\_e2p: this introduces an e2p map that's built with a gaussian mixture model using PyPR lib so we can pass down exteroceptive goals to the proprioceptive layer
-   test\_models: basic model test
-   plot\_system: plot the system response as scattermatrix
-   m1\_goal\_error\_1d: demonstrate the basic operation of model type M1 on a one-dimensional system
-   m2\_error\_nd\_ext: analyze prediction errors on model type M2
-   active\_inference\_naoqi.py: base proprio only learning using webots and naoqi on a simulated nao
-   active\_inference\_hebbsom.py: this is just replicating the gaussian mixture model functionality using SOMs with Hebbian associative connections and conditional inference

Examples
--------

Run it like

``` example
python active_inference_basic.py --mode m1_goal_error_nd
```

which will run the basic proprioceptive only learning scenario.

This will also run a p-only learning but different setup, FWDp only get's the error as an input.

``` example
python active_inference_basic.py --mode m2_error_nd --numsteps 2000 --model knn
```

This command

``` example
python active_inference_basic.py --mode m1_goal_error_nd_e2p --model knn --numsteps 1000
```

which will run the combined etxtero/proprio learning scenario for 1000 timesteps (the default anyway) and produce various plots along the way.

Runs of active\_inference\_basic.py will produce a file "EP.npy" in the current directory. This needs to be passed to active\_inference\_hebbsom.py as a datafile to load for testing only the e2p prbabilistic mapping part using the Hebbian SOMs.

I run it like

``` example
python active_inference_hebbsom.py --datafile data/simplearm_n3000/EP.npy
```

which contains data from a 3000 timesteps run of active\_inference\_basic.py. on the first run, hebbsom will first learn the SOMs for E and P, then learn the Hebbian connections (I separated it for debugging, rejoining the processes is TODO, considers large initial neighborhood\_size and decreasing it with time) and finally evaluate the learnt mapping by feeding extero signal to the E map, activating P map via Hebbian links, sample from P map joint density, feed sampled P\_ into the real system and compute the end effector position.
