# actinf - Active Inference

Sensorimotor learning experiments using the active inference on
predictive coding approach. Inspired by the input from Bruno Lara and
Guido Schillaci. This just came out of my main work repo as a
submodule for easier sharing.

## Dependencies

Apart from the base dependencies numpy, scipy, sklearn, matplotlib we
need specifically:
 - explauto for the simple_arm, probably best to use my fork of
   explauto at https://github.com/x75/explauto, then checkout "smp"
   branch
 - PyPR: pip install pypr for gaussian mixtures with conditional inference
 - kohonen, SOM library, also use my fork at
   https://github.com/x75/kohonen, branch is master. We need this in active_inference_hebbsom.py

## Files

 - actinf_models.py: knn, soesgp and storkgp based online learning
   models
 - active_inference_basic.py: base experiment, here we use explauto's
   simple_arm in low-dimensional configuration (3 joints) to learn to
   control the arm under dynamic online goals. two modes:
  - type03_goal_prediction_error: most basic proprioceptive only model
  - type04_ext_prop: this introduces an e2p map that's built with a
    gaussian mixture model using PyPR lib so we can pass down
    exteroceptive goals to the proprioceptive layer
 - active_inference_naoqi.py: base proprio only learning using webots
   and naoqi on a simulated nao
 - active_inference_hebbsom.py: this is just replicating the gaussian
   mixture model functionality using SOMs with Hebbian associative
   connections and conditional inference

