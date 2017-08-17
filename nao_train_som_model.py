
import argparse

parser = argparse.ArgumentParser(description='Train the models according to the predictive coding theory.')
verbose_parser = parser.add_mutually_exclusive_group(required=False)
verbose_parser.add_argument('--verbose', default=False, action="store_true" , help="Verbose run")
args = parser.parse_args()


print "Verbose is set to ", args.verbose

import numpy as np
import os
import matplotlib.pyplot as plt

import pylab as pl

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.models import load_model
from keras.models import model_from_json
import cv2
import random
import gzip
import cPickle
import time
import glob

from naoqi import ALProxy
import vision_definitions
import Image

import dircache
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf



#verbose = True

def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())


images_path="/home/guido/Desktop/extero_images/"
models_path="/home/guido/datasets/"
dataset_path = models_path + "_nao_images.pkl.gz"
encoder_path = models_path + "_encoder.h5"
encoder_json_path = models_path + "_encoder.json"
decoder_path = models_path + "_decoder.h5"
decoder_json_path = models_path + "_decoder.json"


img_size = 64
batch_size = 100
original_dim = img_size*img_size
latent_dim = 2
intermediate_dim = 256

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

encoder = Model(x, z_mean)
encoder.compile(optimizer='rmsprop', loss=vae_loss)

#encoder=[]
#generator=[]

print "Loading model..."
#encoder = load_model(encoder_path)

enc_json_file = open(encoder_json_path, 'r')
loaded_enc_json = enc_json_file.read()
enc_json_file.close()
encoder = model_from_json(loaded_enc_json)
# load weights into new model
encoder.load_weights(encoder_path)

dec_json_file = open(decoder_json_path, 'r')
loaded_dec_json = dec_json_file.read()
dec_json_file.close()
decoder = model_from_json(loaded_dec_json)
# load weights into new model
decoder.load_weights(decoder_path)

print "Model loaded"


IP = "127.0.0.1"
PORT = 9559
camProxy = ALProxy("ALVideoDevice", IP, PORT)
pcmoduleProxy = ALProxy("PCModule", IP, PORT)
motionProxy = ALProxy("ALMotion", IP, PORT)
angle_names  = ["LShoulderPitch","LShoulderRoll", "LElbowYaw", "LElbowRoll"]

som_size_P=pcmoduleProxy.getProprioceptiveSOMSize()
som_size_E=pcmoduleProxy.getExteroceptiveSOMSize()

resolution = vision_definitions.kQVGA
colorSpace = vision_definitions.kBGRColorSpace
fps = 10
nameId = camProxy.subscribe("python_GVM", resolution, 11, fps)
camProxy.setParam(vision_definitions.kCameraSelectID, 1) #1=bottom

if args.verbose:
  cv2.startWindowThread()
  cv2.namedWindow("current image")
  cv2.resizeWindow("current image", 320, 240)
  cv2.moveWindow("current image", 620, 0)
  cv2.startWindowThread()
  cv2.namedWindow("current goal")
  cv2.resizeWindow("current goal", 320, 240)
  cv2.moveWindow("current goal", 200, 0)
  cv2.startWindowThread()
  cv2.namedWindow("FM prediction")
  cv2.resizeWindow("FM prediction", 320, 240)
  cv2.moveWindow("FM prediction", 1000, 0)

  cv2.startWindowThread()
  cv2.namedWindow("E_activation")
  cv2.moveWindow("E_activation",200, 320)

  cv2.startWindowThread()
  cv2.namedWindow("E2P_activation")
  cv2.moveWindow("E2P_activation", 620, 320)

  cv2.startWindowThread()
  cv2.namedWindow("P_activation")
  cv2.moveWindow("P_activation", 1000, 320)

  cv2.startWindowThread()
  cv2.namedWindow("errorE_activation")
  cv2.moveWindow("errorE_activation",200, 620)

  cv2.startWindowThread()
  cv2.namedWindow("errorP_activation")
  cv2.moveWindow("errorP_activation", 1000, 620)

  cv2.startWindowThread()
  cv2.namedWindow("predictionE_activation")
  cv2.moveWindow("predictionE_activation",200, 1000)

  cv2.startWindowThread()
  cv2.namedWindow("goalE_activation")
  cv2.moveWindow("goalE_activation",620, 1000)

goal_filename = []
filename = random.choice(dircache.listdir(images_path))
goal_filename = images_path + filename
print "goal ", goal_filename


# get the image from naoqi
def getImage():
  current_image = []
  naoImage = camProxy.getImageRemote(nameId)
  Width = naoImage[0]
  Height = naoImage[1]
  array = naoImage[6]
  image = Image.fromstring("RGB", (Width, Height), array)
  current_image = np.array(image) 
  current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
  current_image = cv2.resize(current_image, (img_size, img_size))
  return current_image

# encode the image using the VAE
def encodeImage (encoder, image, batch_size):
  image_ = []
  for i in range(0, batch_size):
    image_.append(image.flatten())
  image_ = np.asarray(image_).astype('float32') / 255.
  return encoder.predict(image_, batch_size = batch_size)

def loadGoal():
  goal = []
  filename = random.choice(dircache.listdir(images_path))
  goal_filename = images_path + filename
  print "goal ", goal_filename
  goal = cv2.imread(goal_filename, 0)  # 0 for loading in gray scale  
  goal = cv2.resize(goal, (img_size, img_size))
  return goal

###########################################
## main loop
###########################################
global encoded_current_goal
global encoded_current_image
encoded_current_goal = np.zeros(( 1, latent_dim))
encoded_current_image = np.zeros(( 1, latent_dim))

switch_iter = 10000

iter = 0
while(iter < (switch_iter*2)):
  # sample an exteroceptive goal
  current_goal = []

  if (iter%120) == 0: # 12 secs
    # load the goal
    current_goal = loadGoal()
    if args.verbose:
      cv2.imshow("current goal", cv2.resize(current_goal, (320, 240)) )
    # encode it with VAE
    encoded_current_goal = encodeImage(encoder, current_goal, batch_size)

  ## get current joint angles
  angles = motionProxy.getAngles(angle_names, True)

  #### execute motor command
  if (iter > switch_iter) : # 10 secs
    #pcmoduleProxy.execute_command(float(encoded_current_goal[0,0]), float(encoded_current_goal[0,1]), float(encoded_current_goal[0,2]), float(encoded_current_goal[0,3]), False, 3)
    pcmoduleProxy.execute_command(float(encoded_current_goal[0,0]), float(encoded_current_goal[0,1]), False, 0.1)
  else:
    # motor babbling
    #if (iter%120) == 0: # 12 secs
    #pcmoduleProxy.execute_command(float(encoded_current_goal[0,0]), float(encoded_current_goal[0,1]), float(encoded_current_goal[0,2]), float(encoded_current_goal[0,3]), True, 3)
    pcmoduleProxy.execute_command(float(encoded_current_goal[0,0]), float(encoded_current_goal[0,1]), True, 0.1)

  ## wait a bit that the command is executed (100ms)
  time.sleep(0.1) 



  ## update proprioceptive FM
  pcmoduleProxy.trainStep_proprio()
  
  ## get current image
  current_image = getImage()
  if args.verbose:
    cv2.imshow("current image", cv2.resize(current_image, (320, 240))  )
  # encode the current image with VAE
  encoded_current_image = encodeImage(encoder, current_image, batch_size)
  #print "current ", encoded_current_image[0]
 
  ## update exteroceptive FM
  #pcmoduleProxy.trainStep_extero(float(encoded_current_image[0,0]), float(encoded_current_image[0,1]), float(encoded_current_image[0,2]), float(encoded_current_image[0,3]))
  pcmoduleProxy.trainStep_extero(float(encoded_current_image[0,0]), float(encoded_current_image[0,1]))


  ## update the E-2-P mapping
  pcmoduleProxy.updateEPmapping()


  ## plot some stuff
  if args.verbose:
    P_activation = []
    P_activation = pcmoduleProxy.getProprioceptiveSOMActivation(angles[0],angles[1],angles[2],angles[3])
    #  print P_activation
    E_activation = []
    #E_activation = pcmoduleProxy.getExteroceptiveSOMActivation(float(encoded_current_image[0,0]), float(encoded_current_image[0,1]), float(encoded_current_image[0,2]), float(encoded_current_image[0,3]))
    E_activation = pcmoduleProxy.getExteroceptiveSOMActivation(float(encoded_current_image[0,0]), float(encoded_current_image[0,1]))

    E2P_prediction = []
    #E2P_prediction = pcmoduleProxy.getE2Pprediction(float(encoded_current_image[0,0]), float(encoded_current_image[0,1]), float(encoded_current_image[0,2]), float(encoded_current_image[0,3]))
    E2P_prediction = pcmoduleProxy.getE2Pprediction(float(encoded_current_image[0,0]), float(encoded_current_image[0,1]))
    E2P_activation = []
    E2P_activation = pcmoduleProxy.getProprioceptiveSOMActivation(E2P_prediction[0],E2P_prediction[1],E2P_prediction[2],E2P_prediction[3])

    cv2.imshow("P_activation", cv2.resize(np.array(P_activation).reshape(som_size_P, som_size_P), (320, 240), interpolation = cv2.INTER_NEAREST)  )
    cv2.imshow("E2P_activation", cv2.resize(np.array(E2P_activation).reshape(som_size_P, som_size_P), (320, 240), interpolation = cv2.INTER_NEAREST)  )
    cv2.imshow("E_activation", cv2.resize(np.array(E_activation).reshape(som_size_E, som_size_E), (320, 240), interpolation = cv2.INTER_NEAREST)  )

    errorP_activation = []
    errorP_activation = pcmoduleProxy.getProprioceptiveErrorSOMActivation()
    #  print P_activation
    errorE_activation = []
    errorE_activation = pcmoduleProxy.getExteroceptiveErrorSOMActivation()

    cv2.imshow("errorP_activation", cv2.resize(np.array(errorP_activation).reshape(som_size_P, som_size_P), (320, 240), interpolation = cv2.INTER_NEAREST)  )
    cv2.imshow("errorE_activation", cv2.resize(np.array(errorE_activation).reshape(som_size_E, som_size_E), (320, 240), interpolation = cv2.INTER_NEAREST)  )

    predictionE_activation = []
    predictionE_activation = pcmoduleProxy.getExteroceptiveSOMPredictionActivation()
    cv2.imshow("predictionE_activation", cv2.resize(np.array(predictionE_activation).reshape(som_size_E, som_size_E), (320, 240), interpolation = cv2.INTER_NEAREST)  )

    goalE_activation = []
    goalE_activation = pcmoduleProxy.getExteroceptiveSOMGoalActivation()
    cv2.imshow("goalE_activation", cv2.resize(np.array(goalE_activation).reshape(som_size_E, som_size_E), (320, 240), interpolation = cv2.INTER_NEAREST)  )

    ##print iter, "P_activation-E2P_activation: ",np.sum(np.abs(np.asarray(P_activation) - np.asarray(E2P_activation)))
    print iter



  # save the errors and the log
  pcmoduleProxy.updateLog()

  if args.verbose:
    ## plot the exteroceptive prediction
    extero_prediction = pcmoduleProxy.getCurrentExteroceptivePrediction()
    # draw image
    z_sample = np.array([extero_prediction])
    x_decoded = decoder.predict(z_sample)
    generated_img = x_decoded[0].reshape(img_size, img_size)
    cv2.imshow("FM prediction", cv2.resize(generated_img, (320, 240)) )


  camProxy.releaseImage(nameId)

  iter = iter +1
