'''File adapted from Keras'''
'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import os
import matplotlib.pyplot as plt

import cv2
import random
import gzip
import cPickle
import time
import glob
from tqdm import tqdm


from naoqi import ALProxy
import vision_definitions
import Image

nr_frames_to_save = 500

IP = "127.0.0.1"
PORT = 9559
camProxy = ALProxy("ALVideoDevice", IP, PORT)
motionProxy = ALProxy("ALMotion", IP, PORT)
angle_names  = ["LShoulderPitch","LShoulderRoll", "LElbowYaw", "LElbowRoll"]


resolution = vision_definitions.kQVGA
colorSpace = vision_definitions.kBGRColorSpace
fps = 10
nameId = camProxy.subscribe("python_GVM", resolution, 11, fps)
camProxy.setParam(vision_definitions.kCameraSelectID, 1) #1=bottom

path="/home/guido/datasets/oswald/"
dataset_path = path + "_nao_images_and_proprio.pkl.gz"

data_img = []
data_proprio = []

# get the image from naoqi
def getImage():
  current_image = []
  naoImage = camProxy.getImageRemote(nameId)
  Width = naoImage[0]
  Height = naoImage[1]
  array = naoImage[6]
  image = Image.fromstring("RGB", (Width, Height), array)
  current_image = np.array(image) 
#  current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
#  current_image = cv2.resize(current_image, (img_size, img_size))
  return current_image


iter = 0
while(iter < nr_frames_to_save):

  current_image = getImage()
  angles = motionProxy.getAngles(angle_names, True)

  data_img.append(current_image)
  data_proprio.append(angles)

  camProxy.releaseImage(nameId)
  print iter
  iter = iter +1
  time.sleep(0.1) 

dataset = [data_img, data_proprio]
f = gzip.open(dataset_path,'wb')
cPickle.dump(dataset, f, protocol=-1)
f.close()
print("Dataset saved into " + dataset_path)

'''
# Is dataset available? If not, load images and create it
x_train=[]
x_val=[]
x_test=[]

if not(os.path.exists(dataset_path)):
    print("Reading images...")
    for img_file in tqdm(glob.glob(images_path + "*.jpg")):
        #print(img_file)
        #img = Image.open(path  + "/" + file)
        img = cv2.imread(img_file, 0) # 0 for loading in gray scale
        #img = cv2.imread(img_file)  # 0 for loading in gray scale
        img = cv2.resize(img, (img_size, img_size))
        rnd = random.random()
        if (rnd < 0.1):
            x_test.append(img)
        elif ((rnd > 0.1) and (rnd < 0.2)):
            x_val.append(img)
        else:
            x_train.append(img)
    print("Images read.")
    dataset = [x_train, x_val, x_test]
    f = gzip.open(dataset_path,'wb')
    cPickle.dump(dataset, f, protocol=-1)
    f.close()
    print("Dataset saved into " + dataset_path)
else:
    print("Loading dataset: "+ dataset_path)
    with gzip.open(dataset_path, 'rb') as f:
        x_train, x_val, x_test = cPickle.load(f)
    print "Dataset loaded."



for i in np.arange(100):
    for idx,s  in enumerate(input):
        input[idx] = input[idx] + np.random.normal(0.0, 0.1)

    #z_sample = np.array([[xi, yi]])
    z_sample = np.array([input])
    x_decoded = decoder.predict(z_sample)
    generated_img = x_decoded[0].reshape(img_size, img_size)
    plt.title(input)
    plt.imshow(generated_img, cmap='Greys_r')
    time.sleep(0.01)
    plt.draw()
    plt.show()
    plt.clf()

'''

