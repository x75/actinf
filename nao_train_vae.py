'''File adapted from Keras'''
'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import os
import matplotlib.pyplot as plt

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
from tqdm import tqdm

from prettytensor.layers import xavier_init

img_size = 64
batch_size = 100
original_dim = img_size*img_size
latent_dim = 2
intermediate_dim = 256
nb_epoch = 50
images_path="/home/guido/Desktop/extero_images/"
models_path="/home/guido/datasets/"
dataset_path = models_path + "_nao_images.pkl.gz"
encoder_path = models_path + "_encoder.h5"
encoder_json_path = models_path + "_encoder.json"
decoder_path = models_path + "_decoder.h5"
decoder_json_path = models_path + "_decoder.json"

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

# train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

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



#x_train = datasets.fetch_mldata(path)
print len(x_train), len(x_train)%batch_size, len(x_train[1:-(len(x_train)%batch_size)+1])

#x_train = np.asarray(x_train).astype('float32') / 255.
#x_test = np.asarray(x_test).astype('float32') / 255.
# len of x_train and x_test has to be %batch_size
x_train = np.asarray(x_train[1:-(len(x_train)%batch_size)+1]).astype('float32') / 255.
x_test = np.asarray(x_test[1:-(len(x_test)%batch_size)+1]).astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoder=[]
decoder=[]
if not(os.path.exists(encoder_path)) and not(os.path.exists(decoder_path)):
    print "Training model..."
    vae.fit(x_train, x_train,
            shuffle=True,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            validation_data=(x_test, x_test))
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
#    encoder.compile(optimizer='rmsprop', loss=vae_loss)
    encoder_json = encoder.to_json()
    with open(encoder_json_path, "w") as json_file:
	json_file.write(encoder_json)
    encoder.save_weights(encoder_path)

    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    decoder = Model(decoder_input, _x_decoded_mean)
    decoder_json = decoder.to_json()
    with open(decoder_json_path, "w") as json_file:
	json_file.write(decoder_json)
    decoder.save_weights(decoder_path)

#    decoder.compile(optimizer='rmsprop', loss=vae_loss)
 #   decoder.save(decoder_path)
    print "Model trained"
else:
    print "Loading model..."
#    encoder = load_model(encoder_path)
#    decoder = load_model(decoder_path)
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




# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#print x_test_encoded[:,0]
print "min ", np.min(x_test_encoded[:,0]), " ", np.min(x_test_encoded[:,1])
print "max ", np.max(x_test_encoded[:,0]), " ", np.max(x_test_encoded[:,1])
print "mean ", np.mean(x_test_encoded[:,0]), " ", np.mean(x_test_encoded[:,1])
print "stddev ", np.std(x_test_encoded[:,0]), " ", np.std(x_test_encoded[:,1])

#print "min ", np.min(x_test_encoded[:,0]), " ", np.min(x_test_encoded[:,1]), np.min(x_test_encoded[:,2]), np.min(x_test_encoded[:,3])
#print "max ", np.max(x_test_encoded[:,0]), " ", np.max(x_test_encoded[:,1]), np.max(x_test_encoded[:,2]), np.max(x_test_encoded[:,3])
#print "mean ", np.mean(x_test_encoded[:,0]), " ", np.mean(x_test_encoded[:,1]), np.mean(x_test_encoded[:,2]), np.mean(x_test_encoded[:,3])
#print "stddev ", np.std(x_test_encoded[:,0]), " ", np.std(x_test_encoded[:,1]), np.std(x_test_encoded[:,2]), np.std(x_test_encoded[:,3])


plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
#plt.colorbar()
plt.show()

#plt.figure(figsize=(6, 6))
##plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.scatter(x_test_encoded[:, 2], x_test_encoded[:, 3])
##plt.colorbar()
#plt.show()


# display a 2D manifold of the digits
#n = 15  # figure with 15x15 digits
#digit_size = 28
#digit_size = img_size
#figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
#grid_x = np.linspace(-15, 15, n)
#grid_y = np.linspace(-15, 15, n)

#print np.linspace(0., 1.5, 15)
#for i, yi in enumerate(grid_x):
#   for j, xi in enumerate(grid_y):
#        z_sample = np.array([[xi, yi]])
#        x_decoded = generator.predict(z_sample)
#        digit = x_decoded[0].reshape(digit_size, digit_size)
#        figure[i * digit_size: (i + 1) * digit_size,
#               j * digit_size: (j + 1) * digit_size] = digit
#plt.figure(figsize=(10, 10))
#plt.imshow(figure, cmap='Greys_r')
#plt.show()

input = np.zeros(latent_dim)
plt.figure(figsize=(10,10))
plt.ion()
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



