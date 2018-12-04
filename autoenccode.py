from keras.layers import Input, Dense
from keras.models import Model, Sequential
    
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import MaxPooling2D,UpSampling2D

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
#input_img = Input(shape=(784,))

inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))
# "encoded" is the encoded representation of the input

encoded = Sequential()

#encoded.add(Conv2D(32, (3, 3), padding="same",
#input_shape=(3,32,32)))
#encoded.add(Activation("relu"))
#encoded = Dense(encoding_dim, activation='relu')(input_img)

conv1 = Conv2D(24, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
conv2 = Conv2D(48, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
encoded = Conv2D(96, (3, 3), activation='relu', padding='same')(pool2)




		 #softmax classifier
#encoded.add(Flatten())
#encoded.add(Dense(classes))
#encoded.add(Activation("relu"))


#encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
         
conv4 = Conv2D(96, (3, 3), activation='relu', padding='same')(encoded) #7 x 7 x 128
up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
conv5 = Conv2D(48, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
         
         
#decoded = Dense(1, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction

autoencoder = Model(input_img, decoded)


# this model maps an input to its encoded representation

autoencoder.compile(optimizer='adadelta', loss='mse', metrics=["accuracy"])









from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_train = x_train.reshape(-1, 28,28, 1)

#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_test = x_test.reshape(-1, 28,28, 1)
print (x_train.shape)
print (x_test.shape)






'''
encoded_input = Input(shape = (28,28, 1))
encoded_input1 = Input(shape = (14,14, 64))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input1))
decoded_imgs = decoder.predict(x_test)



decoder_layer = autoencoder.layers[-1]

encoded_input = Input(shape = (28,28, 1))
decoder = Model(encoded_input, decoder_layer(encoded_input))

#encoded_imgs1 = decoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
'''




#decoder1 = Model(input_img,decoded)
#decoded_imgs = decoder.predict(x_test)


hist=autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.summary()

autoencoder.save('output1.h5')

decoded_imgs = autoencoder.predict(x_test)


'''
encoded_input = Input(shape=(28,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

#decoder1 = Model(input_img,decoded)


conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)




		 #softmax classifier
#encoded.add(Flatten())
#encoded.add(Dense(classes))
#encoded.add(Activation("relu"))


#encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
         
conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded) #7 x 7 x 128
up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
         

'''
#conv1images = Model(input_img, conv1).predict(x_test)
#pool1imgaes = Model((28,28,32), pool1).predict(conv1images)


'''
encoded_input = Input(shape=(28,28,64))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))



decoded_imgs = decoder.predict(x_test)



encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_test)

decoder1 = Model(input_img,decoded)
decoded_imgs = decoder1.predict(x_test)
'''

# encode and decode some digits
# note that we take them from the *test* set



'''
inChannel = 64
x, y = 28, 28
input_img1 = Input(shape = (28, 28, 64))

decoder1=Model(input_img1,up2)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape = (28,28, 1))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[0]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))




encoded_imgs = encoder.predict(x_test)

encoded_imgs1 = decoder1.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs1)
# use Matplotlib (don't ask)
'''
import matplotlib.pyplot as plt
'''
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i, ..., 0], cmap='gray')
    #curr_lbl = test_labels[i]
    #plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
plt.show()    
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(decoded_imgs[i, ..., 0], cmap='gray')  
plt.show()


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 1), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, 1), hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 1), hist.history["acc"], label="train_acc")
plt.plot(np.arange(0, 1), hist.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
