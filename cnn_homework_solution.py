# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Have distinct data structure for training and test set

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D # Speech is 1D, Image is 2D & Video are 3D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# Stride =1
# We start with 32 feature detectors, slowly move to 64 and 128, etc
# Input shape is the image dimension/format, 3 represents the three channels- R, B & G
# The dimension of the training images are > 256x256. We can use higher dim when working on GPU
# Now we restrict the dimension to 64*64
# Argument order in theano is different from tf. Here we use tf backend
# Activation function is to remove any linearity present by removing negative pixels in convolved image
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
# Image translational invariance (or retain the spatial relationship), reduce dimension yet keep the information
# We will also reduce the nodes in the flatten layer and reduce the computational intensive
# Highly recomended to use 2x2
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))# You can control strides here as well

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# Multiple categories- softmax & categorical_cross entropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images


from keras.preprocessing.image import ImageDataGenerator # Image Augmentation
# Image/Data Augmentation avoids overfitting by generalizing the spatial relationships
# Image transformation increasing the data by rotating, shearing, zoom, etc

train_datagen = ImageDataGenerator(rescale = 1./255, # This is compulsory step and is like scaling of the num variables to nullify the effect of units/scale of measurement
# Rescale is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our model to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255                                   
                                   shear_range = 0.2, # random shearing
                                   zoom_range = 0.2, # random zoom
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 800,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 200)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    


# How can you improve the model?
'1. batch normalization'
'2. data augmentation'
'3. pre-trained models/weights such as VGG, InceptionV3, etc.'