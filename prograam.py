import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the path to the directory containing your training and validation data
train_dir = 'train_dir'
validation_dir = 'validation_dir'

# Set the image dimensions
img_width, img_height = 150, 150

# Set the number of training and validation samples
num_train_samples = 8
num_validation_samples = 4

# Set the batch size and number of epochs
batch_size = 2
epochs = 5

# Define the class labels
class_labels = ['cat', 'dog']

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='binary')

# Train the model
history = model.fit_generator(train_generator,
                              steps_per_epoch=num_train_samples // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=num_validation_samples // batch_size)
# Calculate the accuracy on validation data
val_loss, val_accuracy = model.evaluate_generator(validation_generator,
                                                  steps=num_validation_samples // batch_size)
val_accuracy_percentage = val_accuracy * 100

print(f"Validation Accuracy: {val_accuracy_percentage:.2f}%")



# Save the model
model.save('cat_dog_classifier.h5')
