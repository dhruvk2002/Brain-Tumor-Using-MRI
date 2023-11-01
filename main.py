import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import save_model
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
# Directory containing the binary classification dataset (tumor yes or no)

binary_image_directory = 'dataset/'

# Binary classification (tumor yes or no)
no_tumor_images = os.listdir(binary_image_directory + 'no/')
yes_tumor_images = os.listdir(binary_image_directory + 'yes/')
binary_dataset = []
binary_label = []

INPUT_SIZE = 64

# Load images and labels for binary classification (tumor yes or no)
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(binary_image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        binary_dataset.append(np.array(image))
        binary_label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(binary_image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')

        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        binary_dataset.append(np.array(image))
        binary_label.append(1)

# Binary classification dataset
binary_dataset = np.array(binary_dataset)
binary_label = np.array(binary_label)

# Split the binary data into training and testing sets
binary_x_train, binary_x_test, binary_y_train, binary_y_test = train_test_split(binary_dataset, binary_label, test_size=0.2, random_state=0)

# Preprocess the binary data
binary_x_train = normalize(binary_x_train, axis=1)
binary_x_test = normalize(binary_x_test, axis=1)

binary_y_train = to_categorical(binary_y_train, num_classes=2)
binary_y_test = to_categorical(binary_y_test, num_classes=2)

# Model for binary classification
binary_model = Sequential()

binary_model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
binary_model.add(Activation('relu'))
binary_model.add(MaxPooling2D(pool_size=(2, 2)))

binary_model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
binary_model.add(Activation('relu'))
binary_model.add(MaxPooling2D(pool_size=(2, 2)))

binary_model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
binary_model.add(Activation('relu'))
binary_model.add(MaxPooling2D(pool_size=(2, 2)))

binary_model.add(Flatten())
binary_model.add(Dense(64))
binary_model.add(Activation('relu'))
binary_model.add(Dropout(0.5))
binary_model.add(Dense(2))  # Number of classes for binary classification (tumor yes or no)
binary_model.add(Activation('softmax'))

# Compile the binary model
binary_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the binary model
history=binary_model.fit(binary_x_train, binary_y_train, batch_size=16, verbose=1, epochs=50, validation_data=(binary_x_test, binary_y_test), shuffle=False)
# binary_model.save('BrainTumor10EpochsCategorical.h5')
# save_model(binary_model, 'BrainTumor10EpochsCategorical.h5', save_format='h5')
# model_save_location='trained_model.clf'
# if model_save_location is not None:
#     with open(model_save_location,'wb') as f:
#         pickle.dump(binary_model,f)
#     print("Training is complete your trained model has been saved to the given location!!")
    

test_loss, test_accuracy = binary_model.evaluate(binary_x_test, binary_y_test)
print(f"Test accuracy: {test_accuracy}")

# Generate predictions
predictions = binary_model.predict(binary_x_test)
y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

# Print classification report and confusion matrix
print(classification_report(binary_y_test, y_pred))
print(confusion_matrix(np.argmax(binary_y_test,axis=1), np.argmax(y_pred,axis=1)))

# Visualize model training history
# print(binary_model.history.history.keys())
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()