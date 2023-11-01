import cv2
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the saved model
saved_model_dir='BrainTumor10EpochsCategorical.h5'
loaded_model=tf.keras.models.load_model(saved_model_dir)

# Directory containing the testing dataset
test_image_directory = 'TEST/'  # Replace with the path to your testing dataset

test_images = os.listdir(test_image_directory)
test_dataset = []
test_labels = []

INPUT_SIZE = 64

# Load testing images and labels
for image_name in test_images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(os.path.join(test_image_directory, image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        test_dataset.append(np.array(image))
        label = 1 if 'y' in image_name else 0
        test_labels.append(label)

# Preprocess the testing data
test_dataset = np.array(test_dataset)

test_labels = np.array(test_labels)
test_dataset = test_dataset / 255.0  # Normalize the pixel values

# Predict using the loaded model
predictions = loaded_model.predict(test_dataset)

# Convert predicted probabilities to binary labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate the confusion matrix
# confusion = confusion_matrix(test_labels, predicted_labels)
confusion=confusion_matrix(test_labels,predicted_labels)

# Calculate and print sensitivity, accuracy, ppv, f1score, and specificity
accuracy = accuracy_score(test_labels, predicted_labels)
sensitivity = recall_score(test_labels, predicted_labels)
ppv = precision_score(test_labels, predicted_labels)
f1score = f1_score(test_labels, predicted_labels)
specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])

print("Confusion Matrix:")
print(confusion)
print("Accuracy:", accuracy)
print("Sensitivity (Recall):", sensitivity)
print("Precision (PPV):", ppv)
print("F1 Score:", f1score)
print("Specificity:", specificity)
