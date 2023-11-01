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
saved_model_dir='BrainTumor10EpochsCategorical.h5'
loaded_model=tf.keras.models.load_model(saved_model_dir)
print(loaded_model.history.history.keys())
