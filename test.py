import keras
import tensorflow as tf
from generator import test_generator
from keras.models import load_model

filenames = test_generator.filenames
nb_samples = len(filenames)

model = keras.models.load_model('./weights/weights.h5', custom_objects={'ConvCapsuleLayer':ConvCapsuleLayer})
results = model.predict_generator(test_generator, steps=nb_samples)