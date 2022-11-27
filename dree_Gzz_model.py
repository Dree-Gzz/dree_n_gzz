import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('DreeNGzz.model')

GENDER = ['Male','Female']

def load_image(file_path):
    image_array = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(image_array,(128,128))
    return new_array.reshape(1,128,128,1)

prediction = model.predict([load_image('image5.jpg')])

sex=int(np.round(prediction))
print('Gender:'+ GENDER[sex])