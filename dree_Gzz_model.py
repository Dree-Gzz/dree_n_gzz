import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('DreeNGzz.model')

GENDER = ['Male','Female']
img = cv2.VideoCapture(0)
_, frame = img.read()

def load_image(file_path):
    # image_array = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    image_array = file_path.COLOR_RGB2GRAY(128)
    new_array = cv2.resize(image_array,(128,128))
    return new_array.reshape(1,128,128,1)

prediction = model.predict([load_image(frame)])

sex=int(np.round(prediction))
print('Gender:'+ GENDER[sex])