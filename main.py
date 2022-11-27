from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import tensorflow as tf

class MainApp(App):
    def build(self):
        self.model = tf.keras.models.load_model('DreeNGzz.model')
        window = BoxLayout(orientation = 'vertical')
        self.image = Image()
        window.add_widget(self.image)
        self.gender = Label(
            text= 'Gender: ',
            pos_hint = {'center_x':.5,'center_y':.5},
            size_hint = (None,None)
        )
        window.add_widget(
            self.gender
        )
        self.save_button = Button(
                text = 'Take Picture',
                pos_hint = {'center_x':.5,'center_y':.5},
                size_hint=(None,None)
            )
        self.save_button.bind(on_press = self.save_image)
        window.add_widget(
            self.save_button
        )
        
        self.capture = cv2.VideoCapture(0)#opens camera
        Clock.schedule_interval(self.load_video,1.0/30.0)# apply 30fps
        return window
    def load_video(self,*args):
        ret, frame = self.capture.read()
        self.image_frame = frame
        buffer = cv2.flip(frame,0).tobytes()
        texture = Texture.create(size = (frame.shape[1],frame.shape[0]),
        colorfmt = 'bgr')
        texture.blit_buffer(buffer, colorfmt = 'bgr',
        bufferfmt = 'ubyte')#converts byte data to color format 
        #and buffer format in rapid format in real time
        self.image.texture = texture
    def save_image(self,*args):
        cv2.imwrite('image.jpg', self.image_frame) 

        def load_image(file_path):
            image_array = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(image_array,(128,128))
            return new_array.reshape(1,128,128,1)

        prediction = self.model.predict([load_image('image.jpg')])

        sex=int(np.round(prediction))
        GENDER = ['Male','Female']
        self.gender.text = 'Gender:'+ GENDER[sex]
        
if __name__ == '__main__':
    MainApp().run()