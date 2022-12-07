from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import keras
import sys

cascPath = sys.argv[0]
class MainApp(App):
    def build(self):
        
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = keras.models.load_model('DreeNGzz.model')
        self.GENDER = ['Male','Female']
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
        ret, frame = self.capture.read()# reading one frame
        self.image_frame = frame
        buffer = cv2.flip(frame,0).tobytes()
        texture = Texture.create(size = (frame.shape[1],frame.shape[0]),
        colorfmt = 'bgr')
        texture.blit_buffer(buffer, colorfmt = 'bgr',
        bufferfmt = 'ubyte')#converts byte data to color format 
        #and buffer format in rapid format in real time
        self.image.texture = texture
    def save_image(self,*args):
        #detecting face in the frame
        # Convert into grayscale
        gray = cv2.cvtColor(self.image_frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,1.4,4)
        for (x, y, w, h) in faces: 
            # increase size of head being detected
            x=x-100 
            y = y-100
            self.new_image = self.image_frame[y:y + h+200, x:x + w+200]
            
        def load_image():
            image_array = cv2.cvtColor(self.new_image,cv2.COLOR_BGR2GRAY)
            new_array = cv2.resize(image_array,(128,128))
            cv2.imwrite('image.jpg', image_array)
            return new_array.reshape(1,128,128,1)

        prediction = self.model.predict([load_image()])

        sex=int(np.round(prediction))
        
        self.gender.text = 'Gender:'+ self.GENDER[sex]
        
if __name__ == '__main__':
    MainApp().run()
