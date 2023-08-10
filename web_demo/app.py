from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import cv2

app = Flask(__name__)

new_model = load_model('../models/imageclassfilter.h5')
print(new_model.summary())



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html', msg='_', msg2='_', img="https://reactnativecode.com/wp-content/uploads/2018/02/Default_Image_Thumbnail.png")

    image = request.files['file']
    img = Image.open(image)
    img = img.save("./static/images/test.png")

    img = cv2.imread('./static/images/test.png')
    resize = tf.image.resize(img, (256, 256))
    yhat = new_model.predict(np.expand_dims(resize/255, 0))
    if yhat > 0.5: 
        message = 'Predicted class is PNEUMONIA'
    else:
        message = 'Predicted class is NORMAL'


    return render_template('index.html', msg=message, msg2=yhat, img='./static/images/test.png')

if __name__ == '__main__':
    app.run()
