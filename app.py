# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:04:45 2021

@author: Jhanak Gupta
"""

import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from flask import Flask , request, render_template
# secure_filename will ensure the images uploaded will get saved in the uploads folder


app = Flask(__name__)  #our flask app
model = load_model("malaria.h5")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        # This extracts the filepath of the image uploaded
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        # This appends the original filepath to that of uploads
        filepath = os.path.join(basepath,'static/uploads',f.filename)
        print("upload folder is ", filepath)
        # This saves the filepath of the image
        f.save(filepath)
        
        file = "/static/uploads/" + f.filename
        
        # Testing the model
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        preds = model.predict(x)
        
        print("prediction",preds)
            
        index = ["infected", "uninfected"]
        
        print(np.argmax(preds))
        
        result = "The prediction is : " + str(index[np.argmax(preds)])
        
    return render_template("index.html", result=result, uploaded_image=file)

if __name__ == '__main__':
    app.run(debug = True, threaded = False)