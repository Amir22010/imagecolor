# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:14:13 2019

@author: Amir.Khan
"""
# import the necessary packages
import os
from flask import Flask, render_template, request
from PIL import Image
from flask import send_file
import numpy as np
import cv2
import caffe
import base64

app = Flask(__name__)

# load our serialized black and white colorizer model and cluster
# center points from disk
print("[INFO] loading model...")
net = caffe.Net('model/colorization_deploy_v2.prototxt', 'model/colorization_release_v2.caffemodel', caffe.TEST)
pts = np.load('model/pts_in_hull.npy')

# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set([ 'png', 'jpg', 'jpeg', 'JPG'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    file = request.files['image']
    f = os.path.join(file.filename)
    image = cv2.imread(f)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    # resize the Lab image to 224x224 (the dimensions the colorization
    # network accepts), split channels, extract the 'L' channel, and then
    # perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # pass the L channel through the network which will *predict* the 'a'
    # and 'b' channel values
    'print("[INFO] colorizing image...")'
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # resize the predicted 'ab' volume to the same dimensions as our
    # input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    
    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the
    # predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    # convert the output image from the Lab color space to RGB, then
    # clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    
    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned
    # 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    image_content = cv2.imencode('.jpg', colorized)[1].tostring()
    encoded_image = base64.encodestring(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

    return render_template('index.html', image_to_show = to_send, init=True)


if __name__ == '__main__':
    app.run()
