from keras import models
import numpy as np
import cv2

ix = -1
iy = -1

def callback(event,x,y,flags,param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y

cnn = models.load_model('models/tiny_cnn.h5')
camera = cv2.VideoCapture(0)

cv2.namedWindow('image')
cv2.setMouseCallback('image',callback)

size = 4
edge = 28*size

while True:
    image = cv2.pyrDown(camera.read()[1])
    cv2.imshow('image', image)

    if ix is not -1:
        """start model"""
        start_x = ix - edge//2
        end_x = ix + edge//2
        start_y = iy - edge//2
        end_y = iy + edge//2

        image = image[start_y:end_y, start_x:end_x]
        for i in range(size/2):
            image = cv2.pyrDown(image)

        cv2.imshow('input', image)
        image = np.expand_dims(image, axis = 0)/255.

        result = cnn.predict(image)[0]

        if result[0] > result[1]: print('Found finger!!!!')
        else: print('No more finger :(')
        """end model"""

        # print(result)



    ch = 0xFF & cv2.waitKey(1)
    if ch == 27: break
