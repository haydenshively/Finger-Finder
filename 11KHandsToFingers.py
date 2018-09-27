import numpy as np
import cv2
import random

class Constants():
    """COLORING"""
    # to segment hand from background, hue will be compared to this value (0 to 180)
    hand_hue_approx = 10.0
    hand_hue_approx_arr = np.array([hand_hue_approx], dtype = 'float')
    # anywhere (angle between hand_hue_approx and pixel_hue) > this value is labelled background
    max_angular_distance = 20
    # anywhere pixel_sat < this value is labelled background
    min_saturation = 25

    """SIZING"""
    min_contour_area = 1000
    roi_edge_length = 28*4# should be multiple of 28
    roi_shape = (roi_edge_length, roi_edge_length)
    roi_vertical_offset = 28

class Reader(object):
    def __init__(self, start = 0, end = 11000):
        self.start = start
        self.id = start
        self.end = end

    @property
    def next_image(self):
        self.id += 1
        zeros = '0'*(7 - len(str(self.id)))
        filename = '11KHands/Hand_' + zeros + str(self.id) + '.jpg'

        image = cv2.imread(filename)
        if image is None: image = self.next_image
        return image

    @property
    def random_image(self):
        id = str(random.randint(self.start, self.end))
        zeros = '0'*(7 - len(id))
        filename = 'Hand_' + zeros + id + '.jpg'

        image = cv2.imread(filename)
        if image is None: image = get_rand_image()
        return image

    def write(self, title, np_array):
        filename = '11KFingers/' + title
        np.save(filename, np_array)

def ieee_remainder(x, y):
    n = x/y
    return x - n.round(0)*y

def angular_distance(anglesA, anglesB, range = 360):
    return ieee_remainder(anglesA - anglesB, range)

def crop(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def square_around(x, y, in_image, length):
    return crop(in_image, x - length//2, y - length//2, length, length)

def resize(image):
    while image.shape[0] > 28:
        image = cv2.pyrDown(image)
    return image


if __name__ == '__main__':
    reader = Reader()
    i = 0

    while reader.id < reader.end:
        src = reader.next_image
        # switch colorspaces
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        # compute priors that will be helpful when thresholding
        theta_from_red = angular_distance(h.astype('float'), Constants.hand_hue_approx_arr, range = 180)
        theta_from_red = np.absolute(theta_from_red).astype('uint8')
        # perform thresholding to make hand white, background black
        h[theta_from_red > Constants.max_angular_distance] = 255
        h[s < Constants.min_saturation] = 255
        h = 255 - cv2.threshold(h, 127, 255, cv2.THRESH_BINARY)[1]

        # find and filter contours
        _, contours, _ = cv2.findContours(h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:

            M = cv2.moments(contour)
            area = M['m00']

            if area > Constants.min_contour_area:
                hull = cv2.convexHull(contour, returnPoints = True)
                # get fingertip
                fingertip_x, fingertip_y = contour[contour[:,:,1].argmax()][0]
                fingertip = square_around(fingertip_x, fingertip_y - Constants.roi_vertical_offset, src, Constants.roi_edge_length)
                # get hand center
                hand_x, hand_y = (int(M['m10']/area), int(M['m01']/area))
                hand = square_around(hand_x, hand_y, src, Constants.roi_edge_length)
                # ensure correct size
                if fingertip.shape[:2] != Constants.roi_shape or hand.shape[:2] != Constants.roi_shape: continue
                fingertip = resize(fingertip)
                hand = resize(hand)
                # generate random color
                solid = np.full((28, 28, 3), np.random.randint(0, 255, size = 3, dtype = 'uint8'), dtype = 'uint8')

                reader.write(str(i) + 'Finger', fingertip)
                i += 1
                reader.write(str(i) + 'Skin', hand)
                i += 1
                reader.write(str(i) + 'Solid', solid)
                i += 1

                # cv2.imshow('fingertip', fingertip)
                # cv2.imshow('hand', hand)
        if (reader.id%100 == 0): print(reader.id/float(reader.end))
        # ch = cv2.waitKey(0)
        # if ch == 27: break
