# -*- coding: utf-8 -*-
__author__ = 'imressed'

import cv2
import numpy as np
from PIL import Image as PILImage
from cStringIO import StringIO


class Image(object):
    _instance = None

    def __init__(self):
        object.__init__(self)

    @property
    def size(self):
        return self._instance.size

    @property
    def shape(self):
        return self._instance.shape

    @property
    def get_instance(self):
        return self._instance

    def _get_channels_and_depth(self, mode):
        mode = str(mode).upper()
        if mode == '1':
            return 1 , np.bool
        if mode == 'L':
            return 1, np.uint8
        if mode == 'P':
            return 1, np.uint8
        if mode == 'RGB':
            return 3, np.uint8
        if mode == 'RGBA':
            return 4, np.uint8
        if mode == 'CMYK':
            return 4, np.uint8
        if mode == 'YCBCR':
            return 3, np.uint8
        if mode == 'LAB':
            return 3, np.uint8
        if mode == 'HSV':
            return 3, np.uint8
        if mode == 'I':
            return 1, np.int32
        if mode == 'F':
            return 1, np.float32

        raise ValueError('Your mode name is incorect.')

    def new(self, mode, size, color=(0,0,0)):
        channels, depth = self._get_channels_and_depth(mode)
        self._instance = np.zeros(size + (channels,), depth)
        self._instance[:,0:] = color
        return self._instance

    def open(self, fl, mode='r'):
        if isinstance(fl, basestring):
            self._instance = cv2.imread(fl,cv2.IMREAD_UNCHANGED)
            return self._instance
        if isinstance(fl, file):
            file_bytes = np.asarray(bytearray(fl.read()), dtype=np.uint8)
            self._instance = cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_UNCHANGED)
            return self._instance
        if hasattr(fl, 'mode'):
            image = np.array(fl)
            self._mode = fl.mode
            if self._mode == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self._instance = image
            return self._instance

    def crop(self, box):
        """box is a tuple = left, upper, right, lower"""
        self._instance = self._instance[box[1]:box[3], box[0]:box[2]]
        return self._instance

    def copy(self):
        return self._instance.copy()

    def close(self):
        cv2.destroyAllWindows()
        return None

    def resize(self, size, interpolation = cv2.INTER_LINEAR):
        self._instance = cv2.resize(self._instance, size, interpolation = interpolation)
        return self._instance

    def convert(self, flag):
        self._instance = cv2.cvtColor(self._instance, flag)
        return self._instance

    #TODO: rewrite!
    def save(self, format, fl = None):
        if fl == None:
            cv2.imwrite(format, self._instance)
            return None
        cv2.imwrite(format, fl)
        return None

    def show(self):
        cv2.imshow('ImageWindow',self._instance)
        cv2.waitKey()


class ImageDraw(object):
    _img_instance = None

    def __init__(self, img):
        self._img_instance = img

    def point(self, xy, fill, width=3):
        for elem in xy:
            cv2.line(self._img_instance,elem, elem, fill,width)

    def line(self, xy, color, width):
        cv2.line(self._img_instance,xy[0], xy[1], color, width)

    def rectangle(self, xy, fill, outline):
        cv2.rectangle(self._img_instance,xy[0], xy[1], fill, outline)




a = Image()

#a.new('rgb',(300,300),(0,255,0))


pil_image = PILImage.open('1.jpg')

with open('1.jpg', 'rb') as img:
    b = a.open(img)



a.open(pil_image)
d = ImageDraw(a.get_instance)
b = a.copy()
d.rectangle([(10,10),(100,200)], (255,255,0), 5)



a.show()






