# -*- coding: utf-8 -*-
__author__ = 'imressed'

import cv2
import numpy as np
from PIL import Image
from cStringIO import StringIO


class pilLike(object):
    _instance = None

    def __init__(self):
        object.__init__(self)

    @property
    def size(self):
        return self._instance.size

    @property
    def shape(self):
        return self._instance.shape

    def new(self, size, channels, depth):
        self._instance = np.zeros(size + (channels,), depth)
        return self._instance

    def open(self, fl):
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

    def crop(self, x1,y1,x2,y2):
        self._instance = self._instance[y1:y2, x1:x2]
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

    #TODO: write this
    def save(self, format, file = _instance):
        cv2.imwrite(format, file)


    def show(self):
        cv2.imshow('ImageWindow',self._instance)
        cv2.waitKey()




a = pilLike()

pil_image = Image.open('1.jpg')

with open('1.jpg', 'rb') as img:
    b = a.open(img)
    a.save('test.jpg', img)


a.open(pil_image)
a.convert(cv2.COLOR_RGB2GRAY)

a.show()






