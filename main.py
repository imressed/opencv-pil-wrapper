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

    def new(self, size, channels, depth):
        self._instance = np.zeros(size + (channels,), depth)
        return self._instance

    def open(self, fl):
        if isinstance(fl, basestring):
            self._instance = cv2.imread(fl,cv2.IMREAD_UNCHANGED)
        if isinstance(fl, file):
            file_bytes = np.asarray(bytearray(fl.read()), dtype=np.uint8)
            self._instance = cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_UNCHANGED)
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



    def show(self):
        cv2.imshow('ImageWindow',self._instance)
        cv2.waitKey()




a = pilLike()
with open('1.jpg', 'rb') as img:
    a.open(img)


pil_image = Image.open('1.jpg')
a.open(pil_image)

a.crop(0,300,300,200)

a.show()





