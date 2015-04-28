# -*- coding: utf-8 -*-
__author__ = 'imressed'

import cv2
import numpy as np
from PIL import Image as PILImage
from cStringIO import StringIO


class Image(object):
    _instance = None
    _mode = None

    NEAREST = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC
    LANCZOS = cv2.INTER_LANCZOS4

    def __init__(self):
        object.__init__(self)

    @property
    def size(self):
        return self._instance.size

    @property
    def mode(self):
        if self._mode:
            return self._mode
        else:
            raise ValueError('No mode specified.')

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

    def _get_converting_flag(self, mode):
        mode = mode.upper()
        inst = self._mode.upper()
        converting_table = {
            'L':{
                'RGB':cv2.COLOR_GRAY2BGR
            },
            'RGB':{
                'L':cv2.COLOR_BGR2GRAY,
                'LAB':cv2.COLOR_BGR2LAB,
                'HSV':cv2.COLOR_BGR2HSV,
                'YCBCR':cv2.COLOR_BGR2YCR_CB
            },
            'LAB':{
                'RGB':cv2.COLOR_LAB2BGR
            },
            'HSV':{
                'RGB':cv2.COLOR_HSV2BGR
            },
            'YCBCR':{
                'RGB':cv2.COLOR_YCR_CB2BGR
            }
        }
        if converting_table.has_key(inst):
            if converting_table[inst].has_key(mode):
                return converting_table[inst][mode]
            else:
                raise ValueError('You can not convert image to this type')
        else:
            raise ValueError('This image type can not be converted')


    def new(self, mode, size, color=(0,0,0)):
        self._mode = mode
        channels, depth = self._get_channels_and_depth(mode)
        self._instance = np.zeros(size + (channels,), depth)
        self._instance[:,0:] = color
        return self._instance

    def open(self, fl, mode='r'):
        self._mode = None
        if isinstance(fl, basestring):
            self._instance = cv2.imread(fl,cv2.IMREAD_UNCHANGED)
            pil_instance = PILImage.open(fl)
            self._mode = pil_instance.mode
            return self._instance
        if isinstance(fl, file):
            pil_instance = PILImage.open(fl)
            self._mode = pil_instance.mode
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

    def convert(self, mode):
        if self._mode.upper() == mode.upper():
            return self._instance
        flag = self._get_converting_flag(mode)
        self._instance = cv2.cvtColor(self._instance, flag)
        return self._instance

    #TODO: rewrite!
    def save(self, fp, format=None):
        if isinstance(fp,basestring):
            cv2.imwrite(fp, self._instance)
            return None
        if isinstance(fp,file):
            fl = open(format, 'w')
            fl.write(fp.read())
            fl.close()
            return None
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

    def rectangle(self, xy, fill=None, outline=None):
        if fill:
            cv2.rectangle(self._img_instance,xy[0], xy[1], fill, thickness=cv2.cv.CV_FILLED)
        if outline:
            cv2.rectangle(self._img_instance,xy[0], xy[1], outline, 3)




a = Image()

a.new('rgb',(300,300),(0,255,0))
print(a.mode)
#a.crop((100,100,400,500))
#a.resize((200,200),Image.BILINEAR)
#a.convert('hsv')

pil_image = PILImage.open('1.jpg')

with open('1.jpg', 'rb') as img:
    #a.save(img, 'text.jpg')
    b = a.open(img)

a.open(pil_image)



d = ImageDraw(a.get_instance)
b = a.copy()
d.rectangle([(10,10),(100,200)], fill=(255,255,0))

a.save('a.jpg')

a.show()






