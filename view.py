from deeplabv3 import DeeplabV3

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
from PIL import Image, ImageQt
from PyQt5.QtGui import QPixmap

from PyQt5.QtGui import QImage, QPixmap
from PIL import Image

from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image

from PyQt5.QtGui import QPixmap, QImage
from PIL import ImageQt

import cv2

import numpy as np

deeplab = DeeplabV3()

def get_view(input_image):


    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(np.uint8(input_image))
    r_image = deeplab.detect_image(input_image)
    return r_image



def test_deeplab():
    input_image = cv2.imread('testImage/96000118_2348176048815390_2855122339223936182_n.jpg')

    processed_image = get_view(input_image)
    print(type(processed_image[0]))

    cv2.imshow('Processed Image', np.array(processed_image[0]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

