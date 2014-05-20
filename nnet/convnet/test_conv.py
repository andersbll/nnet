import numpy as np
import scipy.ndimage.filters

import pyximport
pyximport.install()
from conv import conv_bc01


def test_conv():
    img = np.eye(8)
    img = np.random.randn(4, 4)
    imgs = img[np.newaxis, np.newaxis, ...]
    filter = np.eye(7)
    filters = filter[np.newaxis, np.newaxis, ...]
    convout = np.empty_like(imgs)
    conv_bc01(imgs, filters, convout)
    print(convout[0, 0])

    convout = scipy.ndimage.filters.convolve(img, filter, mode='constant')
    print(convout)

if __name__ == '__main__':
    test_conv()
