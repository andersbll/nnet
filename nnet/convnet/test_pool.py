import numpy as np

import pyximport
pyximport.install()
from pool import pool_bc01


def test_pool():
    img_h = 3
    img_w = 3
    stride_y, stride_x = 1, 1
    pool_h, pool_w = 2, 2

    poolout_h = img_h // stride_y
    poolout_w = img_w // stride_x

    imgs = np.eye(img_h, img_w)
    imgs[0, 1] = 2
    imgs = imgs[np.newaxis, np.newaxis, ...]
    poolout = np.empty((poolout_h, poolout_w))[np.newaxis, np.newaxis, ...]
    switches = np.empty(poolout.shape + (2,), dtype=np.int)

    pool_bc01(imgs, poolout, switches, pool_h, pool_w, stride_y, stride_x)
    print(imgs)
    print(poolout)
    print(switches)


if __name__ == '__main__':
    test_pool()
