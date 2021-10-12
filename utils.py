import os
import torch
import numpy as np
from glob import glob
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

def grab_image_path(data_path):
    from functools import reduce
    assert os.path.exists(data_path), 'Image directory does not exist'
    image_path = [glob(os.path.join(data_path, f'*.{ext}')) for ext in IMAGE_EXTENSIONS]
    image_path = reduce(lambda x,y: x+y, image_path)
    assert len(image_path)!=0, f'Only support the extensions {IMAGE_EXTENSIONS}, please check the extensions and the image directory path'
    return image_path