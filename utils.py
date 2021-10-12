import os
import torch
import numpy as np
from glob import glob
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

def adjust_range(input, in_range=[0,1], out_range=[-1,1]):
    '''
    adjust the dynamic range, from input range to the target range
    '''
    if in_range!=out_range:
        scale = (np.float32(out_range[1])-np.float32(out_range[0]))/(
            np.float32(in_range[1]-np.float32(in_range[0])))
        bias = np.float32(out_range[0])-np.float32(in_range[0])*scale
        out = input*scale+bias
    return torch.clamp(out, min=out_range[0], max=out_range[1])

def grab_image_path(data_path):
    from functools import reduce
    assert os.path.exists(data_path), 'Image directory does not exist'
    image_path = [glob(os.path.join(data_path, f'*.{ext}')) for ext in IMAGE_EXTENSIONS]
    image_path = reduce(lambda x,y: x+y, image_path)
    assert len(image_path)!=0, f'Only support the extensions {IMAGE_EXTENSIONS}, please check the extensions and the image directory path'
    return image_path