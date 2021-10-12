'''
The code was modified from 'https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py'
'''
import torch
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from inception import InceptionV3

from utils import adjust_range, grab_image_path

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

class FidDatatset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, data_range=[-1,1], batch_size=64):
        self.image_path = image_path
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, i):
        image = Image.open(self.image_path[i]).convert('RGB')     
        return adjust_range(self.transform(image))    

class FID():
    def __init__(self, device, num_workers, dims=2048, batch_size=64):
        self.device = device
        self.num_workers = num_workers
        self.dims = dims
        self.batch_size = batch_size
        self.load_extractor(dims)
    
    def load_extractor(self, dims):
        """Load the feature extracter model, InceptionV3 in this case"""
        assert dims in [64, 192, 768, 2048], 'Dims can only be in [64, 192, 768, 2048] synchronize to first max poolings, second max pooling, pre-aux classifier and final average pooling features, respectively.'
        print('Init Inception model to extract features ...')
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.feature_extractor = InceptionV3([block_idx]).to(self.device)
        self.feature_extractor.eval()

    def get_transform(self):
        from torchvision import transforms as T
        return T.Compose([
            T.ToTensor(),
            T.Resize(size=(299,299), interpolation=T.InterpolationMode.BILINEAR)
        ])

    @torch.no_grad()
    def get_data_distribution(self, data_path):
        image_path = grab_image_path(data_path)
        if len(image_path) < self.batch_size:
            self.batch_size = len(image_path)
        dataset = FidDatatset(
            image_path=image_path,
            transform=self.get_transform(),
            data_range=[-1,1]
        )
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=self.num_workers)

        self.pred_arr = np.empty((len(image_path), self.dims))
        start_idx = 0
        
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            pred = self.get_feature_vector(batch)
            self.pred_arr[start_idx:(start_idx+pred.shape[0])] = pred
            start_idx+=pred.shape[0]
        
    @torch.no_grad()
    def get_feature_vector(self, data):
        from torch.nn.functional import adaptive_avg_pool2d
        pred = self.feature_extractor(data)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(data, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        return pred
    
    def 


        
        

    
    
