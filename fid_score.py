'''
The code was modified from 'https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py'
'''
import torch
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from inception import InceptionV3

from utils import grab_image_path

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

class FidDatatset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, batch_size=64):
        self.image_path = image_path
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, i):
        image = Image.open(self.image_path[i]).convert('RGB')     
        return self.transform(image)    

class FID():
    def __init__(self, device, num_workers, dims=2048, batch_size=64, data_range=[-1,1]):
        self.device = device
        self.num_workers = num_workers
        self.dims = dims
        self.batch_size = batch_size
        self.output_range = data_range

        self.pred_arr = None
        self.num_samples = None
        self.load_extractor(dims)
        self.end_extract = False

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
            transform=self.get_transform()
        )
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=self.num_workers)

        self.create_pred_map(len(image_path))
        
        for batch in tqdm(dataloader):
            batch = self.adjust_range(batch.to(self.device), out_range=self.output_range)
            self.get_feature_vector(batch)
            
    @torch.no_grad()
    def get_feature_vector(self, data):
        from torch.nn.functional import adaptive_avg_pool2d
        # if the distribution is extracted directly from the training process, the 
        # problem may raise when number of synthesis image > number of line create 
        # by create_pred_map
        if self.num_samples:
            if data.shape[0] + self.start_idx >= self.num_samples:
                remaining = self.num_samples-self.start_idx
                idx = torch.randperm(remaining)
                data = data[idx]
                self.end_extract = True
        # auto compute mean and covariance matrix when the prediction map is full
        pred = self.feature_extractor(data)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(data, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        self.pred_arr[self.start_idx:(self.start_idx+pred.shape[0])] = pred
        self.start_idx+=data.shape[0]
        if self.end_extract:
            self.calculate_activation_statistics()
    
    def create_pred_map(self, num_samples):
        if num_samples < 50000:
            print('Warning: The author of the FID paper recommends using at least 50,000 samples to get the most correct FID score.')
        self.pred_arr = np.empty((num_samples, self.dims))
        self.start_idx = 0
        self.num_samples = num_samples

    def calculate_activation_statistics(self, data_path=None):
        if data_path:
            self.get_data_distribution(data_path)
        mu = np.mean(self.pred_arr, axis=0)
        sigma = np.cov(self.pred_arr, rowvar=False)
        self.mu = mu
        self.sigma = sigma
        return mu, sigma

    @staticmethod
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

        
    


        
        

    
    
