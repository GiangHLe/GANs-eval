'''
The code was modified from 'https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py'
'''
from numpy.core.defchararray import _startswith_dispatcher
import torch
from torch.nn.functional import adaptive_avg_pool2d

import os

import numpy as np

from scipy import linalg
from PIL import Image
from tqdm import tqdm
from glob import glob
from inception import InceptionV3

from utils import grab_image_path, get_transform

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
    def __init__(self, device=None, num_workers=0, dims=2048, batch_size=64, data_range=[0,1]):
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = num_workers
        self.dims = dims
        self.batch_size = batch_size
        self.output_range = data_range

        self.pred_arr = None
        self.num_samples = None
        self.load_extractor(dims)
        self.end_extract = False
        # self.online = True

        self.available_dataset_path = os.path.join(os.getcwd(), 'saved_dataset_statistic/')
        os.makedirs(self.available_dataset_path, exist_ok=True)
        exists_dataset = glob(os.path.join(self.available_dataset_path, '*.npy'))
        if len(exists_dataset) == 0:
            print('There is no available datset')
            self.available_dataset = []
        else:
            self.available_dataset = [name.split('/')[-1][:-4] for name in exists_dataset]
            print(f'Available datset: {self.available_dataset}')

    def load_extractor(self, dims):
        """Load the feature extracter model, InceptionV3 in this case"""
        assert dims in [64, 192, 768, 2048], 'Dims can only be in [64, 192, 768, 2048] synchronize to first max poolings, second max pooling, pre-aux classifier and final average pooling features, respectively.'
        print('Init Inception model to extract features ...')
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.feature_extractor = InceptionV3([block_idx]).to(self.device)
        self.feature_extractor.eval()

    @torch.no_grad()
    def get_data_distribution(self, data_path):
        # self.online = False
        image_path = grab_image_path(data_path)
        if len(image_path) < self.batch_size:
            self.batch_size = len(image_path)
        dataset = FidDatatset(
            image_path=image_path,
            transform=get_transform()
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
        # if the distribution is extracted directly from the training process, the 
        # problem may raise when number of synthesis image > number of line create 
        # by create_pred_map
        data = data.to(self.device)
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
        # if self.end_extract and self.online:
        #     self.mu, self.sigma = self.calculate_activation_statistics()
    
    def create_pred_map(self, num_samples):
        if num_samples < 50000:
            print('Warning: The author of the FID paper recommends using at least 50,000 samples to get the most correct FID score.')
        self.pred_arr = np.zeros((num_samples, self.dims))
        self.start_idx = 0
        self.num_samples = num_samples

    def calculate_activation_statistics(self, data_path=None):
        if data_path:
            self.get_data_distribution(data_path)
        mu = np.mean(self.pred_arr, axis=0)
        sigma = np.cov(self.pred_arr, rowvar=False)
        self.end_extract = False
        # del self.pred_arr # delete the used variable
        del self.start_idx
        return [mu, sigma]

    def extract_from_dataset(self, data_name='previous_data', data_path=None):
        npy_path = os.path.join(self.available_dataset_path, f'{data_name}.npy')
        if data_name in self.available_dataset:
            # if exist dataset, load the mu and sigma
            return self.load_numpy(npy_path)
        else:
            if not data_path:
                raise Exception('Need data_path')
            if not os.path.exists(data_path):
                print('Data path is not exists')
            statictis_data = self.calculate_activation_statistics(data_path=data_path)
            self.save_numpy(npy_path, [self.pred_arr, statictis_data[0], statictis_data[1]])
            del self.pred_arr
            print(f'Save the dataset statistics at {npy_path}')
            return statictis_data
    
    def compute_fid(self, data_dis1, data_dis2, eps=1e-16):
        mu1 = np.atleast_1d(data_dis1[0])
        mu2 = np.atleast_1d(data_dis2[0])

        sigma1 = np.atleast_2d(data_dis1[1])
        sigma2 = np.atleast_2d(data_dis2[1])

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    @staticmethod
    def load_numpy(path):
        with open(path, 'rb') as f:
            mu = np.load(f)
            sigma = np.load(f)
        return [mu, sigma]

    @staticmethod
    def save_numpy(path, data):
        with open(path, 'wb') as f:
            for d in data:
                np.save(f, d)

    @staticmethod
    def adjust_range(data, in_range=[0,1], out_range=[-1,1]):
        '''
        adjust the dynamic range, from input range to the target range
        In this work, the scaling was implemented in inception.py -> wait for the answer from the author of pytorch_fid to decide the transform
        '''
        if in_range!=out_range:
            scale = (np.float32(out_range[1])-np.float32(out_range[0]))/(
                np.float32(in_range[1]-np.float32(in_range[0])))
            bias = np.float32(out_range[0])-np.float32(in_range[0])*scale
            data = data*scale+bias
        return torch.clamp(data, min=out_range[0], max=out_range[1])

if __name__=='__main__':
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int, default=0,
                        help=('Number of processes to use for data loading. '
                            'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                            'By default, uses pool3 features'))
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths to the generated images or '
                            'to .npz statistic files'))
    args = parser.parse_args()

    path1, path2 = args.path
    metric = FID(device=args.device, 
                 num_workers=args.num_workers, 
                 dims=args.dims, 
                 batch_size=args.batch_size)
    
    dis_1 = metric.extract_from_dataset('data1', data_path=path1)
    dis_2 = metric.extract_from_dataset('data2', data_path=path2)
    fid = metric.compute_fid(dis_1, dis_2)
    print(fid)


    


        
        

    
    
