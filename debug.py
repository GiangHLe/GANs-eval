import torch
import cv2
import numpy as np

from fid_score import FID


device = 'cuda' if torch.cuda.is_available() else 'cpu'

metric = FID(device = device, num_workers=0)

# # test get_data_distribution (success)
# image_path = 'C:/Users/giang/Desktop/test_image/'
# data_statictis = metric.get_data_distribution(data_path = image_path)
# print(data_statictis[0].shape)

# # test get feature vector during training
# image_path = 'C:/Users/giang/Desktop/test_image/cat.png'
# image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)

# tensor = torch.Tensor(image)
# tensor = tensor.div_(255.).permute(2,0,1)
# tensor = metric.adjust_range(tensor, in_range=[0,1])
# tensor = tensor.repeat(3,1,1,1)
# '''
# remember that images need to be in range [-1,1]
# '''

# # create predict map
# metric.create_pred_map(10)
# # extract the features map
# while not metric.end_extract:
#     metric.get_feature_vector(tensor)
# statictis_data = metric.calculate_activation_statistics()

# print(statictis_data[0].shape)


truth_path = 'D:/DATA/FFHQ/for_new/test_fid/0-499/'

syn_path = 'D:/DATA/FFHQ/for_new/test_fid/500-999'

# for debug
from glob import glob
from utils import grab_image_path

fake_syn_image_path = grab_image_path(syn_path)


# for real


