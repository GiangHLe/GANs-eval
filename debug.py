import torch
import cv2
import numpy as np

from fid_score import FID


device = 'cuda' if torch.cuda.is_available() else 'cpu'

metric = FID(device = device, num_workers=0)

# test get_data_distribution (success)
# image_path = 'C:/Users/giang/Desktop/test_image/'
# metric.get_data_distribution(data_path = image_path)
# print(metric.pred_arr.shape)

# test get feature vector during training
image_path = 'C:/Users/giang/Desktop/test_image/cat.png'
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)

tensor = torch.Tensor(image)
tensor = tensor.div_(255.).permute(2,0,1)
tensor = metric.adjust_range(tensor, in_range=[0,1])
tensor = tensor.repeat(3,1,1,1)
'''
remember that images need to be in range [-1,1]
'''

# create predict map
metric.create_pred_map(5)
# extract the features map
while not metric.end_extract:
    metric.get_feature_vector(tensor)

print(metric.pred_arr.shape)
print(np.sum(metric.pred_arr))
print(metric.start_idx)
print(metric.mu.shape)
print(metric.sigma.shape)
# print('Debug for mean and covariance')
# m , cov = metric.calculate_activation_statistics('C:/Users/giang/Desktop/test_image/')
# print(m.shape)
# print(cov.shape)


