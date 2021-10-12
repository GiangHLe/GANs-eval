import torch
from fid_score import FID


device = 'cuda' if torch.cuda.is_available() else 'cpu'

metric = FID(device = device, num_workers=0)

# test_get_data_distribution (success)
image_path = 'C:/Users/giang/Desktop/test_image/'
metric.get_data_distribution(data_path = image_path)
print(metric.pred_arr.shape)



