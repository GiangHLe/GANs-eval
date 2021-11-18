from fid_score import FID

import torch
metric = FID()
dataset_path = 'path to dataset images'
# exist dataset: FFHQ -> read the name of npy file
metric.get_base_distribution(dataset_name, dataset_path) # get mu and covariance matrix of the dataset, if exist, load it

# Use while training
# don't wanna dump images
with torch.no_grad():
    while not metric.end_extract():
        noise = generate_latent_vector()
        fake_images = generator(noise)
        metric.get_feature_vector(noise)
    # when the loop is finish, the mu and variance will be compute
    fid_score = metric.compute_FID() # return FID score, delete the parameters in get_feature_vector

# compute mu and covariance directly from a dataset_path
get_data_distribution(dataset_name='abc') #-> dump mean and covariance matrix to npy file under the name of new dataset, stack the mu and covariance then dump to one npy file only 

# fid_score.py


