import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=None,
    random_flip=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    # load triplanes                                                                                                                                       
    dataset = TriplaneDataset(                                                                                                                             
        image_size,                                                                                                                                        
        all_files,                                                                                                                                         
        classes=classes,                                                                                                                                   
        shard=MPI.COMM_WORLD.Get_rank(),                                                                                                                   
        normalize=True,                                                                                                                                    
        num_shards=MPI.COMM_WORLD.Get_size(),                                                                                                              
        random_crop=random_crop,                                                                                                                           
        random_flip=random_flip,                                                                                                                           
        stats_dir=stats_dir,                                                                                                                               
    )                                                                                                                                                      
               
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class TriplaneDataset(Dataset):                                                                                                                            
    def __init__(                                                                                                                                          
        self,                                                                                                                                              
        resolution,                                                                                                                                        
        image_paths,                                                                                                                                       
        # Whether to rescale individual channels to [-1, 1] based on their respective ranges                                             
        normalize=False,  
        classes=None,                                                                                                                                      
        shard=0,                                                                                                                                           
        num_shards=1,                                                                                                                                      
        stats_dir=None,                                                                                                                                    
    ):                                                                                                                                                     
        super().__init__()                                                                                                                                 
        self.resolution = resolution                                                                                                                       
        self.local_images = image_paths[shard:][::num_shards]                                                                                              
        self.normalize = normalize                                                                                                                         
        self.local_classes = None if classes is None else classes[shard:][::num_shards]                                                                    
        self.stats_dir = stats_dir                                                                                                                         
                                                                                                                                                           
        if self.normalize:                                                                                                                                 
            print('Will normalize triplanes in the training loop.')                                                                                        
            if self.stats_dir is None:                                                                                                                     
                raise Exception('Need to provide a directory of stats to use for normalization.')                                                          
            # Load in min and max numpy arrays (shape==[96,] - one value per channel) for normalization                                                    
            # self.min_values = np.load('util/min_values.npy').astype(np.float32).reshape(-1, 1, 1)  # should be (96, 1, 1)                                
            # self.max_values = np.load('util/max_values.npy').astype(np.float32).reshape(-1, 1, 1)                                                        
            self.min_values = np.load(f'{self.stats_dir}/lower_bound.npy').astype(np.float32).reshape(-1, 1, 1)                                            
            self.max_values = np.load(f'{self.stats_dir}/upper_bound.npy').astype(np.float32).reshape(-1, 1, 1)                                            
            self.range = self.max_values - self.min_values                                                                                                 
            self.middle = (self.min_values + self.max_values) / 2                                                                                          
        else:                                                                                                                                              
            print('Not using normalization in ds.')                                                                                                        
                                                                                                                                                           
    def __len__(self):                                                                                                                                     
        return len(self.local_images)                                                                                                                      
                                                                                                                                                           
    def __getitem__(self, idx):                                                                                                                            
        path = self.local_images[idx]                                                                                                                      
                                                                                                                                                           
        # Load np array                                                                                                                                    
        arr = np.load(path)                                                                                                                                
                                                                                                                                                           
        # Get rid of these extra operations I guess? (already inheriting variance from triplane generator)                                                 
        # Normalize individual channels                                                                                                                    
        arr = arr.astype(np.float32)  # / 127.5 - 1  <-- need to normalize the triplanes in their own way.                                                 
        arr = arr.reshape([-1, arr.shape[-2], arr.shape[-1]])                                                                                              
        if self.normalize:                                                                                                                                 
            arr = (arr - self.middle) / (self.range / 2)                                                                                                   
        out_dict = {}                                                                                                                                      
        if self.local_classes is not None:                                                                                                                 
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)                                                                              
        return arr, out_dict                                                                                                                               
                                                                                                                                                           
    def unnormalize(self, sample):                                                                                                                         
        sample = sample * (self.range / 2) + self.middle                                                                                                   
        return sample       


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results
