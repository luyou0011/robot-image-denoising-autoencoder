import os
import torch
from torchvision import transforms
from PIL import Image

# %%

class Robot:
    def __init__(self, path: str, transform=None):
        self.path = path
        self.img2tensor = transforms.ToTensor() # convert to tensor
        self.transform = transform # augmentation
        self.data_noise, self.label= self.Image_to_pixels() # load data
        
    def Image_to_pixels(self):
        '''
        Load images from the defined path as tensors and store with and without noise.

        Returns
        -------
        noisy_img_data_list : list
            Includes the synthetically generated noisy image.
        img_data_list : list
            Includes the clean image data.

        '''
        files = os.listdir(self.path)
        img_data_list, noisy_img_data_list = [], [] # create image storages
        for file in files:
            image = self.img2tensor(Image.open(os.path.join(self.path, file))) # load img as tensor
            if self.transform: # use augmentation
                image = self.transform(image.squeeze(0))
                
            mean, stddev = 0.0, 0.1 
            noise = torch.randn_like(image) * stddev + mean
            noisy_image = image + noise # add noise
            noisy_image = torch.clamp(noisy_image, 0, 1) # clamp between 0 and 1
            
            noisy_img_data_list.append(noisy_image) # add to list
            img_data_list.append(image) # add to list
            
        return noisy_img_data_list, img_data_list 
    
# iterable dataset
class Dataset(torch.utils.data.Dataset):
    '''
    Create a basic torch dataset.

    Parameters
    ----------
    data_noise : list
        DESCRIPTION.
    label : list
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    def __init__(self, data_noise, label):
        self.data = data_noise
        self.labels = label

    def __len__(self):
        '''
        Returns the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Parameters
        ----------
        idx : integer

        Returns
        -------
        TYPE
            Tensor.
        TYPE
            Tensor.

        '''
        return self.data[idx], self.labels[idx]

# %%
if __name__ == '__main__':
    path = os.getcwd()
    
