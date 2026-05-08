#Zwei Schichten
# %% imports

import torch.nn as nn

# %% model

class DenoisingModel(nn.Module):
    def __init__(self):
        #Die Klasse DenoisingModel veräbt alle Eingenschaften der basis Klasse nn.Module
        super(DenoisingModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(2, 2, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 2, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 3, kernel_size=5, stride=1),
            nn.Sigmoid()            
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# %% test

if __name__ == '__main__':
    model=DenoisingModel()
############################################################
# #Drei Schichten
# # %% imports

# import torch.nn as nn

# class DenoisingModel(nn.Module):
#     def __init__(self):
#         super(DenoisingModel, self).__init__()
        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 4, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(4, 8, kernel_size=4, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, kernel_size=3, stride=1),
#             nn.ReLU(),
#         )
        
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 4, kernel_size=4, stride=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(4, 3, kernel_size=5, stride=1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# # Test the model
# if __name__ == '__main__':
#     model = DenoisingModel()
#     print(model)
    

    
############################################################
# #Vier Schichten
# import torch.nn as nn

# class DenoisingModel(nn.Module):
#     def __init__(self):
#         super(DenoisingModel, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# # Test the model
# if __name__ == '__main__':
#     model = DenoisingModel()
#     print(model)
   
