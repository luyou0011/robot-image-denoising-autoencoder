 
#%% imports
import os
import torch
import torchvision
from utils import dataset
from model import autoencoder
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image
import torch.nn as nn
from torchvision.transforms import ToPILImage 

#%% load data

transform = torchvision.transforms.Resize((79, 79))

data_path = os.path.join(os.getcwd(), 'data', 'RobotMNIST', 'train', 'camera_1')
Datensatz = dataset.Robot(data_path, transform) # lade daten
train_dataset = dataset.Dataset(Datensatz.data_noise, Datensatz.label)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

test_data_path = os.path.join(os.getcwd(), 'data', 'RobotMNIST', 'test', 'camera_1')
test_dataset = dataset.Robot(test_data_path, transform)
test_dataset = dataset.Dataset(test_dataset.data_noise, test_dataset.label)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
# %% model
model = autoencoder.DenoisingModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% training
# Training configuration
num_epochs = 3
test_interval = 10
train_loss, train_accuracy = [], []
test_loss_value, t_accuracy = [], []

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    total = 0
    correct = 0
    mse_loss = nn.MSELoss()

    for i, (inputs_train, labels) in enumerate(trainloader, 1):
        optimizer.zero_grad()
        outputs_train = model(inputs_train)
        loss = criterion(outputs_train, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss.append(loss.item())

        # Compute accuracy using mean squared error (MSE) loss
        mse = mse_loss(outputs_train, labels)
        total += labels.size(0)
        correct += mse.item()  # Use MSE as an indicator of accuracy
        train_accuracy.append(100.0 * (1 - (correct / total)))  # Invert the MSE to get accuracy

        if i % test_interval == 0:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for inputs_test, labels in testloader:
                    outputs_test = model(inputs_test)
                    test_loss += criterion(outputs_test, labels).item()
                    mse = mse_loss(outputs_test, labels)
                    test_total += labels.size(0)
                    test_correct += mse.item()

            t_accuracy.append(100.0 * (1 - (test_correct / test_total)))  # Invert the MSE to get accuracy
            test_loss_value.append(test_loss)
    #print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss}, Test Accuracy: {correct[-1]}%")
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss}, Test Accuracy: {t_accuracy[-1]}%")

#%% Visiualisieren
# Plot loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(2,2,1)
plt.plot(train_loss)
plt.title("Train Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.subplot(2, 2, 2)
plt.plot(train_accuracy)
plt.title("Train Accuracy")
plt.xlabel("Iterations (every 10)")
plt.ylabel("Accuracy (%)")

plt.subplot(2, 2, 3)
plt.plot(test_loss_value)
plt.title("Test Loss")
plt.xlabel("Iterations (every 10)")
plt.ylabel("Loss (%)")
plt.subplot(2, 2, 4)
plt.plot(t_accuracy)
plt.title("Test Accuracy")
plt.xlabel("Iterations (every 10)")
plt.ylabel("Accuracy (%)")
plt.show


def plot_single_image(input_image_train, output_image_train):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(input_image_train.permute(1, 2, 0).detach().numpy())
    #ax[0].axis('off')
    ax[0].set_title('Input train Image')
    ax[1].imshow(output_image_train.permute(1, 2, 0).detach().numpy())
    #ax[1].axis('off')
    ax[1].set_title('Output train Image')
    plt.show()

# Select a random index from the batch
random_index = random.randint(0, len(inputs_train) - 1)

# Get the input and output images based on the random index
input_image_train = inputs_train[random_index]
output_image_train = outputs_train[random_index]

# Plot the selected input image and its corresponding output image
plot_single_image(input_image_train, output_image_train)

def plot_single_image(input_image_test, output_image_test):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(input_image_test.permute(1, 2, 0).detach().numpy())
    ax[0].set_title('Input test Image')
    ax[1].imshow(output_image_test.permute(1, 2, 0).detach().numpy())
    ax[1].set_title('Output test Image')
    plt.show()

# Select a random index from the batch
random_index = random.randint(0, len(inputs_test) - 1)

# Get the input and output images based on the random index

input_image_test = inputs_test[random_index]
output_image_test = outputs_test[random_index]

# Plot the selected input image and its corresponding output image
plot_single_image(input_image_test, output_image_test)

#%% create path in resluts and save the accuracy und loss graphs in it

# Convert tensors to PIL images
to_pil = ToPILImage()
input_image_train_pil = to_pil(input_image_train)
output_image_train_pil = to_pil(output_image_train)
input_image_test_pil = to_pil(input_image_test)
output_image_test_pil = to_pil(output_image_test)

from datetime import datetime

current_day = datetime.now()

date = current_day.strftime("%D%H%M").replace('/', '_')

path = os.path.join(".\\training", date)
os.mkdir(path)

input_image_train.savefig(os.path.join(path, 'train_loss.png')) 
output_image_train.savefig(os.path.join(path, 'train_acc.png')) 
input_image_test.savefig(os.path.join(path, 'test_loss.png')) 
output_image_test.savefig(os.path.join(path, 'test_acc.png')) 

# save data
torch.save(train_loss, 'loss_data.pt')
torch.save(train_accuracy, 'accuracy_data.pt')
torch.save(test_loss_value, 'test_loss_data.pt')
torch.save(t_accuracy, 'test_accuracy_data.pt')

