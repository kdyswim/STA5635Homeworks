import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class LoadData(Dataset):
  def __init__(self, dir):
    self.dir = dir
    self.img = os.listdir(dir)

  def __len__(self):
    return len(self.img)

  def __getitem__(self, idx):
    img_name = self.img[idx]
    img_path = os.path.join(self.dir, img_name)

    res = float(img_name.split('_')[0]) / 100
    img = Image.open(img_path).convert('RGB')
    img = transforms.ToTensor()(img)

    for i in range(3):
      img[i] = (img[i] - img[i].mean()) / (img[i].std())

    # print(f"Image {img_name} - Mean: {img[i].mean()}, Std: {img[i].std()}, Res: {res}")
    return img, torch.tensor(res, dtype = torch.float)

class aNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.ln1 = nn.Linear(64 * 64 * 3, 128)
    self.act_fn = nn.ReLU()
    self.ln2 = nn.Linear(128, 1)

  def forward(self, x):
    x = self.flatten(x)
    x = self.act_fn(self.ln1(x))
    x = self.ln2(x)
    return x

class bNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.ln1 = nn.Linear(64 * 64 * 3, 128)
    self.ln2 = nn.Linear(128, 128)
    self.ln3 = nn.Linear(128, 1)
    self.act_fn = nn.ReLU()

  def forward(self, x):
    x = self.flatten(x)
    x = self.act_fn(self.ln1(x))
    x = self.act_fn(self.ln2(x))
    x = self.ln3(x)
    return x

class cCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.conv = nn.Conv2d(3, 16, kernel_size = 7)
    self.ln = nn.Linear(16 * 58 * 58, 1)
    self.act_fn = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.act_fn(x)
    x = self.flatten(x)
    x = self.ln(x)
    return x

class dCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.conv = nn.Conv2d(3, 16, kernel_size = 7)
    self.ln = nn.Linear(16 * 29 * 29, 1)
    self.act_fn = nn.ReLU()
    self.pool = nn.MaxPool2d(2)

  def forward(self, x):
    x = self.conv(x)
    x = self.act_fn(x)
    x = self.pool(x)
    x = self.flatten(x)
    x = self.ln(x)
    return x

class eCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.conv1 = nn.Conv2d(3, 16, kernel_size = 7)
    self.conv2 = nn.Conv2d(16, 32, kernel_size = 5)
    self.ln = nn.Linear(32 * 12 * 12, 1)
    self.act_fn = nn.ReLU()
    self.pool = nn.MaxPool2d(2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.act_fn(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.act_fn(x)
    x = self.pool(x)
    x = self.flatten(x)
    x = self.ln(x)
    return x

class fCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.conv1 = nn.Conv2d(3, 16, kernel_size = 7)
    self.conv2 = nn.Conv2d(16, 32, kernel_size = 5)
    self.conv3 = nn.Conv2d(32, 32, kernel_size = 5)
    self.ln = nn.Linear(32 * 4 * 4, 1)
    self.act_fn = nn.ReLU()
    self.pool = nn.MaxPool2d(2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.act_fn(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.act_fn(x)
    x = self.pool(x)
    x = self.conv3(x)
    x = self.act_fn(x)
    x = self.pool(x)
    x = self.flatten(x)
    x = self.ln(x)
    return x

def get_r2(y, yhat):
  ybar = torch.mean(y)
  den = torch.sum((y - ybar) ** 2)
  num = torch.sum((y - yhat) ** 2)
  r2 = 1 - num / den
  return r2

def train_model(model, train_loader, test_loader, optimizer, lossf, epochs = 100):
  train_r2 = []
  test_r2 = []

  for epoch in range(epochs):
    model.train()
    train_ytrue = []
    train_ypred = []

    for x, y in train_loader:
      x, y = x.float(), y.float()
      optimizer.zero_grad()
      pred = model(x).squeeze()
      loss = lossf(pred, y)
      loss.backward()
      optimizer.step()

      train_ytrue.extend(y.detach().numpy())
      train_ypred.extend(pred.detach().numpy())

    train_r2.append(get_r2(torch.tensor(train_ytrue, dtype = torch.float), torch.tensor(train_ypred, dtype = torch.float)).item())

    model.eval()
    test_ytrue = []
    test_ypred = []

    with torch.no_grad():
      for x, y in test_loader:
        x, y = x.float(), y.float()
        pred = model(x).squeeze()
        test_ytrue.extend(y.numpy())
        test_ypred.extend(pred.numpy())

    test_r2.append(get_r2(torch.tensor(test_ytrue, dtype = torch.float), torch.tensor(test_ypred, dtype = torch.float)).item())

    if epoch == epochs - 1:
      final_train_ytrue = np.array(train_ytrue)
      final_test_ytrue = np.array(test_ytrue)

      train_residuals = final_train_ytrue - np.array(train_ypred)
      test_residuals = final_test_ytrue - np.array(test_ypred)

    print(f'Epoch [{epoch + 1} / {epochs}], Train R2 : {train_r2[-1]:.4f}, Test R2 : {test_r2[-1]:.4f}')

  return train_r2, test_r2, final_train_ytrue, train_residuals, final_test_ytrue, test_residuals

os.chdir(os.path.dirname(os.getcwd()))
train_data = LoadData('Data/cnntrain')
test_data = LoadData('Data/cnntest')

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)
lossf = nn.MSELoss()
total_train = np.zeros((6, 100))
total_test = np.zeros((6, 100))

## 1-(a)
model = aNN()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
total_train[0, :], total_test[0, :], _, _, _, _ = train_model(model, train_loader, test_loader, optimizer, lossf)

plt.plot(np.arange(1, 101), total_train[0, :])
plt.plot(np.arange(1, 101), total_test[0, :])
plt.xlabel('Epoch number')
plt.ylabel('R squared')
plt.legend(['Train', 'Test'])
plt.show()

## 1-(b)
model = bNN()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
total_train[1, :], total_test[1, :], _, _, _, _ = train_model(model, train_loader, test_loader, optimizer, lossf)

plt.plot(np.arange(1, 101), total_train[1, :])
plt.plot(np.arange(1, 101), total_test[1, :])
plt.xlabel('Epoch number')
plt.ylabel('R squared')
plt.legend(['Train', 'Test'])
plt.show()

## 1-(c)
model = cCNN()
optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
total_train[2, :], total_test[2, :], _, _, _, _ = train_model(model, train_loader, test_loader, optimizer, lossf)

plt.plot(np.arange(1, 101), total_train[2, :])
plt.plot(np.arange(1, 101), total_test[2, :])
plt.xlabel('Epoch number')
plt.ylabel('R squared')
plt.legend(['Train', 'Test'])
plt.show()

## 1-(d)
model = dCNN()
optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
total_train[3, :], total_test[3, :], _, _, _, _ = train_model(model, train_loader, test_loader, optimizer, lossf)

plt.plot(np.arange(1, 101), total_train[3, :])
plt.plot(np.arange(1, 101), total_test[3, :])
plt.xlabel('Epoch number')
plt.ylabel('R squared')
plt.legend(['Train', 'Test'])
plt.show()

## 1-(e)
model = eCNN()
optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
total_train[4, :], total_test[4, :], _, _, _, _ = train_model(model, train_loader, test_loader, optimizer, lossf)

plt.plot(np.arange(1, 101), total_train[4, :])
plt.plot(np.arange(1, 101), total_test[4, :])
plt.xlabel('Epoch number')
plt.ylabel('R squared')
plt.legend(['Train', 'Test'])
plt.show()

## 1-(f)
model = fCNN()
optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
total_train[5, :], total_test[5, :], final_train_ytrue, train_residuals, final_test_ytrue, test_residuals = train_model(model, train_loader, test_loader, optimizer, lossf)

plt.plot(np.arange(1, 101), total_train[5, :])
plt.plot(np.arange(1, 101), total_test[5, :])
plt.xlabel('Epoch number')
plt.ylabel('R squared')
plt.legend(['Train', 'Test'])
plt.show()

## 1-(g)
print(pd.DataFrame({'Models' : ['OneNN', 'TwoNN', 'OneCNN', 'OneCNNPool', 'TwoCNNPool', 'ThreeCNNPool'], 'Final train R2' : total_train[:, 99], 'Test train R2' : total_test[:, 99]}))

# 1-(h)
plt.scatter(final_train_ytrue, train_residuals)
plt.scatter(final_test_ytrue, test_residuals)
plt.xlabel('True Y')
plt.ylabel('Residuals')
plt.legend(['Train', 'Test'])
plt.show()