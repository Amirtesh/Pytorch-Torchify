# TorchNex

TorchNex is a custom library for PyTorch neural networks, designed to simplify working with image data and tabular data (supervised learning). It provides Keras-like `compile` and `fit` methods, as well as features like learning rate scheduling, gradient clipping, and plotting of losses and accuracy. Additionally, it includes methods for evaluating metrics like accuracy, F1 score, precision, mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), and R² score.  

(Works with python 3.10 and above)

## Features

- Learning rate scheduler
- Gradient clipping
- Plotting of losses and accuracy (for classification)
- Plotting of R² score (for regression)
- Metrics method to easily evaluate model performance

## Installation

Use command ```pip install torchnex``` to install the library.

## Usage

1. **Inherit from TorchKit classes:**
   - When working with image data, inherit from `TorchNex.ImageClassifier.ImageModel`.
   - When working with tabular data, inherit from `TorchNex.TabularData.TabularModel`.

2. **Create an instance of your model:**

   ```python
   model = YourCustomModel()
   ```

3. **Compile the model:**

   ```python
   model.compile(
       loss_function=nn.CrossEntropyLoss(),
       optimizer=optim.Adam(model.parameters(), lr=0.001),
       learning_rate_scheduler=optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=10, steps_per_epoch=len(train_loader)),
       gradient_clip=1.0
   )
   ```

   For `TabularModel`, you can specify the task (default is 'classification'):

   ```python
   model.compile(
       loss_function=nn.MSELoss(),
       optimizer=optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4),
       learning_rate_scheduler=optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=10, steps_per_epoch=len(train_loader)),
       gradient_clip=1.0,
       task='regression'
   )
   ```

4. **Fit the model:**

   ```python
   history = model.fit(
       epochs=10,
       train_loader=train_loader,
       val_loader=val_loader
   )
   ```

   To visualize the loss, accuracy, and R² score plots, make sure to create a variable for the fit method:

   ```python
   history = model.fit(...)
   ```

5. **Make predictions:**

   ```python
   predictions = model.predict(data_loader)
   ```

6. **Evaluate metrics:**

   For `ImageModel` and `TabularModel` (classification):

   ```python
   accuracy, f1_score, precision = model.metrics(dataset)
   ```

   For `TabularModel` (regression):

   ```python
   mse, rmse, mae, r2_score = model.metrics(dataset)
   ```

   7.**Using fit_one_cycle method from either ImageClassifier or TabularData to implement custom steps in model:**

      With ImageClassifier-
      ```python
      import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.optim import Adam
   from torch.optim.lr_scheduler import OneCycleLR
   from torch.utils.data import DataLoader
   from torchvision.datasets import CIFAR10
   from torchvision.transforms import ToTensor
   from TorchKit.ImageClassifier import fit_one_cycle
   
   class CustomImageModel(nn.Module):
       def __init__(self):
           super(CustomImageModel, self).__init__()
           self.conv1 = nn.Conv2d(3, 16, 3, 1)
           self.conv2 = nn.Conv2d(16, 32, 3, 1)
           self.fc1 = nn.Linear(32*6*6, 128)
           self.fc2 = nn.Linear(128, 10)
   
       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           x = self.conv2(x)
           x = F.relu(x)
           x = F.max_pool2d(x, 2)
           x = x.view(x.size(0), -1)
           x = self.fc1(x)
           x = F.relu(x)
           x = self.fc2(x)
           return x
   
       def training_step(self, batch, device):
           images, labels = batch
           images, labels = images.to(device), labels.to(device)
           outputs = self(images)
           loss = F.cross_entropy(outputs, labels)
           return loss
   
       def validation_step(self, batch, device):
           images, labels = batch
           images, labels = images.to(device), labels.to(device)
           outputs = self(images)
           loss = F.cross_entropy(outputs, labels)
           acc = (outputs.argmax(dim=1) == labels).float().mean()
           return {'val_loss': loss.detach(), 'val_acc': acc}
   
       def validation_epoch_end(self, outputs):
           batch_losses = [x['val_loss'] for x in outputs]
           epoch_loss = torch.stack(batch_losses).mean()
           batch_accs = [x['val_acc'] for x in outputs]
           epoch_acc = torch.stack(batch_accs).mean()
           return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
   
       def epoch_end(self, epoch, result):
           print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
   
   train_dataset = CIFAR10(root='data/', train=True, transform=ToTensor(), download=True)
   val_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
      
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32)
      
   model = CustomImageModel()
      
   optimizer = Adam(model.parameters(), lr=0.001)
   scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=10)
      
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model.to(device)
      
   history = fit_one_cycle(
       epochs=10,
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       optimizer=optimizer,
       sched=scheduler,
       device=device
   )
   ```
      With TabularData-
      ```python
      import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.optim import Adam
   from torch.optim.lr_scheduler import OneCycleLR
   from torch.utils.data import DataLoader, TensorDataset
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split
   from TorchKit.TabularData import fit_one_cycle
   
   class CustomTabularModel(nn.Module):
       def __init__(self):
           super(CustomTabularModel, self).__init__()
           self.fc1 = nn.Linear(10, 64)
           self.fc2 = nn.Linear(64, 32)
           self.fc3 = nn.Linear(32, 1)
   
       def forward(self, x):
           x = self.fc1(x)
           x = F.relu(x)
           x = self.fc2(x)
           x = F.relu(x)
           x = self.fc3(x)
           return x
   
       def training_step(self, batch, device):
           features, targets = batch
           features, targets = features.to(device), targets.to(device)
           outputs = self(features)
           loss = F.mse_loss(outputs, targets)
           return loss
   
       def validation_step(self, batch, device):
           features, targets = batch
           features, targets = features.to(device), targets.to(device)
           outputs = self(features)
           loss = F.mse_loss(outputs, targets)
           return {'val_loss': loss.detach()}
   
       def validation_epoch_end(self, outputs):
           batch_losses = [x['val_loss'] for x in outputs]
           epoch_loss = torch.stack(batch_losses).mean()
           return {'val_loss': epoch_loss.item()}
   
       def epoch_end(self, epoch, result):
           print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}")
   
   X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
      
   train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
   val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
      
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32)
      
   model = CustomTabularModel()
      
   optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
   scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=10)
      
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model.to(device)
      
   history = fit_one_cycle(
       epochs=10,
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       optimizer=optimizer,
       sched=scheduler,
       device=device
   )
   ```
   Note that when using this method you will not be able to use the plotting, predict and metrics function since you will be inheriting from nn.Module.

      

## Example

Here's a complete example of using TorchKit for image classification:

```python
import torch
import torch.nn as nn
from TorchKit.ImageClassifier import ImageModel

class YourCustomModel(ImageModel):
    def __init__(self):
        super(YourCustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32*6*6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

model = YourCustomModel()
model.compile(
    loss_function=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    learning_rate_scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=10, steps_per_epoch=len(train_loader), max_lr=0.001),
    gradient_clip=1.0
)
history = model.fit(
    epochs=10,
    train_loader=train_loader,
    val_loader=val_loader
)

model.plot_accuracies()
model.plot_losses()
```

## Dependencies

- PyTorch
- Matplotlib (for plotting)

