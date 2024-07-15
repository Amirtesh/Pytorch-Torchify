import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score

def accuracy(outputs, labels):
    """
    Calculate the accuracy given the model outputs and true labels.

    Args:
        outputs (torch.Tensor): The outputs from the model.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        torch.Tensor: The accuracy as a tensor.
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_lr(optimizer):
    """
    Get the learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer used in training.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def evaluate_model(model, val_loader, device):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the evaluation on (CPU or GPU).

    Returns:
        dict: Dictionary containing validation loss and accuracy.
    """
    model.eval()
    outputs = [model.validation_step(batch, device) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit_one_cycle(epochs, model, train_loader, val_loader, grad_clip=None, optimizer=None, sched=None, device=torch.device('cpu')):
    """
    Train the model for a specified number of epochs.

    Args:
        epochs (int): Number of epochs to train for.
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        grad_clip (float, optional): Gradient clipping value. Default is None.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. Default is None.
        sched (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.
        device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
        list: List of dictionaries containing training and validation metrics for each epoch.
    """
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_accs = []  
        lrs = []
        for batch in train_loader:
            loss, acc = model.training_step(batch, device) 
            train_losses.append(loss)
            train_accs.append(acc) 
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            if sched:
                sched.step()
            
        result = evaluate_model(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_accs).mean().item()
        result['lr'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
            
    return history

class ImageModel(nn.Module):
    """
    A generic image classification model class that includes methods for training, validation, 
    evaluation, and prediction.

    Methods:
        training_step(batch, device): Performs a single training step and returns the loss and accuracy.
        validation_step(batch, device): Performs a single validation step and returns the loss and accuracy.
        validation_epoch_end(outputs): Aggregates validation results at the end of an epoch.
        epoch_end(epoch, result): Prints the metrics at the end of an epoch.
        compile(loss_fn, optimizer, scheduler, grad_clip): Compiles the model with loss function, optimizer, scheduler, and gradient clipping value.
        fit(epochs, train_loader, val_loader): Trains the model for a specified number of epochs.
        evaluate(data_loader): Evaluates the model on a given dataset.
        predict(data): Predicts the classes for the given data.
        _predict_from_loader(data_loader, device): Predicts the classes for the data from a DataLoader.
        _predict_from_dataset(dataset, device): Predicts the classes for the data from a Dataset.
        _predict_from_tensor(tensor, device): Predicts the class for a single image tensor.
        metrics(dataset): Calculates accuracy, F1-score, and precision for the given dataset.
        plot_accuracies(): Plots the validation accuracies over epochs.
        plot_losses(): Plots the training and validation losses over epochs.
    """

    def training_step(self, batch, device):
        """
        Perform a single training step.

        Args:
            batch (tuple): A batch of data and labels.
            device (torch.device): Device to run the training step on.

        Returns:
            tuple: Loss and accuracy for the batch.
        """
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return loss, acc

    def validation_step(self, batch, device):
        """
        Perform a single validation step.

        Args:
            batch (tuple): A batch of data and labels.
            device (torch.device): Device to run the validation step on.

        Returns:
            dict: Dictionary containing validation loss and accuracy for the batch.
        """
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        """
        Aggregate validation results at the end of an epoch.

        Args:
            outputs (list): List of dictionaries containing validation loss and accuracy for each batch.

        Returns:
            dict: Dictionary containing average validation loss and accuracy for the epoch.
        """
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        """
        Print metrics at the end of an epoch.

        Args:
            epoch (int): Current epoch number.
            result (dict): Dictionary containing training and validation metrics.
        """
        lr = result.get('lr', [0.0])
        train_loss = result.get('train_loss', 0.0)
        val_loss = result.get('val_loss', 0.0)
        val_acc = result.get('val_acc', 0.0)
        train_acc = result.get('train_acc', 0.0) 

        if isinstance(lr, list):
            lr = lr[-1]

        print(f"Epoch [{epoch+1}], "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        
    def compile(self, loss_fn, optimizer: torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler=None, grad_clip:float=None):
        """
        Compile the model with loss function, optimizer, scheduler, and gradient clipping value.

        Args:
            loss_fn (callable): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.
            grad_clip (float, optional): Gradient clipping value. Default is None.
        """
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, epochs, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader):
        """
        Train the model for a specified number of epochs.

        Args:
            epochs (int): Number of epochs to train for.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.

        Returns:
            list: List of dictionaries containing training and validation metrics for each epoch.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on {device}")

        self.to(device)
        self.history = fit_one_cycle(epochs=epochs, model=self, train_loader=train_loader, val_loader=val_loader,
                                      grad_clip=self.grad_clip, optimizer=self.optimizer, 
                                      sched=self.scheduler, device=device)
         
        return self.history
    
    def evaluate(self, data_loader:torch.utils.data.DataLoader):
        """
        Evaluate the model on a given dataset.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.

        Returns:
            dict: Dictionary containing validation loss and accuracy.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return evaluate_model(self, data_loader, device)
    
    def predict(self, data):
        """
        Predict the classes for the given data.

        Args:
            data (torch.utils.data.DataLoader, torch.utils.data.Dataset, torch.Tensor): Data to predict.

        Returns:
            numpy.ndarray, int: Predicted classes.
        """
        self.eval()
        device = next(self.parameters()).device  
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_from_loader(data, device)
        elif isinstance(data, torch.utils.data.Dataset):
            return self._predict_from_dataset(data, device)
        elif isinstance(data, torch.Tensor):
            return self._predict_from_tensor(data.unsqueeze(0), device)
        else:
            raise TypeError("Input must be a DataLoader, Dataset, or Tensor")

    def _predict_from_loader(self, data_loader, device):
        """
        Predict the classes for the data from a DataLoader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the data.
            device (torch.device): Device to run the prediction on (CPU or GPU).

        Returns:
            numpy.ndarray: Predicted classes.
        """
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                images, _ = batch  
                images = images.to(device)
                outputs = self(images)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
        return predictions

    def _predict_from_dataset(self, dataset, device):
        """
        Predict the classes for the data from a Dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset for the data.
            device (torch.device): Device to run the prediction on (CPU or GPU).

        Returns:
            numpy.ndarray: Predicted classes.
        """
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        return self._predict_from_loader(data_loader, device)

    def _predict_from_tensor(self, tensor, device):
        """
        Predict the class for a single image tensor.

        Args:
            tensor (torch.Tensor): Image tensor.
            device (torch.device): Device to run the prediction on (CPU or GPU).

        Returns:
            int: Predicted class.
        """
        tensor = tensor.to(device)
        with torch.no_grad():
            outputs = self(tensor)
            _, preds = torch.max(outputs, dim=1)
        return preds.item()
    
    def metrics(self, dataset):
        """
        Calculate accuracy, F1-score, and precision for the given dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset for which to calculate the metrics.

        Returns:
            tuple: Accuracy, F1-score, and precision.
        """
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        predictions = self.predict(dataset)
        true_labels = []
        for i in range(len(dataset)):
            _, labels = dataset[i]
            if isinstance(labels, torch.Tensor):
                true_labels.append(labels.cpu().numpy())
            else:
                true_labels.append(labels)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro')
        precision = precision_score(true_labels, predictions, average='macro')

        return acc, f1, precision

    def plot_accuracies(self):
        """
        Plot the validation accuracies over epochs.
        """
        accuracies = [x['val_acc'] for x in self.history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.show()

    def plot_losses(self):
        """
        Plot the training and validation losses over epochs.
        """
        train_losses = [x.get('train_loss') for x in self.history]
        val_losses = [x['val_loss'] for x in self.history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'valid'])
        plt.title('Loss vs. Epoch')
        plt.show()

