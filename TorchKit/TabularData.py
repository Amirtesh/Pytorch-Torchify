import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, mean_squared_error, mean_absolute_error, r2_score

def get_lr(optimizer):
    """Retrieve the learning rate from the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def evaluate_model(model, val_loader, device, task):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation data.
        device (torch.device): The device to use for computation.
        task (str): The type of task ('classification' or 'regression').

    Returns:
        dict: A dictionary with validation metrics.
    """
    model.eval()
    outputs = [model.validation_step(batch, device, task) for batch in val_loader]
    return model.validation_epoch_end(outputs, task)

def fit_one_cycle(epochs, model, train_loader, val_loader, task, grad_clip=None, optimizer=None, sched=None, device=torch.device('cpu')):
    """
    Train the model for a specified number of epochs using one-cycle learning rate policy.

    Args:
        epochs (int): Number of epochs to train.
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        task (str): The type of task ('classification' or 'regression').
        grad_clip (float, optional): Gradient clipping value. Defaults to None.
        optimizer (Optimizer, optional): Optimizer for training. Defaults to None.
        sched (Scheduler, optional): Learning rate scheduler. Defaults to None.
        device (torch.device, optional): The device to use for computation. Defaults to CPU.

    Returns:
        list: A list of dictionaries with training and validation metrics for each epoch.
    """
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch, device, task)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            if sched:
                sched.step()
            
        result = evaluate_model(model, val_loader, device, task)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lr'] = lrs
        model.epoch_end(epoch, result, task)
        history.append(result)
            
    return history

class TabularModel(nn.Module):
    def training_step(self, batch, device, task):
        """
        Perform a training step.

        Args:
            batch (tuple): A batch of data.
            device (torch.device): The device to use for computation.
            task (str): The type of task ('classification' or 'regression').

        Returns:
            torch.Tensor: The training loss.
        """
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if task == 'classification':
            out = self(inputs)
            loss = F.cross_entropy(out, labels)
        elif task == 'regression':
            out = self(inputs)
            out = out.view(-1)
            labels = labels.float().view_as(out)
            loss = F.mse_loss(out, labels.float())
        else:
            raise ValueError("Task must be 'classification' or 'regression'")
        return loss

    def validation_step(self, batch, device, task):
        """
        Perform a validation step.

        Args:
            batch (tuple): A batch of data.
            device (torch.device): The device to use for computation.
            task (str): The type of task ('classification' or 'regression').

        Returns:
            dict: A dictionary with validation loss and predictions.
        """
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if task == 'classification':
            out = self(inputs)
            loss = F.cross_entropy(out, labels)
        elif task == 'regression':
            out = self(inputs).squeeze()
            loss = F.mse_loss(out, labels.float())
        else:
            raise ValueError("Task must be 'classification' or 'regression'")
        return {'val_loss': loss.detach(), 'outputs': out.detach(), 'labels': labels.detach()}

    def validation_epoch_end(self, outputs, task):
        """
        Aggregate validation results at the end of an epoch.

        Args:
            outputs (list): A list of validation step outputs.
            task (str): The type of task ('classification' or 'regression').

        Returns:
            dict: A dictionary with aggregated validation metrics.
        """
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        if task == 'classification':
            all_outputs = torch.cat([x['outputs'] for x in outputs], dim=0)
            all_labels = torch.cat([x['labels'] for x in outputs], dim=0)
            _, preds = torch.max(all_outputs, dim=1)
            acc = accuracy_score(all_labels.cpu().numpy(), preds.cpu().numpy())
            return {'val_loss': epoch_loss.item(), 'val_acc': acc}
        elif task == 'regression':
            all_outputs = torch.cat([x['outputs'] for x in outputs], dim=0).squeeze().cpu().detach().numpy()
            all_labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu().detach().numpy()
            mse = mean_squared_error(all_labels, all_outputs)
            return {'val_loss': epoch_loss.item(), 'val_mse': mse}
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def epoch_end(self, epoch, result, task):
        """
        Print epoch results.

        Args:
            epoch (int): The current epoch number.
            result (dict): A dictionary with training and validation metrics.
            task (str): The type of task ('classification' or 'regression').
        """
        lr = result.get('lr', [0.0])
        train_loss = result.get('train_loss', 0.0)
        val_loss = result.get('val_loss', 0.0)
        if task == 'classification':
            val_acc = result.get('val_acc', 0.0)
            print(f"Epoch [{epoch+1}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        elif task == 'regression':
            val_mse = result.get('val_mse', 0.0)
            print(f"Epoch [{epoch+1}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_mse: {val_mse:.4f}")
        else:
            raise ValueError("Task must be 'classification' or 'regression'")
        
    def compile(self, loss_fn, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler = None, grad_clip: float = None, task: str = 'classification'):
        """
        Compile the model with the given loss function, optimizer, scheduler, and task type.

        Args:
            loss_fn: Loss function to use.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
            grad_clip (float, optional): Gradient clipping value. Defaults to None.
            task (str, optional): The type of task ('classification' or 'regression'). Defaults to 'classification'.
        """
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task = task

    def fit(self, epochs, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        """
        Train the model for a specified number of epochs.

        Args:
            epochs (int): Number of epochs to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.

        Returns:
            list: A list of dictionaries with training and validation metrics for each epoch.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on {device}")

        self.to(device)
        self.history = fit_one_cycle(epochs=epochs, model=self, train_loader=train_loader, val_loader=val_loader,
                                      task=self.task, grad_clip=self.grad_clip, optimizer=self.optimizer, 
                                      sched=self.scheduler, device=device)
         
        return self.history
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader):
        """
        Evaluate the model on a given data set.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the data to evaluate.

        Returns:
            dict: A dictionary with evaluation metrics.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return evaluate_model(self, data_loader, device, self.task)
    
    def predict(self, data):
        """
        Make predictions on the given data.

        Args:
            data (DataLoader, Dataset, or Tensor): The data to make predictions on.

        Returns:
            np.ndarray: The predicted labels or values.
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
        Make predictions from a DataLoader.

        Args:
            data_loader (torch.utils.data.DataLoader): The DataLoader to make predictions on.
            device (torch.device): The device to use for computation.

        Returns:
            np.ndarray: The predicted labels or values.
        """
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                inputs, _ = batch  
                inputs = inputs.to(device)
                outputs = self(inputs)
                if self.task == 'classification':
                    _, preds = torch.max(outputs, dim=1)
                elif self.task == 'regression':
                    preds = outputs.squeeze()
                else:
                    raise ValueError("Task must be 'classification' or 'regression'")
                preds = np.atleast_1d(preds.cpu().numpy())
                predictions.extend(preds)
        return predictions

    def _predict_from_dataset(self, dataset, device):
        """
        Make predictions from a Dataset.

        Args:
            dataset (torch.utils.data.Dataset): The Dataset to make predictions on.
            device (torch.device): The device to use for computation.

        Returns:
            np.ndarray: The predicted labels or values.
        """
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        return self._predict_from_loader(data_loader, device)

    def _predict_from_tensor(self, tensor, device):
        """
        Make predictions from a single tensor.

        Args:
            tensor (torch.Tensor): The tensor to make predictions on.
            device (torch.device): The device to use for computation.

        Returns:
            int or float: The predicted label or value.
        """
        tensor = tensor.to(device)
        with torch.no_grad():
            outputs = self(tensor)
            if self.task == 'classification':
                _, preds = torch.max(outputs, dim=1)
                return preds.item()
            elif self.task == 'regression':
                return outputs.squeeze().cpu().numpy()
            else:
                raise ValueError("Task must be 'classification' or 'regression'")
    
    def metrics(self, dataset):
        """
        Calculate metrics for the given dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to calculate metrics on.

        Returns:
            tuple: Metrics for the dataset. For classification returns accuracy,f1 and precision.
                                            For regression returns mse,rmse,mae and r2_score
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

        if self.task == 'classification':
            acc = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='macro')
            precision = precision_score(true_labels, predictions, average='macro')
            return acc, f1, precision
        elif self.task == 'regression':
            mse = mean_squared_error(true_labels, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_labels, predictions)
            r2 = r2_score(true_labels, predictions)
            return mse, rmse, mae, r2
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def plot_accuracies(self):
        """
        Plot the validation accuracies over epochs.

        Note:
            This method is only applicable for classification tasks.
        """
        if self.task == 'classification':
            accuracies = [x['val_acc'] for x in self.history if 'val_acc' in x]
            plt.plot(accuracies, '-x')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.title('Accuracy vs. Epoch')
            plt.show()
        else:
            print("Accuracy plot is only available for classification tasks.")

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
        plt.show()
        
    def plot_r2(self,ds):
        """
        Plot the R-squared values over epochs.
        Args:
            Takes a dataset as input

        Note:
            This method is only applicable for regression tasks.
        """
        if self.task == 'regression':
            r2_scores = [self.metrics(ds)[-1] for _ in self.history if 'val_mse' in _]
            plt.plot(r2_scores, '-gx')
            plt.xlabel('epoch')
            plt.ylabel('R-squared')
            plt.title('R-squared vs. Epoch')
            plt.show()
        else:
            print("R-squared plot is only available for regression tasks.")
