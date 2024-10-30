import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def evaluate_model(model, val_loader, device):
    model.eval()
    outputs = [model.validation_step(batch, device) for batch in val_loader]
    return model.validation_epoch_end(outputs)
    

def train(epochs, model, train_loader, val_loader=None, grad_clip=None, optimizer=None, sched=None, device=torch.device('cpu')):
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_mses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch, device)
            train_losses.append(loss)
            if model.task == 'regression':
                mse = model.calculate_mse(batch, device)
                train_mses.append(mse)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            if sched:
                sched.step()
        result = {'train_loss': torch.stack(train_losses).mean().item(), 'lr': lrs}
        if model.task == 'regression':
            result['train_mse'] = torch.stack(train_mses).mean().item()
        if val_loader is not None:
            val_result = evaluate_model(model, val_loader, device)
            result.update(val_result)
        model.epoch_end(epoch, result, val_loader is not None)
        history.append(result)
    return history



class TabularModel(nn.Module):
    def training_step(self, batch, device):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if self.task == 'classification':
            out = self(inputs)
            loss = F.cross_entropy(out, labels)
        elif self.task == 'regression':
            out = self(inputs)
            out = out.view(-1)
            labels = labels.float().view_as(out)
            loss = F.mse_loss(out, labels.float())
        else:
            raise ValueError("Task must be 'classification' or 'regression'")
        return loss

    def calculate_mse(self, batch, device):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        out = self(inputs)
        out = out.view(-1)
        labels = labels.float().view_as(out)
        mse = F.mse_loss(out, labels.float(), reduction='none')
        return mse.mean()

    def validation_step(self, batch, device):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if self.task == 'classification':
            out = self(inputs)
            loss = F.cross_entropy(out, labels)
        elif self.task == 'regression':
            out = self(inputs).squeeze()
            loss = F.mse_loss(out, labels.float())
        else:
            raise ValueError("Task must be 'classification' or 'regression'")
        return {'val_loss': loss.detach(), 'outputs': out.detach(), 'labels': labels.detach()}

    def validation_epoch_end(self, outputs):
        if not outputs:
            return {}
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        if self.task == 'classification':
            all_outputs = torch.cat([x['outputs'] for x in outputs], dim=0)
            all_labels = torch.cat([x['labels'] for x in outputs], dim=0)
            _, preds = torch.max(all_outputs, dim=1)
            acc = accuracy_score(all_labels.cpu().numpy(), preds.cpu().numpy())
            return {'val_loss': epoch_loss.item(), 'val_acc': acc}
        elif self.task == 'regression':
            all_outputs = torch.cat([x['outputs'] for x in outputs], dim=0).squeeze().cpu().detach().numpy()
            all_labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu().detach().numpy()
            mse = mean_squared_error(all_labels, all_outputs)
            return {'val_loss': epoch_loss.item(), 'val_mse': mse}
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def epoch_end(self, epoch, result, has_val_loader):
        lr = result.get('lr', [0.0])
        train_loss = result.get('train_loss', 0.0)
        if has_val_loader:
            val_loss = result.get('val_loss', 0.0)
            if self.task == 'classification':
                val_acc = result.get('val_acc', 0.0)
                print(f"Epoch [{epoch+1}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
            elif self.task == 'regression':
                train_mse = result.get('train_mse', 0.0)
                val_mse = result.get('val_mse', 0.0)
                print(f"Epoch [{epoch+1}], train_loss: {train_loss:.4f}, train_mse: {train_mse:.4f}, val_loss: {val_loss:.4f}, val_mse: {val_mse:.4f}")
        else:
            if self.task == 'classification':
                print(f"Epoch [{epoch+1}], train_loss: {train_loss:.4f}")
            elif self.task == 'regression':
                train_mse = result.get('train_mse', 0.0)
                print(f"Epoch [{epoch+1}], train_loss: {train_loss:.4f}, train_mse: {train_mse:.4f}")
            else:
                raise ValueError("Task must be 'classification' or 'regression'")

    def compile(self, loss_fn, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler = None, grad_clip: float = None, task: str = 'classification'):
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task = task

    def fit(self, epochs, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset = None, batch_size: int = 32):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on {device}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
        self.to(device)
        self.history = train(epochs=epochs, model=self, train_loader=train_loader, val_loader=val_loader,
                                      grad_clip=self.grad_clip, optimizer=self.optimizer, 
                                      sched=self.scheduler, device=device)
        return self.history
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return evaluate_model(self, data_loader, device)
    
    def predict(self, data):
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
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        return self._predict_from_loader(data_loader, device)

    def _predict_from_tensor(self, tensor, device):
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

    def performance(self, dataset):
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
            recall = recall_score(true_labels, predictions, average='macro')
            return acc, f1, precision, recall
        elif self.task == 'regression':
            mse = mean_squared_error(true_labels, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_labels, predictions)
            r2 = r2_score(true_labels, predictions)
            return mse, rmse, mae, r2
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def plot_accuracies(self):
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
        train_losses = [x.get('train_loss') for x in self.history]
        val_losses = [x['val_loss'] for x in self.history if 'val_loss' in x]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'valid'])
        plt.show()
        
    def plot_r2(self, ds):
        if self.task == 'regression':
            r2_scores = [self.performance(ds)[-1] for _ in self.history if 'val_mse' in _]
            plt.plot(r2_scores, '-gx')
            plt.xlabel('epoch')
            plt.ylabel('R-squared')
            plt.title('R-squared vs. Epoch')
            plt.show()
        else:
            print("R-squared plot is only available for regression tasks.")
