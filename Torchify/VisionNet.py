import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def evaluate_model(model, val_loader, device):
    model.eval()
    outputs = [model.validation_step(batch, device) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit_one_cycle(epochs, model, train_loader, val_loader=None, grad_clip=None, optimizer=None, sched=None, device=torch.device('cpu')):
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
        
        result = {}
        if val_loader:
            result = evaluate_model(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_accs).mean().item()
        result['lr'] = lrs
        model.epoch_end(epoch, result, val_loader is not None)
        history.append(result)
            
    return history

class ImageClassificationModel(nn.Module):

    def training_step(self, batch, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return loss, acc

    def validation_step(self, batch, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result, has_val_loader):
        lr = result.get('lr', [0.0])
        train_loss = result.get('train_loss', 0.0)
        train_acc = result.get('train_acc', 0.0) 

        if isinstance(lr, list):
            lr = lr[-1]

        if has_val_loader:
            val_loss = result.get('val_loss', 0.0)
            val_acc = result.get('val_acc', 0.0)
            print(f"Epoch [{epoch+1}], "
                  f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        else:
            print(f"Epoch [{epoch+1}], "
                  f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")

    def compile(self, loss_fn, optimizer: torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler=None, grad_clip:float=None):
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, epochs, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset = None, batch_size: int = 32):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on {device}")


        self.to(device)
        self.history = fit_one_cycle(epochs=epochs, model=self, train_loader=train_loader, val_loader=val_loader,
                                    grad_clip=self.grad_clip, optimizer=self.optimizer, 
                                    sched=self.scheduler, device=device)
        
        return self.history
        
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
                images, _ = batch  
                images = images.to(device)
                outputs = self(images)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
        return predictions

    def _predict_from_dataset(self, dataset, device):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        return self._predict_from_loader(data_loader, device)

    def _predict_from_tensor(self, tensor, device):
        tensor = tensor.to(device)
        with torch.no_grad():
            outputs = self(tensor)
            _, preds = torch.max(outputs, dim=1)
        return preds.item()
    
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

        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro')
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')

        return acc, f1, precision, recall

    def plot_accuracies(self):
        accuracies = [x['val_acc'] for x in self.history if 'val_acc' in x]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.show()

    def plot_losses(self):
        train_losses = [x.get('train_loss') for x in self.history]
        val_losses = [x['val_loss'] for x in self.history if 'val_loss' in x]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'valid'])
        plt.title('Loss vs. Epoch')
        plt.show()