import torch
import numpy as np

class Trainer:
    """
    A simple trainer class for training pytorch models. 
    """
    @property 
    def device(self):
        return self._device 
    
    @device.setter 
    def device(self, d):
        self._device = d
        self.model = self.model.to(self.device)

    def __init__(self, model, criterion, optimizer, metrics=None, device="cpu",  wandb=False):
        """
        Write documentation. 

        Note : Introduce defaults for criterion and optimizer. 
        """

        super().__init__()
        self.model = model
        self.model.eval()
        self._device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.wandb = wandb
        self.model = self.model.to(self.device)
        self.metrics = metrics
    
    def train_step(self, batch):
        """
        Implements the logic for a single training iteration and 
        returns a dictionary consisting of the batch loss and any
        additional metrics if required. 
        """
        res = {}
        self.model.train()
        inputs, labels = batch 
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs.squeeze(), labels)
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        loss = loss.detach().numpy().item()
        res = {'loss':loss}

        if self.metrics:
            for metric in self.metrics:
                metric_name = metric.__name__
                metric_val = metric(outputs, labels) 
                res[metric_name] = metric_val

        return res 
    def eval_step(self, batch):
        """
        Implements the logic for a single validation iteration and 
        returns a dictionary consisting of the batch loss and any
        additional metrics if required. 
        """
        res = {}
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs.squeeze(), labels)
        loss = loss.detach().numpy().item()
        res = {'val_loss':loss}

        if self.metrics:
            for metric in self.metrics:
                metric_name = "val_"+metric.__name__
                metric_val = metric(outputs, labels) 
                res[metric_name] = metric_val

        return res 
    
    def fit(self, train_loader, val_loader=None, nepochs=50, verbose=True):
        """
        Accept data loaders for training and validation and train the model. 
        """
        history = {}

        for epoch in range(1, 1+nepochs):
            train_res = {}
            
            # loop through training batches
            for batch in train_loader:
                res = self.train_step(batch)
                for k, v in res.items():
                    if k not in train_res:
                        train_res[k] = []
                    train_res[k].append(v)
            for k, v in train_res.items():
                if k not in history:
                    history[k] = []
                history[k].append(np.mean(v))
            
            # loop through test batches 
            if val_loader:
                val_res = {}
                for batch in val_loader:
                    res = self.eval_step(batch)
                    for k, v in res.items():
                        if k not in val_res:
                            val_res[k] = []
                        val_res[k].append(v)
                for k, v in val_res.items():
                    if k not in history:
                        history[k] = []
                    history[k].append(np.mean(v))
            
            # print to std. output 
            if verbose:
                out_str = f"Epoch: {epoch:3d}/{nepochs}, "
                for k, v in history.items():
                    out_str += f"{k}: {v[-1]:.4f}, "
                    self.wandb.log({f"{k}": v[-1]})
                out_str = out_str[:-2]
                print(out_str)
        
        return history

    # def _train_epoch(self):
    #     self.model.train()
    #     running_loss = 0.0
    #     for inputs, labels in self.loaders['train']:
    #         inputs, labels = inputs.to(self.device), labels.to(self.device)
    #         self.optimizer.zero_grad()
    #         outputs = self.model(inputs)
    #         loss = self.criterion(outputs.squeeze(), labels)
    #         loss.backward()
    #         self.optimizer.step()
    #         running_loss += loss.item() * inputs.size(0)
    #     return running_loss / len(self.loaders['train'].dataset)

    # def _evaluate_epoch(self):
    #     self.model.eval()
    #     running_loss = 0.0
    #     with torch.no_grad():
    #         for inputs, labels in self.loaders['valid']:
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             outputs = self.model(inputs)
    #             loss = self.criterion(outputs.squeeze(), labels)
    #             running_loss += loss.item() * inputs.size(0)
    #     return running_loss / len(self.loaders['valid'].dataset)


    # def train(self):
    #     """
    #     """
    #     for epoch in range(self.config['epochs']):
    #         best_valid_loss = float('inf')
    #         train_loss = self._train_epoch()
    #         valid_loss = self._evaluate_epoch()
    #         train_losses.append(train_loss)
    #         valid_losses.append(valid_loss)
    #         self.wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})
    #         print(f"Epoch [{epoch+1}/{self.config['epochs']}] - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    #         # Early stopping
    #         if valid_loss < best_valid_loss:
    #             best_valid_loss = valid_loss
    #             current_patience = 0
    #             # Save the best model if you want
    #             torch.save(self.model.state_dict(), "best_model.pt")
    #         else:
    #             current_patience += 1
    #             if current_patience >= self.config['patience']:
    #                 print("Early stopping! Validation loss hasn't improved in the last", self.config['patience'], "epochs.")
    #                 break
