import torch

class Trainer:
    """
    Trainer class for training/validation etc
    """

    def __init__(self, config, device, model, criterion, optimizer, loaders, wandb):
        super().__init__()
        self.config = config
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loaders = loaders
        self.wandb = wandb

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.loaders['train']:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        return running_loss / len(self.loaders['train'].dataset)

    def _evaluate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.loaders['valid']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                running_loss += loss.item() * inputs.size(0)
        return running_loss / len(self.loaders['valid'].dataset)


    def train(self):
        # Training loop with validation
        train_losses = []
        valid_losses = []
        for epoch in range(self.config['epochs']):
            best_valid_loss = float('inf')
            train_loss = self._train_epoch()
            valid_loss = self._evaluate_epoch()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            self.wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})
            print(f"Epoch [{epoch+1}/{self.config['epochs']}] - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

            # Early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                current_patience = 0
                # Save the best model if you want
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                current_patience += 1
                if current_patience >= self.config['patience']:
                    print("Early stopping! Validation loss hasn't improved in the last", self.config['patience'], "epochs.")
                    break
