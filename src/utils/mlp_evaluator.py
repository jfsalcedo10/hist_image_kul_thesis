from utils.mlp_classifier import MLP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class MLPEvaluator():
    def __init__(self, device, batch_size, num_epochs, size_z, learning_rate, betas, num_gpu = 0):
        # self.model = None
        self.num_gpu = num_gpu
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.size_z = size_z
        self.learning_rate = learning_rate
        self.betas = betas
    
    def calculate_accuracy(self, predictions, labels):
        # Apply a threshold of 0.5 to convert the sigmoid output to binary predictions (0 or 1)
        binary_predictions = (predictions >= 0.5).float()
        # Compare predictions with the ground truth labels
        correct_predictions = (binary_predictions == labels).float()
        # Calculate the accuracy
        accuracy = correct_predictions.mean()

        return accuracy.item()

    def get_mlp_accuracy(self, model, X_test, y_test):
        model.eval()
        y_pred = model(X_test.to(self.device))

        accuracy = self.calculate_accuracy(y_pred.cpu(), y_test)
        print('MLP Accuracy:', accuracy)

        return accuracy
    
    def train_mlp(self, X,y):
        mlp_loader = DataLoader(list(zip(X,y)), shuffle = True, batch_size= self.batch_size) # type: ignore

        mlp = MLP(self.size_z).to(self.device)

        if (self.device.type == 'cuda' and (self.num_gpu > 1)):
            mlp = nn.DataParallel(mlp, list(range(self.num_gpu)))

        criterion = nn.BCELoss()
        mlp_optimizer = optim.Adam(mlp.parameters(), lr = self.learning_rate, betas= self.betas)
        mlp_losses = []

        for epoch in range(self.num_epochs):
            for inputs, labels in mlp_loader:
                mlp_optimizer.zero_grad()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                output = mlp(inputs.view(-1, self.size_z ))
                mlp_loss = criterion(output, labels)

                mlp_loss.backward()

                mlp_optimizer.step()

                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {mlp_loss.item():.4f}")
                mlp_losses.append(mlp_loss.item())

        return mlp, mlp_losses