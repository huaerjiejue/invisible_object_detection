import torch
import torch.nn as nn
import torch.optim as optim
from utils.config import Config
from models.model import InvisibleObjectDetector
from utils.metrics import calculate_map
from utils.nms import non_max_suppression
import numpy as np

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

if __name__ == '__main__':
    config = Config('configs/model_config.yaml', 'configs/train_config.yaml')
    model = InvisibleObjectDetector(config)
    train_loader = ...
    val_loader = ... 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_model(model, train_loader, val_loader, device)
