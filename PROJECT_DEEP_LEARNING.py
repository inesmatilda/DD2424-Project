import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# =============================================================================
# PART 0: DATA PREPARATION (Loading, Datasets, Transforms)
# =============================================================================

def load_cifar_batch(file_path):
    with open(file_path, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    raw_data = dict_data[b'data']
    data_reshaped = raw_data.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    labels = np.array(dict_data[b'labels'])
    return data_reshaped, labels

def calculate_mean_std(train_data):
    data_scaled = train_data.astype(np.float32) / 255.0
    mean = np.mean(data_scaled, axis=(0, 1, 2))
    std = np.std(data_scaled, axis=(0, 1, 2))
    return tuple(mean), tuple(std)

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data.astype(np.uint8)
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# =============================================================================
# PART 1: ASSIGNMENT 3 REPLICATION ARCHITECTURE
# =============================================================================

class Assignment3ReplicationNet(nn.Module):
    """
    Replicates the simple network from Assignment 3:
    f=4, nf=10, nh=50, K=10
    """
    def __init__(self):
        super(Assignment3ReplicationNet, self).__init__()
        # Patchify Layer (f=4, stride=4, nf=10)
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=4, stride=4, bias=True),
            nn.ReLU(inplace=True)
        )
        # Fully Connected Layers
        # 32x32 -> patchify(stride 4) -> 8x8 spatial size.
        # Flattened size = 10 channels * 8 * 8 = 640
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 50), # nh = 50
            nn.ReLU(inplace=True),
            nn.Linear(50, 10)   # K = 10
        )

    def forward(self, x):
        x = self.patchify(x)
        x = self.fc(x)
        return x


# =============================================================================
# PART 2 & 3 HELPER: VGG BLOCK
# =============================================================================

class VGGBlock(nn.Module):
    """Reusable VGG-style block: two 3x3 convolutions + BN + ReLU + optional MaxPool"""
    def __init__(self, in_channels, out_channels, apply_pooling=True):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.apply_pooling = apply_pooling
        if self.apply_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        if self.apply_pooling:
            x = self.pool(x)
        return x


# =============================================================================
# PART 2: FIRST UPGRADE ARCHITECTURE
# =============================================================================

class FirstUpgradeNet(nn.Module):
    """
    Patchify (f=2, 64 filters) -> 1 VGG Block -> Linear(128) -> Linear(10)
    """
    def __init__(self):
        super(FirstUpgradeNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block1 = VGGBlock(in_channels=64, out_channels=64, apply_pooling=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)     
        )

    def forward(self, x):
        x = self.patchify(x)
        x = self.block1(x)
        x = self.fc(x)
        return x


# =============================================================================
# PART 3: FULL BASELINE ARCHITECTURE
# =============================================================================

class FullBaselineNet(nn.Module):
    """
    Patchify -> 3 VGG Blocks (doubling filters, no pool on last) -> Global Avg Pool -> Linear(10) 
    """
    def __init__(self):
        super(FullBaselineNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block1 = VGGBlock(in_channels=64, out_channels=64, apply_pooling=True)
        self.block2 = VGGBlock(in_channels=64, out_channels=128, apply_pooling=True)
        
        # Final block has no max-pooling
        self.block3 = VGGBlock(in_channels=128, out_channels=256, apply_pooling=False)
        
        # Replace flattening/massive FC layers with Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, 10) 

    def forward(self, x):
        x = self.patchify(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =============================================================================
# PART 4 & 5: VERSATILE REGULARIZED ARCHITECTURE
# =============================================================================

class RegularizedNet(nn.Module):
    """
    Base network that supports toggling Dropout.
    BN is natively included in the VGGBlocks.
    """
    def __init__(self, dropout_p=0.0):
        super(RegularizedNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.block1 = VGGBlock(in_channels=64, out_channels=64, apply_pooling=True)
        self.block2 = VGGBlock(in_channels=64, out_channels=128, apply_pooling=True)
        self.block3 = VGGBlock(in_channels=128, out_channels=256, apply_pooling=False)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # We inject Dropout right before the final classification
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(256, 10) 

    def forward(self, x):
        x = self.patchify(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout (only active during model.train())
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# =============================================================================
# REUSABLE TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, val_loader, optimizer, epochs, device, scheduler=None):
    """A clean, reusable training loop so we don't duplicate code in main."""
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Starting Training for {epochs} Epochs ---")
    for epoch in range(epochs):
        model.train() 
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Step the scheduler per batch if CyclicLR is used
            if scheduler is not None:
                scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        # Validation Phase
        model.eval() 
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # 1. SETUP DATA 
    data_dir = '/Users/mikolaj/Desktop/Deep Learning/'
    
    print("Loading local CIFAR-10 files...")
    train_data_list, train_labels_list = [], []
    for i in range(1, 6):
        data, labels = load_cifar_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data_list.append(data)
        train_labels_list.append(labels)
        
    train_data_full = np.vstack(train_data_list)
    train_labels_full = np.concatenate(train_labels_list)
    val_size = 1000
    train_data, val_data = train_data_full[:-val_size], train_data_full[-val_size:]
    train_labels, val_labels = train_labels_full[:-val_size], train_labels_full[-val_size:]
    
    train_mean, train_std = calculate_mean_std(train_data)
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    
    train_loader = DataLoader(CustomCIFAR10Dataset(train_data, train_labels, train_transform), batch_size=128, shuffle=True)
    no_aug_train_loader = DataLoader(CustomCIFAR10Dataset(train_data, train_labels, eval_transform), batch_size=128, shuffle=True)
    val_loader   = DataLoader(CustomCIFAR10Dataset(val_data, val_labels, eval_transform), batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")


    # ---------------------------------------------------------
    # RUN TESTS OF DIFFERNET NETWORKS 
    # ---------------------------------------------------------
   
    print("\n>>> RUNNING PART 1: Assignment 3 Replication")
    model = Assignment3ReplicationNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.003)
    # PyTorch implementation of the CyclicLR from Assignment 3
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=800, mode='triangular')
    train_model(model, train_loader, val_loader, optimizer, epochs=10, device=device, scheduler=scheduler)

    print("\n>>> RUNNING PART 2: First Upgrade (1 VGG Block)")
    model = FirstUpgradeNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    train_model(model, train_loader, val_loader, optimizer, epochs=25, device=device)

    print("\n>>> RUNNING PART 3: Full Baseline (3 VGG Blocks + GAP)")
    model = FullBaselineNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    train_model(model, train_loader, val_loader, optimizer, epochs=25, device=device)

    print("\n>>> RUNNING PART 4a: Dropout ONLY")
    # Dropout = 50%, No Weight Decay, No Data Augmentation
    model = RegularizedNet(dropout_p=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0) 
    train_model(model, no_aug_train_loader, val_loader, optimizer, epochs=25, device=device)

    print("\n>>> RUNNING PART 4b: L2 Regularization (Weight Decay) ONLY")
    # No Dropout, Heavy Weight Decay (L2), No Data Augmentation
    model = RegularizedNet(dropout_p=0.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2) # 1e-2 is strong L2
    train_model(model, no_aug_train_loader, val_loader, optimizer, epochs=25, device=device)

    print("\n>>> RUNNING PART 4c: Data Augmentation ONLY")
    # No Dropout, No Weight Decay, YES Data Augmentation
    model = RegularizedNet(dropout_p=0.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    # Note: We pass `train_loader` here instead of `no_aug_train_loader`
    train_model(model, train_loader, val_loader, optimizer, epochs=25, device=device)

    print("\n>>> RUNNING PART 5: Combined Regularizations")
    # Dropout = 50%, Moderate Weight Decay, YES Data Augmentation
    model = RegularizedNet(dropout_p=0.5).to(device)
        
    # Because BN + Dropout stabilizes training, we can push the learning rate slightly higher 
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        
    # We must train for longer because heavy regularization slows down memorization 
    train_model(model, train_loader, val_loader, optimizer, epochs=50, device=device)