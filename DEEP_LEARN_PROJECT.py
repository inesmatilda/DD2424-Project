import os
import pickle
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# =============================================================================
# PART 0: DATA PREPARATION
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
# PART 1: ASSIGNMENT 3 REPLICATION (The Sanity Check)
# =============================================================================

class Assignment3ReplicationNet(nn.Module):
    """
    Replicates the simple network from Assignment 3:
    Patchify (f=4) -> Flatten -> FC(50) -> FC(10)
    """
    def __init__(self):
        super(Assignment3ReplicationNet, self).__init__()
        # Patchify Layer (f=4, stride=4, 10 filters)
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=4, stride=4, bias=True),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 8 * 8, 50), # nh = 50
            nn.ReLU(inplace=True),
            nn.Linear(50, 10)          # 10 classes
        )

    def forward(self, x):
        x = self.patchify(x)
        x = self.fc(x)
        return x

# =============================================================================
# PARTS 2 & 3: THE "RAW" BASELINE 
# =============================================================================

class RawVGGBlock(nn.Module):
    """Bare-bones VGG block matching the start of the ML Mastery tutorial."""
    def __init__(self, in_channels, out_channels, apply_pooling=True):
        super(RawVGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.apply_pooling = apply_pooling
        if self.apply_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        if self.apply_pooling:
            x = self.pool(x)
        return x

class FirstUpgradeNet(nn.Module):
    """Patchify -> 1 Raw VGG Block -> Massive FC Layer"""
    def __init__(self):
        super(FirstUpgradeNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.block1 = RawVGGBlock(in_channels=64, out_channels=64, apply_pooling=True)
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

class FullBaselineNet(nn.Module):
    """Patchify -> 3 Raw VGG Blocks -> Massive FC Layer"""
    def __init__(self):
        super(FullBaselineNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.block1 = RawVGGBlock(in_channels=64, out_channels=64, apply_pooling=True)
        self.block2 = RawVGGBlock(in_channels=64, out_channels=128, apply_pooling=True)
        self.block3 = RawVGGBlock(in_channels=128, out_channels=256, apply_pooling=False)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)     
        )

    def forward(self, x):
        x = self.patchify(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)
        return x

# =============================================================================
# PART 4 & 5: VERSATILE REGULARIZED ARCHITECTURE
# =============================================================================

class RegVGGBlock(nn.Module):
    """VGG Block upgraded with Batch Norm and Dropout (ML Mastery aligned)."""
    def __init__(self, in_channels, out_channels, apply_pooling=True, dropout_p=0.0):
        super(RegVGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.apply_pooling = apply_pooling
        if self.apply_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        if self.apply_pooling:
            x = self.pool(x)
        x = self.dropout(x) 
        return x

class RegularizedNet(nn.Module):
    """The fully regularized baseline with a massive FC layer."""
    def __init__(self, block_dropout=0.2, fc_dropout=0.5):
        super(RegularizedNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block1 = RegVGGBlock(64, 64, apply_pooling=True, dropout_p=block_dropout)
        self.block2 = RegVGGBlock(64, 128, apply_pooling=True, dropout_p=block_dropout)
        self.block3 = RegVGGBlock(128, 256, apply_pooling=False, dropout_p=block_dropout)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.BatchNorm1d(128),     # BN after first FC layer as per assignment
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),# Heavy 0.5 Dropout before classification
            nn.Linear(128, 10) 
        )

    def forward(self, x):
        x = self.patchify(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)
        return x

# =============================================================================
# PART 6c: STRIDE-BASED DOWNSAMPLING ARCHITECTURE
# =============================================================================

class StrideVGGBlock(nn.Module):
    """VGG Block that down-samples using a Stride=2 Convolution instead of MaxPool."""
    def __init__(self, in_channels, out_channels, downsample=True, dropout_p=0.0):
        super(StrideVGGBlock, self).__init__()
        # First conv always has stride 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second conv handles the down-sampling if downsample=True
        stride = 2 if downsample else 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
            
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout(x) 
        return x

class StrideNet(nn.Module):
    """Network using StrideVGGBlocks instead of standard RegVGGBlocks."""
    def __init__(self, block_dropout=0.2, fc_dropout=0.5):
        super(StrideNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 1 and 2 perform down-sampling (stride=2)
        self.block1 = StrideVGGBlock(64, 64, downsample=True, dropout_p=block_dropout)
        self.block2 = StrideVGGBlock(64, 128, downsample=True, dropout_p=block_dropout)
        # Block 3 does NOT down-sample, as requested by the assignment
        self.block3 = StrideVGGBlock(128, 256, downsample=False, dropout_p=block_dropout)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.BatchNorm1d(128),     
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),
            nn.Linear(128, 10) 
        )

    def forward(self, x):
        x = self.patchify(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)
        return x
    
# =============================================================================
# PART 6d: GLOBAL AVERAGE POOLING (GAP) ARCHITECTURE
# =============================================================================

class GapNet(nn.Module):
    """Network replacing the massive FC layer with Global Average Pooling."""
    def __init__(self, block_dropout=0.2):
        super(GapNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Standard Regularized VGG Blocks
        self.block1 = RegVGGBlock(64, 64, apply_pooling=True, dropout_p=block_dropout)
        self.block2 = RegVGGBlock(64, 128, apply_pooling=True, dropout_p=block_dropout)
        self.block3 = RegVGGBlock(128, 256, apply_pooling=False, dropout_p=block_dropout)
        
        # Global Average Pooling replaces the entire 4096-node FC sequence!
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # We map directly from the 256 channels to our 10 classes
        self.classifier = nn.Linear(256, 10) 

    def forward(self, x):
        x = self.patchify(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Squash spatial dimensions (Height/Width) to 1x1
        x = self.gap(x)
        
        # Flatten from (Batch, 256, 1, 1) to (Batch, 256)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =============================================================================
# THE ULTIMATE BASELINE (Best Performance & Speed)
# =============================================================================

class UltimateNet(nn.Module):
    """Combines GAP, RegVGGBlocks (MaxPool), and eliminates massive FC layers."""
    def __init__(self, block_dropout=0.2):
        super(UltimateNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Standard VGG Blocks with Batch Norm and Block Dropout
        self.block1 = RegVGGBlock(64, 64, apply_pooling=True, dropout_p=block_dropout)
        self.block2 = RegVGGBlock(64, 128, apply_pooling=True, dropout_p=block_dropout)
        self.block3 = RegVGGBlock(128, 256, apply_pooling=False, dropout_p=block_dropout)
        
        # Global Average Pooling replaces the FC layer completely
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final Classifier
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
# PART 7: A/B GRADE EXTENSIONS (Tested on the Clean Part 5 + GAP Baseline)
# =============================================================================

# --- 7a: SQUEEZE-AND-EXCITATION ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pooling=True, dropout_p=0.0):
        super(SEVGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) # SE injected here!
        self.apply_pooling = apply_pooling
        if self.apply_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.se(x) 
        if self.apply_pooling:
            x = self.pool(x)
        return self.dropout(x)

# --- 7b & 7c: CUSTOM NORMALIZATIONS ---
class RMSNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super(RMSNorm2d, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=(1, 2, 3), keepdim=True) + self.eps)
        return (x / rms) * self.weight

class RLayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(RLayerNorm2d, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.noise_threshold = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        noise_gate = torch.sigmoid(self.noise_threshold - torch.abs(x_norm))
        x_robust = x_norm * noise_gate
        return x_robust * self.weight + self.bias

class NormVGGBlock(nn.Module):
    """Generic block where we can pass in RMSNorm or RLayerNorm."""
    def __init__(self, in_channels, out_channels, norm_layer, apply_pooling=True, dropout_p=0.0):
        super(NormVGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.apply_pooling = apply_pooling
        if self.apply_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        if self.apply_pooling:
            x = self.pool(x)
        return self.dropout(x)

# --- 7d: LARGE KERNEL (ConvNeXt-style) ---
class LargeKernelVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pooling=True, dropout_p=0.0):
        super(LargeKernelVGGBlock, self).__init__()
        # 7x7 kernel with padding=3 to keep spatial dimensions identical
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.apply_pooling = apply_pooling
        if self.apply_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        if self.apply_pooling:
            x = self.pool(x)
        return self.dropout(x)

# --- THE MASTER NETWORK BUILDER ---
class ExtNet(nn.Module):
    """A single network class where we plug in the specific block type we want to test."""
    def __init__(self, block_class, norm_layer=None, block_dropout=0.2):
        super(ExtNet, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=False),
            norm_layer(64) if norm_layer else nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        if norm_layer:
            self.block1 = block_class(64, 64, norm_layer, apply_pooling=True, dropout_p=block_dropout)
            self.block2 = block_class(64, 128, norm_layer, apply_pooling=True, dropout_p=block_dropout)
            self.block3 = block_class(128, 256, norm_layer, apply_pooling=False, dropout_p=block_dropout)
        else:
            self.block1 = block_class(64, 64, apply_pooling=True, dropout_p=block_dropout)
            self.block2 = block_class(64, 128, apply_pooling=True, dropout_p=block_dropout)
            self.block3 = block_class(128, 256, apply_pooling=False, dropout_p=block_dropout)
            
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, 10) 

    def forward(self, x):
        x = self.block3(self.block2(self.block1(self.patchify(x))))
        return self.classifier(torch.flatten(self.gap(x), 1))
    

# =============================================================================
# REUSABLE TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, val_loader, optimizer, epochs, device, scheduler=None, criterion=None):
    # If no custom loss is provided, use standard CrossEntropy
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    start_time = time.time() # Start the total timer
    
    for epoch in range(epochs):
        model.train() 
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss using the dynamically passed criterion
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        model.eval() 
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Calculate validation loss using the same criterion
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
              
    total_time = time.time() - start_time
    print(f"\n--- Training Complete in {total_time/60:.2f} minutes ---")
    return total_time

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. DATA SETUP 
    # ---------------------------------------------------------
    
    data_dir = '/Users/mikolaj/Desktop/Deep Learning/'
    
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
    
    # Transforms WITH Augmentation (Flips & Shifts)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    
    # Transforms WITHOUT Augmentation (Raw Images)
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    
    # DataLoaders
    train_loader        = DataLoader(CustomCIFAR10Dataset(train_data, train_labels, train_transform), batch_size=128, shuffle=True)
    no_aug_train_loader = DataLoader(CustomCIFAR10Dataset(train_data, train_labels, eval_transform), batch_size=128, shuffle=True)
    val_loader          = DataLoader(CustomCIFAR10Dataset(val_data, val_labels, eval_transform), batch_size=128, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # <--- This uses the Apple Silicon GPU!
    else:
        device = torch.device("cpu")
    print(f"Device set to: {device}")

    # ---------------------------------------------------------
    # ADVANCED DATA SETUP (Needs to be declared before the switchboard)
    # ---------------------------------------------------------
    # 6a Transform (Cut-out/Random Erasing added)
    transform_6a = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.20)) 
    ])
    
    # 6a DataLoader (If you experience slowdowns on Mac, remember to add num_workers=0 here)
    loader_6a = DataLoader(CustomCIFAR10Dataset(train_data, train_labels, transform_6a), batch_size=128, shuffle=True)

    # ---------------------------------------------------------
    # SWITCHBOARD: Change EXPERIMENT_TO_RUN to test different parts
    # Options: "1", "2", "3", "4a", "4b", "4c", "5", "6a", "6b", "6c", "6d", "ultimate"
    # ---------------------------------------------------------
    EXPERIMENT_TO_RUN = "8"  # Change this variable to run different experiments

    if EXPERIMENT_TO_RUN == "1":
        print("\n>>> RUNNING PART 1: Assignment 3 Replication (Sanity Check)")
        model = Assignment3ReplicationNet().to(device)
        
        # Using SGD and CyclicLR as requested by the assignment
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.003)
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=800, mode='triangular'
        )
        # Note: we use the loader WITHOUT augmentation for strict Assignment 3 replication
        train_model(model, no_aug_train_loader, val_loader, optimizer, epochs=10, device=device, scheduler=scheduler)
    
    elif EXPERIMENT_TO_RUN == "2":
        print("\n>>> RUNNING PART 2: First Upgrade (1 Raw VGG Block - Expect Overfitting)")
        model = FirstUpgradeNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        train_model(model, no_aug_train_loader, val_loader, optimizer, epochs=25, device=device)

    elif EXPERIMENT_TO_RUN == "3":
        print("\n>>> RUNNING PART 3: Full Baseline (3 Raw VGG Blocks - Expect Heavy Overfitting)")
        model = FullBaselineNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        train_model(model, no_aug_train_loader, val_loader, optimizer, epochs=25, device=device)

    elif EXPERIMENT_TO_RUN == "4a":
        print("\n>>> RUNNING PART 4a: Dropout ONLY")
        model = RegularizedNet(block_dropout=0.2, fc_dropout=0.5).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0) 
        train_model(model, no_aug_train_loader, val_loader, optimizer, epochs=25, device=device)

    elif EXPERIMENT_TO_RUN == "4b":
        print("\n>>> RUNNING PART 4b: L2 Regularization ONLY")
        model = RegularizedNet(block_dropout=0.0, fc_dropout=0.0).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3) 
        train_model(model, no_aug_train_loader, val_loader, optimizer, epochs=25, device=device)

    elif EXPERIMENT_TO_RUN == "4c":
        print("\n>>> RUNNING PART 4c: Data Augmentation ONLY")
        model = RegularizedNet(block_dropout=0.0, fc_dropout=0.0).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        train_model(model, train_loader, val_loader, optimizer, epochs=25, device=device)

    elif EXPERIMENT_TO_RUN == "5":
        print("\n>>> RUNNING PART 5: Combined Regularization (Dropout + L2 + Augmentation + BatchNorm)")
        model = RegularizedNet(block_dropout=0.2, fc_dropout=0.5).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        train_model(model, train_loader, val_loader, optimizer, epochs=50, device=device)

    elif EXPERIMENT_TO_RUN == "6a":
        print("\n>>> RUNNING PART 6a: Adv. Augmentation (Cut-out + Label Smoothing)")
        model = RegularizedNet(block_dropout=0.2, fc_dropout=0.5).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        
        # Adding Label Smoothing (10%) via the custom criterion
        smooth_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # Make sure to pass loader_6a and our custom criterion
        train_model(model, loader_6a, val_loader, optimizer, epochs=50, device=device, criterion=smooth_criterion)

    elif EXPERIMENT_TO_RUN == "6b":
        print("\n>>> RUNNING PART 6b: LR Scheduler (Cosine Annealing with Warm Restarts)")
        # We use the standard regularized model from Part 5
        model = RegularizedNet(block_dropout=0.2, fc_dropout=0.5).to(device)
        # We start with a higher learning rate (5e-3 instead of 1e-3) 
        optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
        
        # Because train_model calls scheduler.step() EVERY BATCH, T_0 must be calculated in total batches
        batches_per_epoch = len(train_loader)
        restarts_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10 * batches_per_epoch,  
            T_mult=1,                    # Keeps the restart interval fixed at 10 epochs
            eta_min=1e-5                 # The lowest the learning rate will go before restarting
        )
        # passing our restarts_scheduler here
        train_model(model, train_loader, val_loader, optimizer, epochs=50, device=device, scheduler=restarts_scheduler)

    elif EXPERIMENT_TO_RUN == "6c":
        print("\n>>> RUNNING PART 6c: Stride 2 Down-sampling (No MaxPool)")
        # We instantiate our new StrideNet 
        model = StrideNet(block_dropout=0.2, fc_dropout=0.5).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        train_model(model, train_loader, val_loader, optimizer, epochs=50, device=device)

    elif EXPERIMENT_TO_RUN == "6d":
        print("\n>>> RUNNING PART 6d: Global Average Pooling (Replacing FC Layer)")
        # Instantiate the GAP network. Notice we don't have an fc_dropout parameter anymore!
        model = GapNet(block_dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        # Train using the standard augmented train_loader
        train_model(model, train_loader, val_loader, optimizer, epochs=50, device=device)

    elif EXPERIMENT_TO_RUN == "ultimate":
        print("\n>>> RUNNING THE ULTIMATE BASELINE (GAP + Cosine Annealing + Adv. Augmentation)")
        # Instantiate our fastest, leanest model
        model = UltimateNet(block_dropout=0.2).to(device)
        # Start with a high learning rate (5e-3) for Cosine Annealing to sweep down from
        optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
        
        # 1. Cosine Annealing Scheduler (Restart every 10 epochs)
        batches_per_epoch = len(loader_6a)
        restarts_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10 * batches_per_epoch,  
            T_mult=1,                    
            eta_min=1e-5                 
        )
        
        # 2. Label Smoothing (10%)
        smooth_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 3. Train using the Cut-out dataloader (loader_6a)
        train_model(
            model, 
            loader_6a, 
            val_loader, 
            optimizer, 
            epochs=50, 
            device=device, 
            scheduler=restarts_scheduler,
            criterion=smooth_criterion
        )

    elif EXPERIMENT_TO_RUN == "7a":
        print("\n>>> RUNNING PART 7a: SE Blocks (Attention Ablation)")
        model = ExtNet(block_class=SEVGGBlock, block_dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        train_model(model, train_loader, val_loader, optimizer, epochs=50, device=device)

    elif EXPERIMENT_TO_RUN == "7b":
        print("\n>>> RUNNING PART 7b: RMSNorm (Speed Ablation)")
        model = ExtNet(block_class=NormVGGBlock, norm_layer=RMSNorm2d, block_dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        train_model(model, train_loader, val_loader, optimizer, epochs=50, device=device)

    elif EXPERIMENT_TO_RUN == "7c":
        print("\n>>> RUNNING PART 7c: R-LayerNorm (Noise/Stability Ablation)")
        model = ExtNet(block_class=NormVGGBlock, norm_layer=RLayerNorm2d, block_dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        train_model(model, train_loader, val_loader, optimizer, epochs=50, device=device)

    elif EXPERIMENT_TO_RUN == "7d":
        print("\n>>> RUNNING PART 7d: Large 7x7 Kernels (ConvNeXt Spatial Ablation)")
        model = ExtNet(block_class=LargeKernelVGGBlock, block_dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        train_model(model, train_loader, val_loader, optimizer, epochs=50, device=device)

    elif EXPERIMENT_TO_RUN == "8":
        print("\n>>> RUNNING PART 8: The Final SOTA Model (SE-Net + Ultimate Training)")
        
        # 1. The Winning Architecture: GAP Baseline + Squeeze-and-Excitation
        model = ExtNet(block_class=SEVGGBlock, block_dropout=0.2).to(device)
        
        # 2. The Winning Optimizer: AdamW (starting at a high 5e-3 for the scheduler)
        optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
        
        # 3. The Winning Scheduler: Cosine Annealing with Warm Restarts
        batches_per_epoch = len(loader_6a)
        restarts_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10 * batches_per_epoch,  
            T_mult=1,                    
            eta_min=1e-5                 
        )
        
        # 4. The Winning Loss Function: CrossEntropy with Label Smoothing
        smooth_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 5. The Winning Data: The loader with Cut-out / Random Erasing
        train_model(
            model, 
            loader_6a, 
            val_loader, 
            optimizer, 
            epochs=50, 
            device=device, 
            scheduler=restarts_scheduler,
            criterion=smooth_criterion
        )