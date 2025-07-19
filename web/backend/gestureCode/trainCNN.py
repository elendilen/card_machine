import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import numpy.random as random
import os
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

class KeypointDataset(Dataset):
    def __init__(self, tensor_path):
        data = torch.load(tensor_path)
        self.X=data['X'].reshape(-1, 1,6,7)
        self.Y=data['y']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class GestureCNN(nn.Module):
    def __init__(self,num_classes=8):
        super(GestureCNN,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1).squeeze(-1)  # shape: (B, 64)
        return self.fc(x)


def train_model(train_path="keypoints_train.pt", val_path="keypoints_val.pt", save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_set = KeypointDataset(train_path)
    val_set = KeypointDataset(val_path)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    # 模型与优化器
    model = GestureCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience = 5
    wait = 0

    for epoch in range(50):
        model.train()
        total_loss, total_correct = 0, 0

        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1} Train]"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (out.argmax(1) == y).sum().item()

        train_acc = total_correct / len(train_set)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")

        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()

        val_acc = val_correct / len(val_set)
        print(f"Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model2.pt"))
            print("Saved new best model!")
        else:
            wait += 1
            print(f"No improvement. ({wait}/{patience})")

        #早停
        if wait >= patience:
            print("Early stopping triggered.")
            break

#启动训练
if __name__ == "__main__":
    train_model()