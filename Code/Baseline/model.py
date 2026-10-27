import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from data_process_pro import *

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 强制 PyTorch 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 45
set_seed(seed)
print(f"随机种子已设置为: {seed}")

x_train = eye_train_data
y_train = train_label
x_test = eye_test_data
y_test = test_label

print(f"眼动训练集: {x_train.shape}, 真实标签: {y_train.shape}")
print(f"眼动测试集: {x_test.shape}, 真实标签: {y_test.shape}")

x_train_t=torch.tensor(x_train, dtype=torch.float32)
y_train_t=torch.tensor(y_train, dtype=torch.long)
x_test_t=torch.tensor(x_test, dtype=torch.float32)
y_test_t=torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=128, shuffle=False)

class SimpleEyeNet(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_classes):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            #nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            #nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleEyeNet(input_dim=33, hidden_dim=64, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

epochs = 50
loss_arr = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x,batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs,batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_arr.append(total_loss/len(train_loader))
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

with torch.no_grad():
    model.eval()
    correct = 0
    total = 0

    correct_neg=0
    total_neg=0
    correct_neu=0
    total_neu=0
    correct_pos=0
    total_pos=0
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        # 统计各类别的准确率
        for i in range(batch_y.size(0)):
            if batch_y[i] == 0:
                total_neg += 1
                if predicted[i] == 0:
                    correct_neg += 1
            elif batch_y[i] == 1:
                total_neu += 1
                if predicted[i] == 1:
                    correct_neu += 1
            else:
                total_pos += 1
                if predicted[i] == 2:
                    correct_pos += 1

accuracy = correct / total
print(f"测试集准确率: {accuracy:.4f}")

# 打印各类别准确率
if total_neg > 0:
    print(f"负样本准确率: {correct_neg / total_neg:.4f}")
if total_neu > 0:
    print(f"中性样本准确率: {correct_neu / total_neu:.4f}")
if total_pos > 0:
    print(f"正样本准确率: {correct_pos / total_pos:.4f}")
print(f"Seed={seed} Accuracy={accuracy:.4f} Neg={correct_neg / total_neg:.4f} Neu={correct_neu / total_neu:.4f} Pos={correct_pos / total_pos:.4f}")

import matplotlib.pyplot as plt

# plt.plot(loss_arr)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.show()