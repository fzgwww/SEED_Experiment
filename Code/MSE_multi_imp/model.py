import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from torch.utils.data import DataLoader, TensorDataset
from data_process import *

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 强制 PyTorch 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 41
set_seed(seed)
print(f"随机种子已设置为: {seed}")

from sklearn.preprocessing import StandardScaler

scaler_eeg = StandardScaler()
scaler_eye = StandardScaler()
eeg_train = scaler_eeg.fit_transform(eeg_train_data)
eye_train = scaler_eye.fit_transform(eye_train_data)
y_train = train_label

eeg_test = scaler_eeg.transform(eeg_test_data)
eye_test = scaler_eye.transform(eye_test_data)
y_test = test_label

# print(f"EEG训练集: {eeg_train.shape}, 眼动训练集: {eye_train.shape}, 真实标签: {y_train.shape}")
# print(f"EEG测试集: {eeg_test.shape}, 眼动测试集: {eye_test.shape}, 真实标签: {y_test.shape}")

eeg_train_t=torch.tensor(eeg_train, dtype=torch.float32)
eye_train_t=torch.tensor(eye_train, dtype=torch.float32)
y_train_t=torch.tensor(y_train, dtype=torch.long)
eeg_test_t=torch.tensor(eeg_test, dtype=torch.float32)
eye_test_t=torch.tensor(eye_test, dtype=torch.float32)
y_test_t=torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(eye_train_t,eeg_train_t, y_train_t), batch_size=128, shuffle=True)
test_loader = DataLoader(
    TensorDataset(eye_test_t,eeg_test_t, y_test_t), batch_size=128, shuffle=False)

class MiniCANDA(nn.Module):
    def __init__(self, eye_dim=33,eeg_dim=310, hidden_dim=80, num_classes=3):
        super().__init__()
        self.eye_encoder = nn.Sequential(
            nn.Linear(eye_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_classes)
        )
    
    def forward(self,eye_input,eeg_input):
        eye_features = self.eye_encoder(eye_input)
        eeg_features = self.eeg_encoder(eeg_input)
        output = self.classifier(eye_features)
        return output,eye_features,eeg_features

model = MiniCANDA()
criterion_cls = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=5e-4)

epochs = 50
alpha = 1e-4
loss_arr = []
for epoch in range(epochs):
    model.train()
    total_loss,total_cls,total_mse = 0,0,0
    for batch_eye,batch_eeg,batch_y in train_loader:
        optimizer.zero_grad()
        
        outputs,eye_feature,eeg_feature = model(batch_eye,batch_eeg)

        loss_cls = criterion_cls(outputs,batch_y)
        loss_mse = criterion_mse(eye_feature,eeg_feature)
        loss = loss_cls*(loss_mse**0.2)  # 结合分类损失和MSE损失

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_mse += loss_mse.item()
    if (epoch+1>=5):
        loss_arr.append(total_loss/len(train_loader))
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Classification Loss: {total_cls/len(train_loader):.4f}, MSE Loss: {total_mse/len(train_loader):.4f}")

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
    for batch_eye,batch_eeg,batch_y in test_loader:
        outputs,_,_ = model(batch_eye,batch_eeg)
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

plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()