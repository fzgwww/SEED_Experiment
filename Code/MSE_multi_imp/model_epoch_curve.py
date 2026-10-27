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

seed = 45
set_seed(seed)
print(f"随机种子已设置为: {seed}")

eeg_train = eeg_train_data
eye_train = eye_train_data
y_train = train_label
eeg_test = eeg_test_data
eye_test = eye_test_data
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_classes)
        )
    
    def forward(self,eye_input,eeg_input):
        eye_features = self.eye_encoder(eye_input)
        eeg_features = self.eeg_encoder(eeg_input)
        output = self.classifier(eye_features)
        return output,eye_features,eeg_features

# ================= 修正后的训练与记录部分 =================
acc_list = []
neg_list = []
neu_list = []
pos_list = []
epoch_list = []

# 1. 模型、损失函数、优化器 只能初始化一次！放在循环外！
model = MiniCANDA()
criterion_cls = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 2. 设置总共要跑的 100 个 epoch
total_epochs = 500

for epoch in range(1, total_epochs + 1):
    # --- 训练阶段 ---
    model.train()
    total_loss, total_cls, total_mse = 0, 0, 0
    for batch_eye, batch_eeg, batch_y in train_loader:
        optimizer.zero_grad()
        
        outputs, eye_feature, eeg_feature = model(batch_eye, batch_eeg)

        loss_cls = criterion_cls(outputs, batch_y)
        loss_mse = criterion_mse(eye_feature, eeg_feature)
        loss = loss_cls * (loss_mse ** 0.2) 

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_mse += loss_mse.item()

    # --- 测试与记录阶段（只有在偶数 Epoch 才记录，实现 step=2） ---
    if epoch % 2 == 0:
        with torch.no_grad():
            model.eval()
            correct, total = 0, 0
            correct_neg, total_neg = 0, 0
            correct_neu, total_neu = 0, 0
            correct_pos, total_pos = 0, 0
            
            for batch_eye, batch_eeg, batch_y in test_loader:
                outputs, _, _ = model(batch_eye, batch_eeg)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                for i in range(batch_y.size(0)):
                    if batch_y[i] == 0:
                        total_neg += 1
                        if predicted[i] == 0: correct_neg += 1
                    elif batch_y[i] == 1:
                        total_neu += 1
                        if predicted[i] == 1: correct_neu += 1
                    else:
                        total_pos += 1
                        if predicted[i] == 2: correct_pos += 1
        
        # 记录数据
        epoch_list.append(epoch)
        acc_list.append(correct / total)
        neg_list.append(correct_neg / total_neg)
        neu_list.append(correct_neu / total_neu)
        pos_list.append(correct_pos / total_pos)
        
        print(f"Epoch {epoch:3d} | Acc: {correct/total:.4f} | Neg: {correct_neg/total_neg:.4f} | Neu: {correct_neu/total_neu:.4f} | Pos: {correct_pos/total_pos:.4f}")

# ================= 保存数据 =================
data_dict = {
    "epochs": epoch_list,
    "accuracy": acc_list,
    "neg_accuracy": neg_list,
    "neu_accuracy": neu_list,
    "pos_accuracy": pos_list
}
np.savez("MSE_multi_loss_results.npz", **data_dict)
print("数据已成功保存为 MSE_multi_loss_results.npz！")

# ================= 补全的画图代码 =================
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6), dpi=150)

# 画四条线：整体、负样本、中性、正样本
plt.plot(epoch_list, acc_list, label='Overall Accuracy', color='black', linewidth=2.5, marker='o', markevery=5)
plt.plot(epoch_list, pos_list, label='Positive (Pos)', color='green', linestyle='--', linewidth=1.5)
plt.plot(epoch_list, neu_list, label='Neutral (Neu)', color='orange', linestyle='--', linewidth=1.5)
plt.plot(epoch_list, neg_list, label='Negative (Neg)', color='red', linestyle='--', linewidth=1.5)

# 标出断崖相变的可能区域（根据你之前的图，大概是 70-80 之间）
plt.axvline(x=80, color='blue', linestyle=':', label='Phase Transition Area')

plt.title('Mini-CANDA Multiplicative MSE: Accuracy by Category', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()

# 保存并显示
plt.savefig('multi_loss_category_curve.png')
plt.show()