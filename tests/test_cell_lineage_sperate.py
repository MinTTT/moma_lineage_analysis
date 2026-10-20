import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib

# Training runs must never open GUI figure windows. Figures are rendered to PNG
# files instead; set MOMA_SHOW_FIGURES=1 to also display them interactively.
SHOW_FIGURES = os.environ.get("MOMA_SHOW_FIGURES", "0") == "1"
if not SHOW_FIGURES:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

FIGURES_DIR = Path(__file__).resolve().parent / "figures_cell_lineage_experiment"


def _finish_figure(name: str) -> None:
    """Save the current figure to PNG; never pop a window unless opted in."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"figure saved: {path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close()


# This file is a standalone training experiment (it trains and plots at module
# level), not a pytest module. Running it under pytest would start a full
# training job; keep it runnable directly with python.
if "pytest" in sys.modules:
    import pytest

    pytest.skip(
        "manual training experiment; run directly with python",
        allow_module_level=True,
    )


# ========== 1. 轨迹生成 ==========
def simulate_trajectory(n_points=200, noise=0.05, div_threshold=5.0, split_window=0):
    times, lengths, split_flags = [], [], []
    t = np.random.uniform(0, 100)
    dt = 0.1
    length = np.random.uniform(div_threshold/2, div_threshold)

    mode = np.random.choice(["linear", "log"])
    growth_rate = np.random.uniform(0.2, 1.8)
    div_thre_random = div_threshold + np.random.randn() * 0.2
    for i in range(n_points):

        times.append(t)
        lengths.append(length + np.random.randn() * noise)
        split_flags.append(0)

        if mode == "linear":
            length += growth_rate * dt + np.random.randn() * 0.01
        else:
            length *= (1 + growth_rate * dt + np.random.randn() * 0.01)

        t += dt
        if length >= div_thre_random:
            # 标记分裂点 ±window
            for k in range(-split_window, split_window + 1):
                idx = i + k
                if 0 <= idx < len(split_flags):
                    split_flags[idx] = 1
            length = length / 2.0 + np.random.randn() * noise
            div_thre_random = div_threshold + np.random.randn() * 0.6

    return np.array(times, dtype=np.float32), np.array(lengths, dtype=np.float32), np.array(split_flags, dtype=np.int64)


# ========== 2. 数据集 ==========
class CellTrajectoryDataset(Dataset):
    def __init__(self, n_traj=50, n_points=200, div_threshold=5.0, q99=None, split_window=0):
        self.trajs, self.labels = [], []
        all_lengths = []

        for _ in range(n_traj):
            times, lengths, flags = simulate_trajectory(
                n_points=n_points, div_threshold=div_threshold, split_window=split_window
            )
            dlength = np.diff(lengths, prepend=lengths[0])
            ddlength = np.diff(dlength, prepend=dlength[0])
            traj = np.stack([times, lengths, dlength, ddlength], axis=1)
            self.trajs.append(traj)
            self.labels.append(flags)
            all_lengths.extend(lengths)

        if q99 is None:
            self.q99 = np.percentile(all_lengths, 99)
        else:
            self.q99 = q99

        for traj in self.trajs:
            traj[:,1] /= self.q99
            traj[:,2] /= self.q99
            traj[:,3] /= self.q99

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return torch.tensor(self.trajs[idx]), torch.tensor(self.labels[idx])


# ========== 3. collate_fn ==========
def collate_fn(batch):
    trajs, labels = zip(*batch)
    lengths = [len(t) for t in trajs]
    max_len = max(lengths)

    padded_trajs, padded_labels, masks = [], [], []
    for t, l in zip(trajs, labels):
        pad_len = max_len - len(t)
        padded_trajs.append(torch.cat([t, torch.zeros(pad_len, t.shape[1])], dim=0))
        padded_labels.append(torch.cat([l, torch.full((pad_len,), -100)]))
        masks.append(torch.cat([torch.zeros(len(t)), torch.ones(pad_len)]))

    return torch.stack(padded_trajs), torch.stack(padded_labels), torch.stack(masks)


# ========== 4. Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha_t = torch.zeros_like(targets, dtype=torch.float)
            for i, w in enumerate(self.alpha):
                alpha_t[targets == i] = w
            ce_loss = alpha_t * ce_loss
        focal_loss = (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


# ========== 5. CNN + BiLSTM 模型 ==========
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, num_classes=2):
        super().__init__()
        # CNN 提取局部模式
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # BiLSTM
        self.lstm = nn.LSTM(32, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)          # (batch, input_dim, seq_len)
        x = self.relu(self.conv1(x))    # (batch, 32, seq_len)
        x = x.permute(0, 2, 1)          # (batch, seq_len, 32)
        out, _ = self.lstm(x)
        out = self.fc(out)              # (batch, seq_len, 2)
        return out


# ========== 6. 数据准备 ==========
train_dataset = CellTrajectoryDataset(n_traj=1000, n_points=200, split_window=0)
test_dataset = CellTrajectoryDataset(n_traj=300, n_points=200, q99=train_dataset.q99, split_window=0)

traj_split_counts = [labels.sum() for labels in train_dataset.labels]
sample_weights = [2.0 if c > 0 else 1.0 for c in traj_split_counts]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)


# ========== 7. 训练 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_BiLSTM().to(device)

criterion = FocalLoss(alpha=[1.0, 1.0], gamma=2, ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses = []
for epoch in range(50):
    model.train()
    total_loss = 0
    for X, y, mask in train_loader:
        X, y, mask = X.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred.view(-1, 2), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss={avg_loss:.4f}")


# ========== 8. 测试 ==========
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X, y, mask in test_loader:
        X, y, mask = X.to(device), y.to(device), mask.to(device)
        pred = model(X)
        pred_labels = pred.argmax(dim=-1).cpu().numpy()
        true_labels = y.cpu().numpy()
        for p, t in zip(pred_labels, true_labels):
            valid = (t != -100)
            all_preds.extend(p[valid])
            all_true.extend(t[valid])

print(classification_report(all_true, all_preds, target_names=["Normal", "Split"], zero_division=0))


# ========== 9. 可视化 ==========
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training Loss Curve (CNN + BiLSTM)")
plt.legend()
_finish_figure("training_loss_curve")

#%%
def visualize_true_vs_pred(model, test_loader, device, n_samples=3):
    """
    在同一张图上显示真实和预测的分裂点
    """
    model.eval()
    samples_shown = 0

    with torch.no_grad():
        for X, y, mask in test_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            pred = model(X)
            pred_labels = pred.argmax(dim=-1).cpu().numpy()
            true_labels = y.cpu().numpy()
            trajs = X.cpu().numpy()

            batch_size = X.shape[0]
            for i in range(batch_size):
                if samples_shown >= n_samples:
                    return
                valid = (true_labels[i] != -100)
                times = trajs[i, valid, 0]
                lengths = trajs[i, valid, 1] * test_loader.dataset.q99  # 还原尺度
                true_split = true_labels[i][valid]
                pred_split = pred_labels[i][valid]

                # 绘制轨迹
                plt.figure(figsize=(10,5))
                plt.plot(times, lengths, color="gray", alpha=0.6, label="Trajectory")

                # 标记真实分裂点
                plt.scatter(times[true_split==1], lengths[true_split==1],
                            c="red", marker="o", label="True Split", s=40)

                # 标记预测分裂点
                plt.scatter(times[pred_split==1], lengths[pred_split==1],
                            c="blue", marker="x", label="Pred Split", s=50)

                plt.title(f"Trajectory {samples_shown+1}: True vs Predicted Splits")
                plt.xlabel("Time")
                plt.ylabel("Cell Length")
                plt.legend()
                _finish_figure(f"true_vs_pred_trajectory_{samples_shown+1}")

                samples_shown += 1

visualize_true_vs_pred(model, test_loader, device, n_samples=3)
