from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from isl_shared import FEATURE_DIM, SEQUENCE_LENGTH


def stratified_split(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0].tolist()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        if n_val >= len(idx):
            n_val = len(idx) - 1
        if n_val <= 0:
            raise RuntimeError(f"Class {cls} has insufficient samples for train/val split")
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ISL dynamic LSTM and export ONNX")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.npz"))
    parser.add_argument("--labels", type=Path, default=Path("models/dynamic_labels.json"))
    parser.add_argument("--out", type=Path, default=Path("models/dynamic_lstm.onnx"))
    parser.add_argument("--meta", type=Path, default=Path("models/dynamic_meta.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--min-samples-per-class", type=int, default=20)
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as ex:
        raise RuntimeError(
            "PyTorch training environment not available. Use Python 3.11/3.12 for training, "
            "install with: pip install -r requirements-train.txt"
        ) from ex

    class ISLLSTM(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.25,
            )
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            logits = self.fc(last)
            return logits

    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += float(loss.item()) * xb.size(0)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())
                total += int(xb.size(0))
        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.dataset)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    if np.isnan(X).any() or np.isinf(X).any():
        raise RuntimeError("Dataset contains NaN/Inf values")

    if X.ndim != 3 or X.shape[1] != SEQUENCE_LENGTH or X.shape[2] != FEATURE_DIM:
        raise RuntimeError(
            f"Expected X shape [N,{SEQUENCE_LENGTH},{FEATURE_DIM}] but got {X.shape}. "
            "Use prepare_dataset.py from this project to generate dataset."
        )

    labels_map = json.loads(args.labels.read_text(encoding="utf-8"))
    num_classes = len(labels_map)
    if num_classes <= 1:
        raise RuntimeError("Need at least 2 classes to train")

    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) != num_classes:
        raise RuntimeError(
            f"Label map has {num_classes} classes but dataset has {len(unique_classes)} classes"
        )

    min_count = int(np.min(counts))
    max_count = int(np.max(counts))
    print(f"Dataset checks -> classes: {num_classes}, samples: {len(y)}, min/class: {min_count}, max/class: {max_count}")
    if min_count < args.min_samples_per_class:
        raise RuntimeError(
            f"Too few samples in at least one class ({min_count}). "
            f"Need >= {args.min_samples_per_class} per class for stable training."
        )

    both_hands = np.mean(np.any(np.abs(X[:, :, :126]) > 1e-6, axis=2))
    print(f"Hand-visibility ratio across frames: {both_hands:.3f}")

    train_idx, val_idx = stratified_split(y, args.val_ratio, args.seed)
    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ISLLSTM(input_dim=FEATURE_DIM, hidden_dim=args.hidden, num_layers=args.layers, num_classes=num_classes).to(device)
    class_count_map = {int(c): int(n) for c, n in zip(unique_classes, counts)}
    class_weights = np.array(
        [float(counts.max() / class_count_map[i]) for i in range(num_classes)],
        dtype=np.float32,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item()) * xb.size(0)
            seen += int(xb.size(0))

        train_loss = running_loss / max(1, seen)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed: best checkpoint unavailable")

    model.load_state_dict(best_state)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        export_params=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
        opset_version=args.opset,
    )

    meta = {
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": FEATURE_DIM,
        "num_classes": num_classes,
        "val_accuracy": best_val_acc,
    }
    args.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved ONNX model: {args.out}")
    print(f"Saved metadata: {args.meta}")


if __name__ == "__main__":
    main()
