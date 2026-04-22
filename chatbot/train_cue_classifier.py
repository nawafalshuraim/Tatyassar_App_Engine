"""
This script:
    - loads challenges + solutions
    - optionally loads SEAL-generated data
    - builds cue vocab
    - encodes each example with NLPModel (MiniLM)
    - trains an MLP classifier on top of embeddings
    - tunes per-cue thresholds on validation set
    - saves model + vocab + thresholds to models/cue_classifier/
"""

from pathlib import Path
import json
from typing import Tuple, List
import csv
from datetime import datetime
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from chatbot.preprocessor import NLPModel
from chatbot.cue_data import build_examples, build_cue_vocab, CueVocab
from chatbot.cue_classifier_model import CueClassifier


# PATHS & LOGGING
project_root = Path(__file__).resolve().parents[1]

LOG_PATH = project_root / "models" / "training_logs.csv"


# create log file if not exists
if not LOG_PATH.exists():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "version",
            "timestamp",
            "mode",
            "num_samples",
            "final_train_loss",
            "final_val_loss"
        ])


# TRAINING MODE
TRAINING_MODE = "manual"
if len(sys.argv) > 1 and sys.argv[1] == "self":
    TRAINING_MODE = "self"

VERSION = datetime.now().strftime("v%Y%m%d_%H%M%S")


# DATA BUILDING
def build_tensors(
    project_root: Path,
    nlp: NLPModel,
) -> Tuple[torch.Tensor, torch.Tensor, CueVocab]:
    """
    Build X (embeddings) and Y (multi-label targets) from:
        - challenges.json
        - solutions.json
        - optional seal_generated_examples.json
    """
    challenges_path = project_root / "challenges_solutions_data" / "challenges.json"
    solutions_path = project_root / "challenges_solutions_data" / "solutions.json"
    seal_path = project_root / "seal_generated_examples.json"

    # 1) Build vocab from original dataset
    vocab = build_cue_vocab(solutions_path)

    # 2) Load base examples from ground-truth files
    examples, vocab = build_examples(challenges_path, solutions_path, vocab)

    # 3) Optionally load SEAL-generated data
    if seal_path.exists():
        print("[SEAL] Loading SEAL-generated examples...")
        added = 0

        with open(seal_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                # Respect "accepted" flag if present
                if data.get("accepted") is False:
                    continue

                # Pick best text field we have
                text = (
                    data.get("refined_text")
                    or data.get("input_text")
                    or data.get("text")
                )
                if not text:
                    continue

                # Cue field may be "true_cues" or "new_cues"
                cue_names = (
                    data.get("true_cues")
                    or data.get("new_cues")
                    or []
                )

                cue_indices = [
                    vocab.cue_to_id[cue]
                    for cue in cue_names
                    if cue in vocab.cue_to_id
                ]
                if not cue_indices:
                    continue

                # temp example object with same attributes as CueExample
                class TempExample:
                    pass

                ex = TempExample()
                ex.text = text
                ex.label_indices = cue_indices

                examples.append(ex)
                added += 1

        print(f"[SEAL] Added {added} new examples from SEAL file.\n")

    # 4) Convert examples to tensors
    X_list = []
    Y_list = []

    for ex in examples:
        emb = nlp.encode(ex.text)          # [D]
        X_list.append(emb.unsqueeze(0))    # [1, D]

        y = torch.zeros(vocab.size, dtype=torch.float32)
        for idx in ex.label_indices:
            y[idx] = 1.0
        Y_list.append(y.unsqueeze(0))      # [1, L]

    X = torch.cat(X_list, dim=0)  # [N, D]
    Y = torch.cat(Y_list, dim=0)  # [N, L]

    return X, Y, vocab


# METRICS & THRESHOLD TUNING
def compute_metrics_for_label(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float
) -> dict:
    """
    probs, targets: [N]
    threshold: scalar in [0, 1]
    """
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def tune_thresholds(
    model: CueClassifier,
    val_loader: DataLoader,
    device: torch.device,
    num_labels: int,
    search_min: float = 0.3,
    search_max: float = 0.9,
    steps: int = 13
) -> List[float]:
    """
    Tune per-label thresholds on the validation set
    by maximizing F1 score for each cue independently.
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = model.predict_proba(logits)
            all_probs.append(probs.cpu())
            all_targets.append(yb.cpu())

    all_probs = torch.cat(all_probs, dim=0)     # [N, L]
    all_targets = torch.cat(all_targets, dim=0) # [N, L]

    thresholds = [0.5] * num_labels
    threshold_grid = torch.linspace(search_min, search_max, steps=steps)

    print("\n[THRESHOLD TUNING]")
    for label_idx in range(num_labels):
        probs_l = all_probs[:, label_idx]
        targets_l = all_targets[:, label_idx]

        if targets_l.sum().item() == 0:
            # no positives at all -> keep default 0.5
            print(f"  Label {label_idx}: no positive examples in val set, using 0.5")
            thresholds[label_idx] = 0.5
            continue

        best_f1 = 0.0
        best_th = 0.5

        for th in threshold_grid:
            m = compute_metrics_for_label(probs_l, targets_l, float(th))
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_th = float(th)

        thresholds[label_idx] = best_th
        print(f"  Label {label_idx}: best_th={best_th:.2f}, F1={best_f1:.3f}")

    print()
    return thresholds


# TRAINING
def train(
    project_root: Path,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 1e-3,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    final_train_loss = None
    final_val_loss = None

    # 1) NLP model
    nlp = NLPModel()

    # 2) Build tensors
    X, Y, vocab = build_tensors(project_root, nlp)

    input_dim = X.shape[1]
    num_labels = Y.shape[1]

    print(f"[DATA] {X.shape[0]} total examples")
    print(f"       Embedding dim = {input_dim}")
    print(f"       Number of cues = {num_labels}\n")

    # 3) Train/Val split
    dataset = TensorDataset(X, Y)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 4) Model
    model = CueClassifier(input_dim=input_dim, num_labels=num_labels).to(device)

    # Handle class imbalance: compute positive counts per label
    label_counts = Y.sum(dim=0)  # [L]
    # pos_weight formula: (N - pos) / (pos + 1e-6) to give more weight to rare cues
    pos_weight = (Y.shape[0] - label_counts) / (label_counts + 1e-6)
    pos_weight = pos_weight.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 5) Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / train_size

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)

        avg_val_loss = val_loss / val_size

        print(f"[EPOCH {epoch}] Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}")

        final_train_loss = avg_train_loss
        final_val_loss = avg_val_loss

    # 6) Threshold tuning
    thresholds = tune_thresholds(model, val_loader, device, num_labels=num_labels)

    # 7) Save model, vocab, thresholds
    out_dir = project_root / "models" / "cue_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    vocab_path = out_dir / "cue_vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({"cues": vocab.cues}, f, ensure_ascii=False, indent=2)
    print(f"Vocab saved to {vocab_path}")

    thresholds_path = out_dir / "thresholds.json"
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump({"thresholds": thresholds}, f, indent=2)
    print(f"Thresholds saved to {thresholds_path}")

    # 8) Log training summary
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            VERSION,
            datetime.now().isoformat(),
            TRAINING_MODE,
            X.shape[0],
            final_train_loss,
            final_val_loss
        ])

    print(f"\nTraining logged to: {LOG_PATH}")


# ENTRY POINT
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    train(root)
