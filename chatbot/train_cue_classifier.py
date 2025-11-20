"""
Train the cue classifier (MLP over MiniLM embeddings)

This will:
    - load challenges + solutions
    - build cue vocab
    - encode each example with NLPModel (MiniLM)
    - train an MLP classifier on top of embeddings
    - save model + vocab to models/cue_classifier/
"""

from pathlib import Path
import json
from typing import Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from chatbot.preprocessor import NLPModel
from chatbot.cue_data import build_examples, build_cue_vocab, CueVocab
from chatbot.cue_classifier_model import CueClassifier


def build_tensors(
    project_root: Path,
    nlp: NLPModel,
) -> Tuple[torch.Tensor, torch.Tensor, CueVocab]:
    """
    Encodes all examples to embeddings and builds label tensors.
    Returns:
        X: [N, D] embeddings
        Y: [N, L] multi-label targets
        vocab: CueVocab
    """
    challenges_path = project_root / "challenges_solutions_data" / "challenges.json"
    solutions_path = project_root / "challenges_solutions_data" / "solutions.json"

    vocab = build_cue_vocab(solutions_path)
    examples, vocab = build_examples(challenges_path, solutions_path, vocab)

    X_list = []
    Y_list = []

    for ex in examples:
        emb = nlp.encode(ex.text)  # [D]
        X_list.append(emb.unsqueeze(0))  # [1, D]

        y = torch.zeros(vocab.size, dtype=torch.float32)
        for idx in ex.label_indices:
            y[idx] = 1.0
        Y_list.append(y.unsqueeze(0))  # [1, L]

    X = torch.cat(X_list, dim=0)  # [N, D]
    Y = torch.cat(Y_list, dim=0)  # [N, L]

    return X, Y, vocab


def train(
    project_root: Path,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 1e-3,
):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[TRAIN] Using device: {device}")

    # 1) NLP model for embeddings
    nlp = NLPModel()

    # 2) Build tensors
    X, Y, vocab = build_tensors(project_root, nlp)
    input_dim = X.shape[1]
    num_labels = Y.shape[1]

    print(f"[DATA] {X.shape[0]} examples, embedding dim = {input_dim}, num cues = {num_labels}")

    # 3) Train/val split (90/10)
    dataset = TensorDataset(X, Y)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 4) Model
    model = CueClassifier(input_dim=input_dim, num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 5) Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = model.loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / train_size

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = model.loss_fn(logits, yb)
                val_loss += loss.item() * xb.size(0)

        avg_val_loss = val_loss / val_size
        print(f"[EPOCH {epoch}] train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

    # 6) Save model + vocab
    out_dir = project_root / "models" / "cue_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[SAVE] Model saved to {model_path}")

    vocab_path = out_dir / "cue_vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({"cues": vocab.cues}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Vocab saved to {vocab_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    train(root)
