import torch
from pathlib import Path
import json

from chatbot.preprocessor import NLPModel
from chatbot.cue_classifier_model import CueClassifier


def load_vocab(vocab_path):
    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cues"]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    # MODEL + VOCAB PATHS
    model_path = root / "models" / "cue_classifier" / "model.pt"
    vocab_path = root / "models" / "cue_classifier" / "cue_vocab.json"

    # LOAD VOCAB
    cues = load_vocab(vocab_path)
    num_labels = len(cues)
    print("Loaded cue vocab:", cues)

    # LOAD NLP MODEL
    nlp = NLPModel()

    # LOAD CLASSIFIER
    input_dim = 384  # MiniLM embedding size
    model = CueClassifier(input_dim=input_dim, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("\nCue Classifier ready!")
    print("Type any text below and press Enter.\n")

    while True:
        text = input("Your text: ")

        if text.strip() == "":
            continue

        # Get embedding
        emb = nlp.encode(text).unsqueeze(0)  # shape [1, 384]

        # Predict
        logits = model(emb)
        probs = torch.sigmoid(logits)[0]

        print("\nProbabilities:")
        for cue, p in zip(cues, probs):
            print(f"{cue:20s}: {p.item():.4f}")

        print("\nPredicted cues (> 0.50):")
        found = False
        for cue, p in zip(cues, probs):
            if p.item() >= 0.50:
                print(" -", cue)
                found = True

        if not found:
            print("No cue above threshold.")

        print("\n" + "-" * 50 + "\n")
