import torch
from pathlib import Path
import json
from datetime import datetime
from chatbot.preprocessor import NLPModel
from chatbot.cue_classifier_model import CueClassifier
from chatbot.local_llm import LocalEditModel


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

    # LOAD LOCAL LLM (Phi-3)
    llm = LocalEditModel()

    print("\TRAUMA CHATBOT READY")
    print("Type a sentence (or 'exit'):\n")

    while True:
        user_text = input("You: ")

        if user_text.lower().strip() == "exit":
            break

        if user_text.strip() == "":
            continue

        print("\n[INPUT TEXT]")
        print(user_text)

        # Encode text
        emb = nlp.encode(user_text).unsqueeze(0)

        # Predict
        logits = model(emb)
        probs = torch.sigmoid(logits)[0]

        print("\n[PROBABILITIES]")
        predicted_cues = []

        for cue, p in zip(cues, probs):
            print(f"{cue:22s}: {p.item():.4f}")
            if p.item() >= 0.50:
                predicted_cues.append(cue)

        print("\n[PREDICTED CUES]")
        print(predicted_cues if predicted_cues else "None detected")

        print("\n[RUNNING PHI-3 SELF-EDIT (SEAL MODE)]\n")
        #instruction for Phi-3
        system = (
            "You are an expert therapist + ML engineer. "
            "Given a patient text, true cues, and model-predicted cues, "
            "return ONE improved training example in JSON."
        )

        user = f"""
text: "{user_text}"
true_cues: {predicted_cues}
model_predicted_cues: {predicted_cues}

Return a JSON object with fields:
{{"improved_text": "...", "correct_cues": [...]}}
"""

        out = llm.generate_edit(system, user)
        print(out)

        print("\n" + "-" * 60 + "\n")

# Save Phi-3’s output to dataset
try:
    new_example = json.loads(out)

    save_obj = {
        "id": f"seal_{datetime.now().timestamp()}",
        "input_text": new_example["improved_text"],
        "true_cues": new_example["correct_cues"]
    }

    with open("seal_generated_examples.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(save_obj) + "\n")

    print("SEAL example saved")

except Exception as e:
    print("Could not save SEAL output:", e)

