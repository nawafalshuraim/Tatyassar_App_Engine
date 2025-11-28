import json
import re
from datetime import datetime
from pathlib import Path
import subprocess
import torch
from chatbot.preprocessor import NLPModel
from chatbot.cue_classifier_model import CueClassifier
from chatbot.local_llm import LocalEditModel
from chatbot.input_validator import is_valid_input


# HELPERS
def load_vocab(vocab_path: Path):
    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cues"]


def count_seal_examples(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


SEAL_TRIGGER = 10   # retrain after 10 examples


# MAIN
if __name__ == "__main__":

    root = Path(__file__).resolve().parents[1]

    model_path = root / "models" / "cue_classifier" / "model.pt"
    vocab_path = root / "models" / "cue_classifier" / "cue_vocab.json"
    seal_output_path = root / "seal_generated_examples.jsonl"

    # 1) Load vocab
    cues = load_vocab(vocab_path)
    print("Loaded cue vocab:", cues)

    # 2) Load NLP model
    nlp = NLPModel()

    # 3) Load classifier
    model = CueClassifier(input_dim=384, num_labels=len(cues))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 4) Load Phi-3
    llm = LocalEditModel()

    PARAPHRASE_PROMPT = (
        "You are an expert at understanding messy, emotional, and misspelled text.\n"
        "Rewrite the user's message in clear, correct English.\n"
        "Write it in the SECOND PERSON (use 'you' instead of 'I').\n"
        "Return ONLY the corrected sentence. No explanations."
    )

    SEAL_PROMPT = f"""
You are a strict data-cleaning engine.

Return ONLY one valid JSON object and NOTHING else:

{{"improved_text":"...","correct_cues":[...] }}

RULES:
- improved_text must be ONE short sentence ONLY
- NO advice
- NO empathy
- NO suggestions
- NO multi-sentence answers
- Just a corrected version of the user text

You may ONLY use the following cues:
{cues}
"""

    print("\nCHATBOT READY")
    print("Type a sentence (or 'exit'):\n")

    # INTERACTIVE LOOP
    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in ["exit", "quit"]:
            print("\nGoodbye")
            break

        if not user_text:
            continue

        # 1) INPUT VALIDATION
        valid, message = is_valid_input(user_text)
        if not valid:
            print(f"\nInvalid input: {message}\n")
            continue

        # 2) PARAPHRASE & CONFIRM
        print("\nInterpreting your message...\n")

        fixed_text = llm.generate_edit(PARAPHRASE_PROMPT, user_text)

        fixed_text = fixed_text.split("\n")[0]
        fixed_text = re.sub(r"##.*", "", fixed_text)
        fixed_text = fixed_text.strip().strip('"').strip("'")

        print("So what’s happening is that:")
        print(f'   "{fixed_text}"')

        confirm = input("\nIs this correct? (yes / no): ").strip().lower()
        if confirm not in ["yes", "y"]:
            print("\nOkay, please re-enter your sentence.\n")
            continue

        user_text = fixed_text

        # 3) EMBEDDING + CLASSIFICATION
        with torch.no_grad():
            emb = nlp.encode(user_text).unsqueeze(0)
            logits = model(emb)
            probs = torch.sigmoid(logits)[0]

        print("\n[PROBABILITIES]")
        for cue, p in zip(cues, probs):
            print(f"{cue:22s}: {p.item():.4f}")

        predicted_cues = [cue for cue, p in zip(cues, probs) if p.item() >= 0.50]

        print("\n[PREDICTED CUES]")
        print(predicted_cues if predicted_cues else "None above threshold")

        # 4) SEAL SELF-EDIT
        print("\n[RUNNING PHI-3 SELF-EDIT (SEAL MODE)]\n")

        user_prompt = f"""
text: "{user_text}"
model_predicted_cues: {predicted_cues}
allowed_cues: {cues}

Return ONLY:
{{"improved_text": "...", "correct_cues": [...]}}
"""

        out = llm.generate_edit(SEAL_PROMPT, user_prompt).strip()

        if not out.startswith("{"):
            out = "{" + out
        if not out.endswith("}"):
            out = out + "}"

        print("\n[PHI-3 OUTPUT]\n", out)

        # 5) SAVE SEAL DATA — SAFE VERSION
        try:
            # Make sure file + folder exist
            seal_output_path.parent.mkdir(parents=True, exist_ok=True)
            if not seal_output_path.exists():
                seal_output_path.touch()

            # Extract JSON from Phi output
            match = re.search(r"\{[\s\S]*\}", out)
            if not match:
                raise ValueError("No valid JSON object found in Phi output")

            clean_json = match.group(0).strip()

            # AUTO-FIX broken JSON (missing first quote)
            if not clean_json.startswith('{"'):
                clean_json = re.sub(r'^\{([^"])', r'{"\1', clean_json)

            # Remove newlines that break JSON
            clean_json = clean_json.replace("\n", " ").strip()
            
            # FIX SMART QUOTES 
            clean_json = clean_json.replace("“", '"').replace("”", '"').replace("’", "'")

            # Load JSON safely
            print("\n[FINAL CLEAN JSON]\n", clean_json)
            new_example = json.loads(clean_json)

            # ---- VALIDATE TEXT ----
            improved_text = new_example.get("improved_text", "").strip()

            if not improved_text:
                raise ValueError("Empty improved_text, skip save")

            if improved_text.count(".") > 1:
                raise ValueError("Too many sentences — looks like advice")

            # ---- FILTER CUES ----
            raw_cues = new_example.get("correct_cues", [])
            valid_cues = [c for c in raw_cues if c in cues]

            # Fallback to classifier
            if not valid_cues and predicted_cues:
                valid_cues = predicted_cues

            if not valid_cues:
                raise ValueError("No valid cues found")

            # ---- FINAL OBJECT ----
            save_obj = {
                "id": f"seal_{datetime.now().timestamp()}",
                "input_text": improved_text,
                "true_cues": valid_cues
            }

            # SAVE — JSONL
            with seal_output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(save_obj, ensure_ascii=False) + "\n")

            print("\nSEAL example saved:")
            print(save_obj)

        except Exception as e:
            print("\nCould not parse/save SEAL output:")
            print(e)
            
            # 6) AUTO-SEAL RETRAIN
        seal_count = count_seal_examples(seal_output_path)
        print(f"\n[SEAL DATA COUNT]: {seal_count} / {SEAL_TRIGGER}")

        if seal_count >= SEAL_TRIGGER:
            print("\n AUTO-SEAL TRIGGERED — Retraining model...\n")

            subprocess.run(
                ["python", "-m", "chatbot.train_cue_classifier"],
                check=True
            )

            print("\nNew model trained using SEAL data")

            # Clear file after retraining (start fresh batch)
            seal_output_path.open("w", encoding="utf-8").close()
            print("SEAL data file cleared\n")

        print("\n" + "="*60 + "\n")
    
            

