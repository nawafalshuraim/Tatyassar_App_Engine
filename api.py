# api.py — FINAL PRODUCTION VERSION for IDSS Chatbot (with improved crisis handling)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json
import torch

# =============================
# INTERNAL IMPORTS
# =============================

from chatbot.preprocessor import NLPModel, detect_crisis
from chatbot.cue_classifier_model import CueClassifier
from chatbot.local_llm import LocalEditModel
from chatbot.narrative_generator import generate_final_narrative
from chatbot.chatbot import paraphrase_with_guard
from chatbot.input_validator import is_valid_input
from chatbot.knowledge_base import load_kb

load_kb()


# =============================
# FASTAPI APP + CORS
# =============================

app = FastAPI(title="IDSS Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================
# LOAD MODELS ONCE
# =============================

root = Path(__file__).resolve().parents[0]

# Cue labels
vocab_path = root / "models" / "cue_classifier" / "cue_vocab.json"
with vocab_path.open("r", encoding="utf-8") as f:
    cues = json.load(f)["cues"]

# Thresholds
thresholds_path = root / "models" / "cue_classifier" / "thresholds.json"
with thresholds_path.open("r", encoding="utf-8") as f:
    thresholds = json.load(f).get("thresholds", [0.55] * len(cues))

# Encoder
nlp = NLPModel()

# Local LLM
llm = LocalEditModel()

# Classifier
model_path = root / "models" / "cue_classifier" / "model.pt"
model = CueClassifier(input_dim=384, num_labels=len(cues))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


# =============================
# REQUEST MODEL
# =============================

class ChatRequest(BaseModel):
    message: str


# =============================
# MAIN ENDPOINT
# =============================

@app.post("/chat")
def chat(req: ChatRequest):

    user_text = req.message.strip()

    # 1. Input validation
    valid, err = is_valid_input(user_text)
    if not valid:
        return {"reply": err}

    # 2. CRISIS DETECTION (HIGHEST PRIORITY)
    if detect_crisis(user_text):
        return {
            "reply": (
                "I'm really glad you told me this. What you're describing sounds very serious, "
                "and you deserve support from someone who can be with you in person right now.\n\n"
                "I’m not able to provide emergency help, but please reach out immediately to:\n"
                "- Your local emergency number\n"
                "- A trusted friend, family member, or counselor\n\n"
                "You do not have to go through this alone — someone can help you right now."
            ),
            "is_crisis": True
        }

    # 3. Paraphrase safely
    paraphrased = paraphrase_with_guard(llm, user_text)

    # 4. Embeddings
    emb = nlp.encode(paraphrased).unsqueeze(0)

    # 5. Classification
    with torch.no_grad():
        probs = torch.sigmoid(model(emb))[0]

    predicted_cues = [
        cue for cue, p, th in zip(cues, probs, thresholds)
        if p.item() >= th
    ]

    # 6. If no emotional cues detected
    if not predicted_cues:
        return {
            "reply": (
                "It sounds like something important is weighing on you. "
                "I’m here — tell me a little more about what feels heavy."
            ),
            "predicted_cues": [],
            "paraphrased_input": paraphrased
        }

    # 7. Build therapeutic narrative
    final_message = generate_final_narrative(predicted_cues)

    # 8. Final response for Flutter
    return {
        "reply": final_message,
        "predicted_cues": predicted_cues,
        "paraphrased_input": paraphrased,
        "is_crisis": False
    }
