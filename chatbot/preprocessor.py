import re
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel

# DEVICE SELECTION (Apple MPS preferred)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# CRISIS FILTERING (Handled before embedding/classification)
CRISIS_TERMS = [
    r"\bsuicide\b", r"\bkill myself\b", r"\bkill him\b", r"\bkill her\b",
    r"\bself harm\b", r"\bself-harm\b", r"\bcutting\b", r"\bI want to die\b",
    r"\bI don'?t want to live\b"
]

def detect_crisis(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(re.search(pattern, text) for pattern in CRISIS_TERMS)


# TEXT CLEANING (keeps emotional emojis, strips junk)
SAFE_EMOJIS = ["😢","😭","😔","😞","😟","😡","😠","😨","😖","😣","😩"]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # keep safe emojis, remove others
    text = "".join(ch for ch in text if ch.isascii() or ch in SAFE_EMOJIS)

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# SENTENCE SPLITTING
def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    parts = re.split(
        r'(?<=[\.\!\?])\s+|(?=Therapist:)|(?=Client:)', text
    )
    return [p.strip() for p in parts if len(p.strip()) > 1]


# MINI LM EMBEDDINGS
class NLPModel:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.device = get_device()
        print(f"[NLP] Loading model: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, text: str) -> torch.Tensor:

        text = clean_text(text)

        # handle blank or crisis before embedding
        if text == "" or detect_crisis(text):
            return torch.zeros(384)  # static fallback vector

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**tokens)
            emb = out.last_hidden_state.mean(dim=1)

        return emb.cpu().squeeze(0)

    def encode_sentences(self, sentences: List[str]) -> torch.Tensor:
        return torch.stack([self.encode(s) for s in sentences])


# SEMANTIC EMOTION VECTORS (CACHED)
BASE_EMOTIONS = {
    "fear": "fear",
    "sadness": "sad",
    "anger": "anger",
    "shame": "shame",
    "anxiety": "anxiety",
    "dissociation": "dissociation",
    "withdrawal": "withdrawal"
}

_cached_vectors = None

def get_emotion_vectors(nlp: NLPModel) -> Dict[str, torch.Tensor]:
    global _cached_vectors
    if _cached_vectors is None:
        _cached_vectors = {e: nlp.encode(w) for e, w in BASE_EMOTIONS.items()}
    return _cached_vectors

def semantic_emotion_scores(text: str, nlp: NLPModel) -> Dict[str, float]:
    text_emb = nlp.encode(text)
    emotion_vectors = get_emotion_vectors(nlp)

    scores = {}
    for emotion, vector in emotion_vectors.items():
        sim = torch.nn.functional.cosine_similarity(text_emb, vector, dim=0).item()
        scores[emotion] = round(sim, 4)

    return scores
