import re
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

# DEVICE SELECTION (Apple MPS preferred)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# CRISIS FILTERING (Handled before embedding/classification)
CRISIS_TERMS = [
    r"\bsuicide\b",
    r"\bkill my ?self\b",
    r"\bkill him\b",
    r"\bkill her\b",
    r"\bself[- ]?harm\b",
    r"\bcutting\b",
    r"\bi want to die\b",
    r"\bi don'?t want to live\b",
    r"\bno reason to live\b",
    r"\bend my life\b",
    r"\bhurt myself\b",
    r"\bnot wake up\b",
    r"\bnever wake up\b",
    r"\bwish i didn'?t exist\b",
    r"\bdisappear forever\b",
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

    # keep ASCII, Arabic characters, and safe emojis
    text = "".join(
        ch for ch in text
        if ch.isascii() or '\u0600' <= ch <= '\u06FF' or ch in SAFE_EMOJIS
    )

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
