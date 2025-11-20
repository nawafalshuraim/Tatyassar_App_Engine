"""
 NLP ENGINE 
------------------------
This module provides:
    text cleaning
    sentence splitting
    transformer embeddings
    semantic similarity
    semantic emotion vectors (learnable)
    sentence/document encoding
    NO lexicons
    NO keyword matching
    fully embedding-based semantic NLP

"""

import re
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel


# -------------------------------------------------------
# DEVICE: On Apple Silicon (M1, M2, M3, M4), PyTorch can use the Apple GPU
# This makes NLP/ML much faster than CPU
#M PS = “Apple’s version of CUDA”
# -------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------------------------------
# CLEANING
# -------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalizes therapy text for semantic processing.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text


# -------------------------------------------------------
# SENTENCE SPLITTING
# -------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """
    Splits therapy dialogue into meaningful sentences.
    """
    text = clean_text(text)
    parts = re.split(
        r'(?<=[\.\!\?])\s+|(?=Therapist:)|(?=Client:)',
        text
    )
    return [p.strip() for p in parts if len(p.strip()) > 1]


# -------------------------------------------------------
# TRANSFORMER MODEL
# -------------------------------------------------------

class NLPModel:
    """
    COMPLETE semantic model for embeddings & NLP tasks.
    - Not generative (understanding-only)
    - Produces embeddings capturing meaning
    - Works on CPU/MPS
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.device = get_device()
        print(f"[NLP] Loading model: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    # ---------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        """
        Full document embedding (mean pooled).
        """
        text = clean_text(text)
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


# -------------------------------------------------------
# SEMANTIC SIMILARITY
# -------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# -------------------------------------------------------
# SEMANTIC EMOTION VECTORS (NO LEXICON)
# -------------------------------------------------------

"""""
Later:
- dataset expands them
- classifier expands them
- SEAL evolves them

so we NEVER modify these manually again.
"""

BASE_EMOTIONS = {
    "fear": "fear",
    "sadness": "sad",
    "anger": "anger",
    "shame": "shame",
    "anxiety": "anxiety",
    "dissociation": "dissociation",
    "withdrawal": "withdrawal"
}

def generate_emotion_vectors(nlp: NLPModel) -> Dict[str, torch.Tensor]:
    vectors = {}
    for emotion, seed_word in BASE_EMOTIONS.items():
        vectors[emotion] = nlp.encode(seed_word)
    return vectors


def semantic_emotion_scores(text: str, nlp: NLPModel) -> Dict[str, float]:
    """
    Computes semantic similarity between text and emotion seed vectors.
    """
    text_emb = nlp.encode(text)
    emotion_vectors = generate_emotion_vectors(nlp)

    scores = {}
    for emotion, vector in emotion_vectors.items():
        scores[emotion] = round(cosine_similarity(text_emb, vector), 4)

    return scores


# -------------------------------------------------------
# TEST
# -------------------------------------------------------

if __name__ == "__main__":
    nlp = NLPModel()
    text = "I fee likee I'm disapponteed. I can't braathe. My heart is racig."

    print("\nCLEANED:")
    print(clean_text(text))

    print("\nSPLIT:")
    print(split_sentences(text))

    print("\nEMOTION SCORES:")
    print(semantic_emotion_scores(text, nlp))
