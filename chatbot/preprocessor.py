
import re
from typing import List, Dict #for type hints
import torch
from transformers import AutoTokenizer, AutoModel #for MiniLM transformer

# DEVICE: On Apple Silicon (M1, M2, M3, M4), PyTorch can use the Apple GPU
# This makes NLP much faster than CPU
# MPS = “Apple’s version of CUDA”
def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# CLEANING
def clean_text(text: str) -> str:
    
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text


# SENTENCE SPLITTING FOR THE DATASET (LATER TO BE USED)
def split_sentences(text: str) -> List[str]:

    text = clean_text(text)
    parts = re.split(
        r'(?<=[\.\!\?])\s+|(?=Therapist:)|(?=Client:)',
        text
    )
    return [p.strip() for p in parts if len(p.strip()) > 1]


# TRANSFORMER MODEL (PRODUCES EMBEDDINGS CAPTURING MEANING/UNDERSTANDING ONLY/ NOT GENERATIVE/ ANSWER WHAT DOES THIS TEXT MEAN IN MATH FORM? OUTPUT: VECTOR OF 384 numbers)
class NLPModel:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.device = get_device()
        print(f"[NLP] Loading model: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    # EMBEDDINGS
    def encode(self, text: str) -> torch.Tensor:

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


# SEMANTIC SIMILARITY (1: same meaning, 0: different, -1: opposite)
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# SEMANTIC EMOTION VECTORS (NO LEXICON BUT SEMANTIC ANCHOR)
"""""
Later:
- dataset expands them
- classifier expands them
- SEAL evolves them
so we never modify these manually again.
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



