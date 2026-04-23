"""
Local LLM Loader (Phi-3 Mini) with Safe SEAL JSON Editing
---------------------------------------------------------
Adds:
    - Crisis bypass
    - Safe JSON extraction + schema enforcement
    - Required fields: refined_text, new_cues, confidence
    - TRUE SINGLETON → prevents reloading Phi-3 every request
"""

import json
import re
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chatbot.preprocessor import detect_crisis, clean_text

PHI3_MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"


# ============================
# DEVICE SELECTION
# ============================

def get_device() -> torch.device:
    """Prefer MPS on Mac; fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================
# SAFE JSON HELPERS
# ============================

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first valid JSON object from the text.
    If invalid → return empty {} → SEAL automatically rejects.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def normalize_seal_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize SEAL output fields:
       refined_text: str
       new_cues: list
       confidence: float
    Reject unsafe or crisis-altering edits.
    """
    refined = raw.get("refined_text")
    cues = raw.get("new_cues")
    conf = raw.get("confidence")

    # MUST contain refined_text + list of cues
    if not refined or not isinstance(cues, list):
        return {}

    # Clean + safety check
    refined = clean_text(refined)
    if detect_crisis(refined):
        return {}  # never allow model to rewrite crisis content

    # Normalize confidence
    if isinstance(conf, (float, int)):
        conf = max(0.0, min(float(conf), 1.0))
    else:
        conf = None

    return {
        "refined_text": refined,
        "new_cues": sorted(set(cues)),
        "confidence": conf,
    }


# ============================
# LOCAL LLM WRAPPER (SINGLETON)
# ============================

class LocalEditModel:
    """
    Phi-3 text editor for:
        • paraphrasing (generate_edit)
        • SEAL JSON refinement (seal_edit)

    NOTE: Enforced as a TRUE SINGLETON → model loads ONCE.
    """

    _instance: "LocalEditModel" = None

    # --------- SINGLETON CONSTRUCTOR --------- #
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LocalEditModel, cls).__new__(cls)
        return cls._instance

    # --------- INIT (runs once) --------- #
    def __init__(
        self,
        model_name: str = PHI3_MODEL_NAME,
        max_new_tokens: int = 200,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        # Prevent re-initializing if singleton already exists
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.device = get_device()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"[LLM] Loading {model_name} on {self.device} ...")

        # Load tokenizer + model once
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
        ).to(self.device)
        self.model.eval()

        print("[LLM] Loaded successfully.")

        self._initialized = True  # lock init

    # --------- PROMPT ASSEMBLY --------- #
    def _build_prompt(self, system_msg: str, user_msg: str) -> str:
        """
        Build Phi-3 Instruct-style prompt.
        """
        return (
            "<|system|>\n"
            f"{system_msg.strip()}\n"
            "<|user|>\n"
            f"{user_msg.strip()}\n"
            "<|assistant|>\n"
        )

    # --------- RAW GENERATION --------- #
    def _raw_llm_call(self, system_msg: str, user_msg: str) -> str:
        """
        Execute Phi-3 generation safely with correct decoding.
        """
        prompt = self._build_prompt(system_msg, user_msg)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # --------- SEAL EDIT (JSON) --------- #
    def seal_edit(self, system_msg: str, user_msg: str) -> Dict[str, Any]:
        """
        SEAL: Self-editing with strict safety rules.
        Returns normalized dict or {} if invalid.
        """
        # Never rewrite crisis text
        if detect_crisis(user_msg):
            return {}

        full = self._raw_llm_call(system_msg, user_msg)
        generated = full.split("<|assistant|>")[-1].strip()

        raw = extract_json(generated)
        return normalize_seal_output(raw)

    # --------- TEXT REWRITE (NON-JSON) --------- #
    def generate_edit(self, system_msg: str, user_msg: str) -> str:
        """
        LLM rewrite → plain text string only.
        Used for paraphrasing & humanization.
        """
        full = self._raw_llm_call(system_msg, user_msg)
        text = full.split("<|assistant|>")[-1]

        # Strip Phi-3 special tokens and trailing whitespace
        text = re.sub(r"<\|.*?\|>", "", text).strip()

        return text or user_msg
