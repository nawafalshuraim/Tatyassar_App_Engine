"""
Local LLM Loader (Phi-3 Mini)
-----------------------------
This module loads a local causal LLM (Phi-3 Mini) for:
    - generating self-edits
    - proposing synthetic training examples
    - helping SEAL refine the cue classifier

It does NOT answer user queries directly.
It is only used internally by the system.
"""

from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PHI3_MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"


def get_device() -> torch.device:
    """
    Prefer Apple Silicon GPU (MPS) if available, else CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LocalEditModel:
    """
    Wrapper around Phi-3 Mini for generating SEAL-style self-edits.

    Usage:
        llm = LocalEditModel()
        text = llm.generate_edit("your prompt here")
    """

    def __init__(
        self,
        model_name: str = PHI3_MODEL_NAME,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        self.device = get_device()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"[LLM] Loading {model_name} on {self.device} ...")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model – we keep it in 16-bit to save memory on MPS
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
            device_map=None,  # we move manually to device
        )

        self.model.to(self.device)
        self.model.eval()
        print("[LLM] Loaded successfully.")

    def _build_prompt(self, system_msg: str, user_msg: str) -> str:
        """
        Simple chat-style prompt for Phi-3 Mini.
        We keep this minimal; later we can make it SEAL-specific.
        """
        # Many instruct models work well with this simple formatting:
        # <|system|> ... <|user|> ... <|assistant|>
        prompt = (
            "<|system|>\n"
            f"{system_msg}\n"
            "<|user|>\n"
            f"{user_msg}\n"
            "<|assistant|>\n"
        )
        return prompt

    def generate_edit(
        self,
        system_msg: str,
        user_msg: str,
        stop_tokens: List[str] | None = None,
    ) -> str:
        """
        Generate a self-edit / suggestion from Phi-3 Mini.

        system_msg: high-level instruction (e.g., "You are an expert...")
        user_msg:   concrete content (example text, predictions, labels...)
        """
        if stop_tokens is None:
            stop_tokens = ["}", "<|end|>"]


        prompt = self._build_prompt(system_msg, user_msg)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False
        )

        # We only want the assistant part (after our prompt)
        generated = full_text[len(prompt):]

        # Stop at first stop token if present
        for t in stop_tokens:
            if t in generated:
                generated = generated.split(t)[0]
                break

        return generated.strip()


if __name__ == "__main__":
    # Quick test
    llm = LocalEditModel()
    system = (
        "You are an expert therapist + ML engineer.\n"
        "Given a patient text, true cues, and model-predicted cues, "
        "you suggest ONE improved training example in JSON."
    )
    user = (
    "text: \"i faild and exam i wanna drop out college\"\n"
    "true_cues: [\"self-blame\", \"withdrawal\"]\n"
    "model_predicted_cues: [\"avoidance\", \"withdrawal\", \"self-blame\"]\n\n"
    "Return a JSON object with fields: {\"improved_text\": ..., \"correct_cues\": [...]}."
)


    out = llm.generate_edit(system, user)
    print("\n[TEST OUTPUT]\n", out)
