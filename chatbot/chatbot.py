# chatbot/chatbot.py  (v4 – Therapy + Vent + Coach)

import json
import re
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Optional, Dict, Any, List

import torch

from chatbot.preprocessor import NLPModel, detect_crisis
from chatbot.cue_classifier_model import CueClassifier
from chatbot.local_llm import LocalEditModel
from chatbot.input_validator import is_valid_input
from chatbot.recommendation_engine import build_recommendation
from chatbot.narrative_generator import generate_final_narrative
from chatbot.knowledge_base import load_kb


# ==============================
# CONFIG & CONSTANTS
# ==============================

MEMORY_PATH = Path("conversation_memory.jsonl")
SEAL_TRIGGER = 10

DEFAULT_CLASSIFIER_THRESHOLD = 0.55         # fallback if thresholds.json missing
FOLLOWUP_SIM_THRESHOLD = 0.55               # semantic similarity for follow-up
SHORT_FOLLOWUP_MAX_TOKENS = 6               # short “yes/ok/etc” replies

VENT_KEYWORDS = [
    "just listen", "no advice", "don't give advice", "dont give advice",
    "i don't want advice", "i dont want advice", "just hear me",
    "just need to vent", "let me vent"
]

COACH_KEYWORDS = [
    "what should i do", "what should i do?", "give me steps",
    "how can i fix", "how do i fix", "help me plan", "give me a plan",
    "what can i do", "how do i start"
]


# ==============================
# MEMORY HANDLING
# ==============================

def load_memory():
    if not MEMORY_PATH.exists():
        return []
    try:
        with MEMORY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def save_memory(history):
    with MEMORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_memory_entry(history, role, text, paraphrased=None,
                     predicted_cues=None, bot_mode=None):
    history.append({
        "role": role,
        "text": text,
        "paraphrased": paraphrased,
        "predicted_cues": predicted_cues,
        "bot_mode": bot_mode,
        "timestamp": datetime.now().isoformat()
    })
    save_memory(history)


def get_last_user_message(history):
    for item in reversed(history):
        if item.get("role") == "user":
            return item
    return None


def get_last_bot_message(history):
    for item in reversed(history):
        if item.get("role") == "bot":
            return item
    return None


# ==============================
# SUPPORT HELPERS
# ==============================

def load_vocab(vocab_path: Path) -> List[str]:
    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cues"]


def load_thresholds(thresholds_path: Path, num_labels: int) -> List[float]:
    thresholds = [DEFAULT_CLASSIFIER_THRESHOLD] * num_labels
    if thresholds_path.exists():
        try:
            with thresholds_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = data.get("thresholds")
            if isinstance(loaded, list) and len(loaded) == num_labels:
                thresholds = [float(x) for x in loaded]
        except Exception:
            pass
    return thresholds


def count_seal_examples(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def is_semantic_followup(current_text: str, last_text: str,
                         nlp: NLPModel, threshold: float = FOLLOWUP_SIM_THRESHOLD):
    emb1 = nlp.encode(current_text)
    emb2 = nlp.encode(last_text)
    sim = cosine_sim(emb1, emb2)
    return sim >= threshold, sim


def is_acknowledgement(text: str) -> bool:
    t = text.strip().lower()
    if t in {"yes", "y", "yeah", "yep", "ok", "okay", "k",
             "sure", "right", "true", "exactly", "thanks", "thank you"}:
        return True

    words = t.split()
    if len(words) <= 3 and any(w in t for w in
                               ["yes", "ok", "okay", "thanks", "right", "sure", "true"]):
        return True

    return False


def handle_crisis():
    print("\n===== CRISIS RESPONSE =====\n")
    print(
        "I'm really glad you told me this. What you're describing sounds very serious,\n"
        "and you deserve support from someone who can be with you in real life.\n\n"
        "I’m not a replacement for professional help or emergency services.\n"
        "If you are in immediate danger, please contact your local emergency number\n"
        "or a trusted person nearby right now.\n\n"
        "If you can, consider reaching out to a mental health professional, a doctor,\n"
        "or a trusted person in your life and let them know how you’re feeling."
    )
    print("\n============================\n")


# ==============================
# MODE HANDLING
# ==============================

def detect_mode_switch(text: str) -> Optional[str]:
    """
    Detects if the user wants to switch between:
      - therapy
      - vent
      - coach
    Returns new_mode or None.
    """
    t = text.strip().lower()

    # Explicit commands
    if t.startswith("/therapy"):
        return "therapy"
    if t.startswith("/vent"):
        return "vent"
    if t.startswith("/coach"):
        return "coach"

    # Natural-language vent triggers
    if any(k in t for k in VENT_KEYWORDS):
        return "vent"

    # Natural-language coach triggers
    if any(k in t for k in COACH_KEYWORDS):
        return "coach"

    return None


# ==============================
# LLM PROMPTS
# ==============================

PARAPHRASE_PROMPT = """
You are a STRICT literal paraphrasing engine.

Your job is to restate the EXACT MEANING of the user's message
WITHOUT CHANGING, SOFTENING, IMPROVING, OR REINTERPRETING IT.

ABSOLUTE RULES:
- Do NOT remove, reduce, or alter any mention of self-harm, suicide, harm to others, or crisis language.
- NO positive reinterpretation.
- NO emotional reframing.
- NO advice.
- NO changing intent.
- Keep SECOND PERSON ("you") if the user used "I".
- Keep the emotional intensity exactly the same.
- Preserve dangerous content exactly.
- Only restate for clarity (misspellings, grammar).
- Output ONE short neutral sentence.

Now paraphrase the user's message literally:
"""


def paraphrase_with_guard(llm: LocalEditModel, text: str) -> str:
    raw = llm.generate_edit(PARAPHRASE_PROMPT, text)
    fixed = raw.split("\n")[0]
    fixed = re.sub(r"^#+\s*", "", fixed)
    fixed = fixed.strip().strip('"').strip("'")

    if fixed.lower().startswith("ou "):  # 'ou ' bug
        fixed = "Y" + fixed

    if not fixed:
        fixed = text.strip()

    return fixed


# ==============================
# SEAL HELPERS
# ==============================

def save_seal_example(
    path: Path,
    improved_text: str,
    cues: List[str],
    confidence: Optional[float] = None,
) -> None:
    obj = {
        "id": f"seal_{datetime.now().timestamp()}",
        "input_text": improved_text,
        "true_cues": cues,
    }
    if confidence is not None:
        obj["confidence"] = confidence
    obj["accepted"] = True

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print("\nSEAL example saved:", obj)


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    root = Path(__file__).resolve().parents[1]

    conversation_history = load_memory()
    empty_streak = 0
    mode = "therapy"  # "therapy" | "vent" | "coach"

    model_dir = root / "models" / "cue_classifier"
    model_path = model_dir / "model.pt"
    vocab_path = model_dir / "cue_vocab.json"
    thresholds_path = model_dir / "thresholds.json"

    seal_output_path = root / "seal_generated_examples.json"

    load_kb()
    cues = load_vocab(vocab_path)

    nlp = NLPModel()
    model = CueClassifier(input_dim=384, num_labels=len(cues))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    thresholds = load_thresholds(thresholds_path, num_labels=len(cues))

    llm = LocalEditModel()

    print("\nCHATBOT READY (v4)  —  Mode: THERAPY (default)")
    print("Commands: /therapy, /vent, /coach")
    print("Type a sentence (or 'exit'):\n")

    last_predicted_cues = None

    while True:
        user_text_raw = input(f"[{mode.upper()}] You: ").strip()

        # Exit
        if user_text_raw.lower() in ["exit", "quit"]:
            print("\nGoodbye 👋")
            break

        # Double-Enter → reset conversation
        if not user_text_raw:
            empty_streak += 1
            if empty_streak >= 2:
                print("\nNew conversation started.\n")
                conversation_history = []
                save_memory(conversation_history)
                empty_streak = 0
                last_predicted_cues = None
                continue
            else:
                print("\n(Press Enter again to reset conversation...)\n")
                continue
        else:
            empty_streak = 0

        # Check for mode switch (before anything else)
        new_mode = detect_mode_switch(user_text_raw)
        if new_mode and new_mode != mode:
            mode = new_mode
            print(f"\nMode switched to: {mode.upper()}\n")
            if mode == "vent":
                print("Okay, I'll focus on listening and understanding, and I won't give advice unless you ask.\n")
            elif mode == "coach":
                print("Alright, I'll help you think in terms of small, practical steps.\n")
            else:
                print("Back to therapy mode: validation + gentle strategies.\n")
            # Don't treat this line as emotional content
            continue

        # Crisis check (BEFORE validation/paraphrase)
        if detect_crisis(user_text_raw):
            add_memory_entry(conversation_history, "user", user_text_raw)
            handle_crisis()
            continue

        # Input validation
        valid, msg = is_valid_input(user_text_raw)
        if not valid:
            print(f"\nInvalid input: {msg}\n")
            continue

        # Acknowledgement-only reply?
        last_bot = get_last_bot_message(conversation_history)
        if is_acknowledgement(user_text_raw) and last_bot and last_bot.get("bot_mode") == "therapy":
            add_memory_entry(conversation_history, "user", user_text_raw)
            print("\n(Short acknowledgement detected — not redoing analysis.)\n")
            print("I’m glad you shared that. If you’d like, you can tell me a bit more "
                  "about what still feels heavy or what you need right now.\n")
            add_memory_entry(
                conversation_history,
                "bot",
                "Acknowledgement bridge",
                paraphrased=None,
                predicted_cues=None,
                bot_mode="small",
            )
            continue

        # Semantic follow-up detection
        last_user = get_last_user_message(conversation_history)
        if last_user:
            ref_text = last_user.get("paraphrased") or last_user.get("text", "")
            followup, sim = is_semantic_followup(user_text_raw, ref_text, nlp)
        else:
            followup, sim = False, 0.0

        if followup:
            print(f"\nFollow-up detected (similarity = {sim:.2f}) → using memory context.\n")
        else:
            print(f"\nNew topic detected (similarity = {sim:.2f}).\n")

        # Save raw user message to memory first
        add_memory_entry(conversation_history, "user", user_text_raw)

        # Very short follow-up → keep it conversational, no new cues
        if followup and len(user_text_raw.split()) <= SHORT_FOLLOWUP_MAX_TOKENS:
            print("\n(Short follow-up detected — no new cue analysis.)\n")
            print("I hear you. Tell me a bit more about what’s going through your mind right now.\n")
            add_memory_entry(
                conversation_history,
                "bot",
                "Short follow-up bridge",
                paraphrased=None,
                predicted_cues=None,
                bot_mode="small",
            )
            continue

        # Build memory string for paraphrasing
        memory_str = "\n".join(
            [
                f"Earlier you said: {m['text']}"
                for m in conversation_history
                if m["role"] == "user"
            ][:-1]  # exclude current
        )

        if followup and memory_str:
            paraphrase_input = memory_str + "\nCurrent message: " + user_text_raw
        else:
            paraphrase_input = user_text_raw

        # Paraphrase with guard
        print("\nInterpreting your message...\n")
        fixed_text = paraphrase_with_guard(llm, paraphrase_input)

        print("So what’s happening is that:")
        print(f'   "{fixed_text}"')

        confirm = input("\nIs this correct? (yes / no): ").strip().lower()
        if confirm not in ["yes", "y"]:
            print("\nOkay, let’s try again — you can rephrase your message.\n")
            continue

        user_text = fixed_text

        # Update last user message entry with paraphrased text
        for item in reversed(conversation_history):
            if item["role"] == "user" and item.get("paraphrased") is None:
                item["paraphrased"] = user_text
                break
        save_memory(conversation_history)

        # ===================
        # CLASSIFICATION
        # ===================
        with torch.no_grad():
            emb = nlp.encode(user_text).unsqueeze(0)
            logits = model(emb)
            probs = torch.sigmoid(logits)[0]

        prob_map = {cue: p.item() for cue, p in zip(cues, probs)}

        print("\n[PROBABILITIES]")
        for cue, p, th in zip(cues, probs, thresholds):
            print(f"{cue:22s}: {p.item():.4f}  (th={th:.2f})")

        predicted_cues = [
            cue for cue, p, th in zip(cues, probs, thresholds)
            if p.item() >= th
        ]

        print("\n[PREDICTED CUES]")
        print(predicted_cues if predicted_cues else "None above thresholds")

        # No cues → mode-aware neutral response
        if not predicted_cues:
            print("\n===== FINAL RESPONSE =====\n")
            if mode == "vent":
                msg = (
                    "Thank you for opening up. It’s clear that this means a lot to you, "
                    "and you don’t have to turn it into a plan right now. "
                    "If you want, you can just keep telling me what it feels like."
                )
            elif mode == "coach":
                msg = (
                    "I hear that something important is weighing on you. "
                    "Even if we don’t have all the details yet, we can still think in terms of small steps. "
                    "If you’d like, you can tell me one specific situation you want to work on first."
                )
            else:  # therapy
                msg = (
                    "Thank you for sharing this. Even if it feels hard to put into words, "
                    "you’re allowed to feel exactly how you feel right now, and you don’t "
                    "have to figure it all out alone in this moment."
                )

            print(msg + "\n")
            add_memory_entry(
                conversation_history,
                "bot",
                msg,
                paraphrased=None,
                predicted_cues=None,
                bot_mode="small" if mode != "therapy" else "therapy",
            )
            continue

        # Avoid repeating block for identical cues in follow-up
        if followup and last_predicted_cues == predicted_cues:
            print("\n(No new emotional cues — continuing conversation without repeating advice.)\n")
            bridge = (
                "It sounds like these feelings are still really present. "
                "If you want, you can tell me what part feels hardest right now, "
                "or what you wish could change first."
            )
            print(bridge + "\n")

            add_memory_entry(
                conversation_history,
                "bot",
                bridge,
                paraphrased=None,
                predicted_cues=predicted_cues,
                bot_mode="small",
            )
            continue

        last_predicted_cues = predicted_cues

        # ================================
        # PHI-3 SELF-EDIT (SEAL MODE)
        # ================================
        print("\n[RUNNING PHI-3 SELF-EDIT (SEAL MODE)]\n")

        system_msg = (
            "You refine emotional texts for training a multi-label cue classifier.\n"
            "Return JSON: {\"refined_text\": \"...\", \"new_cues\": [...], \"confidence\": 0-1}.\n"
            "Keep the same emotional meaning. Do NOT soften, remove, or add cues."
        )
        user_msg = user_text

        seal_dict = llm.seal_edit(system_msg, user_msg)
        if seal_dict:
            improved_text = seal_dict.get("refined_text", "").strip() or user_text
            raw_cues = seal_dict.get("new_cues", []) or predicted_cues
            confidence = seal_dict.get("confidence")

            valid_cues = [c for c in raw_cues if c in cues]
            if not valid_cues:
                valid_cues = predicted_cues

            # classifier support for those cues
            min_prob = min(prob_map.get(c, 0.0) for c in valid_cues)

            # Accept only if SEAL is confident AND classifier agrees reasonably
            if(confidence is None or confidence >= 0.30) and min_prob >= 0.20:

                try:
                    save_seal_example(seal_output_path, improved_text, valid_cues, confidence)
                except Exception as e:
                    print("\nCould not save SEAL example:", e)
            else:
                print(
                    f"\nSEAL edit not accepted "
                    f"(confidence={confidence}, min_prob={min_prob:.2f})."
                )
        else:
            print("\nNo valid SEAL edit returned — skipping SEAL example.")

        seal_count = count_seal_examples(seal_output_path)
        print(f"\n[SEAL DATA COUNT]: {seal_count} / {SEAL_TRIGGER}")

        if seal_count >= SEAL_TRIGGER:
            print("\nAUTO-SEAL TRIGGERED — Retraining model...\n")
            subprocess.run(
                [sys.executable, "-m", "chatbot.train_cue_classifier", "self"],
                check=True
            )
            print("\nNew model trained using SEAL data")
            seal_output_path.open("w", encoding="utf-8").close()
            print("SEAL data file cleared\n")

        # ================================
        # BUILD RESPONSE (THERAPY / VENT / COACH)
        # ================================
        recommendation = build_recommendation(predicted_cues)

        print("\n===== RECOMMENDATION (STRUCTURED) =====")
        print("Cues:", recommendation.get("cue_labels"))
        print("Is crisis:", recommendation.get("is_crisis"))
        print("Validation:", recommendation.get("validation"))
        print("Normalization:", recommendation.get("normalization"))
        print("Psychoeducation:", recommendation.get("psychoeducation"))
        print("Strategies:")
        for s in recommendation.get("strategies", []):
            print("  -", s)
        print("Reflection question:", recommendation.get("reflection_question"))

        # ----- Mode-specific final text -----
        if recommendation.get("is_crisis"):
            # should rarely happen here because we already handle crisis earlier
            final_narrative = (
                "Thank you for sharing this. What you're describing sounds really serious, "
                "and it deserves support from someone who can be with you in person. "
                "If you're in danger or feel you might hurt yourself, please contact local emergency services "
                "or a crisis hotline right away."
            )
        elif mode == "vent":
            # validation + normalization only, no strategies
            final_narrative = (
                f"{recommendation['validation']} {recommendation['normalization']}\n\n"
                "You don’t need to fix anything right now. If it helps, you can just keep telling me "
                "what this feels like for you."
            )
        elif mode == "coach":
            # more action-oriented, using strategies as tiny steps
            strategies = recommendation.get("strategies", [])
            base = f"{recommendation['validation']} {recommendation['normalization']}\n\n"
            base += (
                "Let’s focus on one or two very small steps you could try:\n"
            )
            if strategies:
                base += f"- {strategies[0]}\n"
            if len(strategies) > 1:
                base += f"- {strategies[1]}\n"

            rq = recommendation.get("reflection_question")
            if rq:
                base += (
                    "\nIf these still feel too big, think about an even smaller version. "
                    f"{rq}"
                )
            final_narrative = base.strip()
        else:
            # therapy mode → use narrative generator (which already humanizes via Phi)
            final_narrative = generate_final_narrative(
                predicted_cues,
                severity=None,
                humanize=True,
            )

        print("\n===== FINAL THERAPEUTIC RESPONSE =====\n")
        print(final_narrative)
        print("\n" + "=" * 60 + "\n")

        # Save bot therapy block to memory
        add_memory_entry(
            conversation_history,
            "bot",
            final_narrative,
            paraphrased=None,
            predicted_cues=predicted_cues,
            bot_mode="therapy" if mode == "therapy" else mode,
        )
