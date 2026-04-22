"""
Narrative generator (v3)
------------------------
Creates a unified therapeutic response from blended cue rules,
then hands the structured content to Phi-3 humanizer for final tone.

Key principles:
- No rigid CBT paragraph formatting.
- No double-humanization.
- Humanizer controls final tone.
- Generator handles crisis safety, rule blending, and structure.
"""

from typing import List, Optional, Dict
from chatbot.knowledge_base import (
    get_rules_for_cues,
    has_crisis_risk,
    blend_rules,
    apply_severity_modifiers,
)
from chatbot.preprocessor import detect_crisis
from chatbot.llm_humanizer import humanize_with_phi3
from chatbot.local_llm import LocalEditModel


# -----------------------------------------------------------
# Lazy-load the LLM one time only
# -----------------------------------------------------------
_llm: Optional[LocalEditModel] = None

def _llm_loader() -> LocalEditModel:
    global _llm
    if _llm is None:
        _llm = LocalEditModel()
    return _llm


# -----------------------------------------------------------
# CRISIS MESSAGE — NEVER HUMANIZED
# -----------------------------------------------------------
CRISIS_MESSAGE = (
    "Thank you for telling me this. What you're feeling matters, and you deserve care and support from someone who "
    "can be present with you. If you're in immediate danger or feel unsafe, please consider contacting emergency "
    "services or a trusted person right now."
)


# -----------------------------------------------------------
# MAIN GENERATOR
# -----------------------------------------------------------
def generate_final_narrative(
    predicted_cues: List[str],
    severity: Optional[str] = None,
    humanize: bool = True,
) -> str:
    """
    Steps:
      0) No cues → gentle default structure
      1) Fetch rules
      2) Crisis check (NEVER humanized)
      3) Blend rules
      4) Apply severity modifiers
      5) Humanize with Phi-3
    """

    # =======================================================
    # 0) No cues → gentle universal structure
    # =======================================================
    if not predicted_cues:
        base = {
            "validation": "It sounds like you're carrying something meaningful, and I'm here with you.",
            "normalization": "",
            "psychoeducation": "",
            "coping_strategies": [],
            "reflection_question": "What part of this feels most present for you right now?",
        }
        return (
            humanize_with_phi3(_llm_loader(), base)
            if humanize
            else base["validation"]
        )

    # =======================================================
    # 1) Get rules for cues
    # =======================================================
    rules = get_rules_for_cues(predicted_cues)
    if not rules:
        base = {
            "validation": "Thank you for sharing this with me.",
            "normalization": "",
            "psychoeducation": "",
            "coping_strategies": [],
            "reflection_question": "What feels most important about this moment for you?",
        }
        return (
            humanize_with_phi3(_llm_loader(), base)
            if humanize
            else base["validation"]
        )

    # =======================================================
    # 2) Crisis check — NEVER humanized
    # =======================================================
    if has_crisis_risk(rules):
        return CRISIS_MESSAGE

    # =======================================================
    # 3) Blend rule templates → structured dict
    # =======================================================
    structured = blend_rules(rules)
    # Example structure:
    # {
    #   "validation": "...",
    #   "normalization": "...",
    #   "psychoeducation": "...",
    #   "coping_strategies": [...],
    #   "reflection_question": "..."
    # }

    # =======================================================
    # 4) Apply severity modifiers
    # =======================================================
    structured = apply_severity_modifiers(structured, severity)

    # Safety: Ensure structure always contains expected keys
    structured.setdefault("validation", "")
    structured.setdefault("normalization", "")
    structured.setdefault("psychoeducation", "")
    structured.setdefault("coping_strategies", [])
    structured.setdefault("reflection_question", "What feels most present for you right now?")

    # =======================================================
    # 5) Humanize the structured block
    # =======================================================
    return (
        humanize_with_phi3(_llm_loader(), structured)
        if humanize
        else structured
    )
