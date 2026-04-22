
from typing import List, Dict, Optional
from chatbot.knowledge_base import (
    get_rules_for_cues,
    has_crisis_risk,
    blend_rules,
    apply_severity_modifiers,
)


def build_recommendation(
    cues: List[str],
    severity: Optional[str] = None
) -> Dict:
    """
    Combine multiple cues into ONE structured therapeutic recommendation.
    Includes safety routing and optional severity scaling.
    """

    # 1) Get valid rules
    rules = get_rules_for_cues(cues)

    if not rules:
        return {
            "cue_labels": [],
            "is_crisis": False,
            "validation": "Thank you for sharing.",
            "normalization": "Your feelings matter, and it’s okay to take things step by step.",
            "psychoeducation": "",
            "strategies": [],
            "reflection_question": None
        }

    # 2) Crisis safety
    if has_crisis_risk(rules):
        return {
            "cue_labels": [r.cue for r in rules],
            "is_crisis": True,
            "validation": (
                "Thank you for sharing this. What you're feeling matters, and you deserve support."
            ),
            "normalization": "",
            "psychoeducation": "",
            "strategies": [],
            "reflection_question": None
        }

    # 3) Blend the cues into unified structured text
    content = blend_rules(rules)

    # 4) Apply optional severity modifiers
    content = apply_severity_modifiers(content, severity)

    # 5) Expand into structured dict for UI + narrative module
    return {
        "cue_labels": [r.cue for r in rules],
        "is_crisis": False,
        "validation": content["validation"],
        "normalization": content["normalization"],
        "psychoeducation": content["psychoeducation"],
        "strategies": content["coping_strategies"],  # <= MAX 2 automatically
        "reflection_question": content["reflection_question"]
    }
