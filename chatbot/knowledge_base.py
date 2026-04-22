"""
Knowledge-base engine (v3)
--------------------------
Enhances cue blending for IDSS psychotherapy.

Upgrades:
- Emotion-aware blending with lead + support cues
- High-quality template synthesis for humanizer
- Predictable rule merging for SEAL-valid outputs
- Zero-None guarantees for all structured fields
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json
import random
from pathlib import Path


# ======================================================
# DATA STRUCTURE FOR EACH CUE
# ======================================================

@dataclass
class CueRule:
    cue: str
    short_label: str
    validation: List[str]
    normalization: List[str]
    psychoeducation: List[str]
    coping_strategies: List[str]
    reflection_questions: List[str]
    crisis_flag: bool = False


# ======================================================
# GLOBAL REGISTRY
# ======================================================

RECOMMENDATION_RULES: Dict[str, CueRule] = {}


# ======================================================
# LOAD KNOWLEDGE BASE
# ======================================================

def load_kb():
    """Load cue profiles from JSON into CueRule objects."""
    kb_path = Path(__file__).resolve().parents[1] / "knowledge_base.json"

    with kb_path.open("r", encoding="utf-8") as f:
        data = json.load(f)["cue_profiles"]

    RECOMMENDATION_RULES.clear()

    for cue, info in data.items():
        RECOMMENDATION_RULES[cue] = CueRule(
            cue=cue,
            short_label=info.get("short_label", ""),
            validation=info.get("validation_templates", []),
            normalization=info.get("normalization_templates", []),
            psychoeducation=info.get("psychoeducation_templates", []),
            coping_strategies=info.get("coping_ideas", []),
            reflection_questions=info.get("reflection_questions", []),
            crisis_flag=info.get("crisis_flag", False)
        )

    print(f"[KB] Loaded {len(RECOMMENDATION_RULES)} cue profiles.")


# ======================================================
# RULE ACCESS HELPERS
# ======================================================

def get_rules_for_cues(active_cues: List[str]) -> List[CueRule]:
    """Return CueRule objects for all cues that exist in KB."""
    return [RECOMMENDATION_RULES[c] for c in active_cues if c in RECOMMENDATION_RULES]


def has_crisis_risk(rules: List[CueRule]) -> bool:
    """Any cue with crisis_flag=True forces crisis response."""
    return any(rule.crisis_flag for rule in rules)


# ======================================================
# UTILITY PICKERS
# ======================================================

def pick(options: List[str]) -> str:
    """Pick one string safely; return '' if list is empty."""
    return random.choice(options) if options else ""


def merge_unique(groups: List[List[str]], limit: int) -> List[str]:
    """
    Merge multiple lists into a stable unique list, up to limit.
    Keeps order of appearance.
    """
    seen = set()
    out = []

    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                out.append(item)
                if len(out) >= limit:
                    return out

    return out


# ======================================================
# EMOTION-AWARE MULTI-CUE BLENDING
# ======================================================

def blend_rules(rules: List[CueRule]) -> Dict[str, Any]:
    """
    Blends cue rules into a coherent therapeutic structure.

    Rules:
    - First cue = emotional anchor (main tone)
    - Secondary cues enrich strategies + reflection
    - Guarantees valid output for humanizer
    """

    if not rules:
        return {
            "validation": "",
            "normalization": "",
            "psychoeducation": "",
            "coping_strategies": [],
            "reflection_question": "",
        }

    lead = rules[0]
    secondary = rules[1:]

    # ----------------------------
    # Primary tone from lead cue
    # ----------------------------
    validation = pick(lead.validation)
    normalization = pick(lead.normalization)
    psycho = pick(lead.psychoeducation)

    # ----------------------------
    # Strategies — merge unique across cues
    # ----------------------------
    strategies = merge_unique(
        [lead.coping_strategies] + [r.coping_strategies for r in secondary],
        limit=3
    )

    # ----------------------------
    # Reflection — pick exactly one if possible
    # ----------------------------
    reflections = merge_unique(
        [[q for q in lead.reflection_questions]] +
        [[q for q in r.reflection_questions] for r in secondary],
        limit=1
    )
    reflection = reflections[0] if reflections else ""

    # ----------------------------
    # Build structured therapeutic block
    # ----------------------------
    return {
        "validation": validation or "",
        "normalization": normalization or "",
        "psychoeducation": psycho or "",
        "coping_strategies": strategies or [],
        "reflection_question": reflection or "",
    }


# ======================================================
# SEVERITY MODIFIERS
# ======================================================

def apply_severity_modifiers(message: Dict[str, Any], severity: Optional[str]) -> Dict[str, Any]:
    """
    Add stronger normalization + gentle outreach suggestion
    when severity is high.
    """
    sev = severity.lower() if isinstance(severity, str) else None

    if sev in ["high", "severe"]:
        message["normalization"] = (
            (message.get("normalization") or "")
            + " This level of emotional weight can feel extremely hard to carry alone."
        )

        # Only add if not already present
        support_line = "Consider reaching out to a supportive person or counselor if that feels safe."
        if support_line not in message.get("coping_strategies", []):
            message.setdefault("coping_strategies", []).append(support_line)

    return message
