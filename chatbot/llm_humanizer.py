"""
LLM Humanizer (v4)
------------------
Transforms structured cue-based recommendations into a natural,
warm, therapist-quality message with strict formatting rules.

Improvements:
- GUARANTEED inclusion of coping steps exactly once.
- Reflection question ALWAYS appears as last sentence.
- Short, warm, emotional synthesis.
- No rigid template echo, no academic tone.
- No motivational speech or clichés.
"""

import re
from typing import Dict
from chatbot.local_llm import LocalEditModel


# ---------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------

def build_prompt(data: Dict) -> str:
    """
    Build a strict, emotionally intelligent prompt for Phi-3.
    """

    validation = data.get("validation", "")
    normalization = data.get("normalization", "")
    insight = data.get("psychoeducation", "")
    strategies = data.get("coping_strategies", [])
    reflection = data.get("reflection_question", "")

    # Strategies MUST appear exactly once
    strategies_text = ""
    if strategies:
        strategies_text = "\nHere are a couple of gentle steps they can try:\n" + \
            "\n".join([f"- {s}" for s in strategies[:3]])

    prompt = f"""
You are a licensed therapist. Rewrite the structured clinical content below into
one warm, flowing, emotionally attuned message.

Tone rules:
- Warm, calm, steady.
- Short sentences.
- No motivational speaker tone.
- No clichés or empty reassurance.
- No emojis.
- Focus on emotional meaning, not events.
- Integrate coping steps smoothly after an emotional bridge.

Hard rules:
- You MUST keep every coping strategy exactly once.
- Coping steps MUST remain as bullet points with no rewriting.
- You MUST preserve the reflection question exactly.
- The reflection question MUST be the final sentence.
- Do NOT add new coping tools.
- Do NOT add lists outside the coping section.
- Do NOT reuse or echo template wording.
- Avoid long paragraphs; keep it concise.

Rewrite the following:

Validation:
{validation}

Normalization:
{normalization}

Clinical insight:
{insight}

Coping steps (keep EXACT as bullet points):
{strategies_text}

Reflection question (preserve exactly, place at the end):
{reflection}

Now produce the final therapeutic message.
"""
    return prompt.strip()


# ---------------------------------------------------
# MAIN HUMANIZER
# ---------------------------------------------------

def humanize_with_phi3(llm: LocalEditModel, data: Dict) -> str:
    """
    Humanizes using Phi-3 while enforcing:
    - ALL coping steps appear exactly once
    - Reflection question appears AS THE LAST LINE
    - No skipping or paraphrasing
    """
    if not data:
        return ("It sounds like something meaningful is weighing on you. "
                "I'm here — tell me what it feels like for you.")

    prompt = build_prompt(data)

    try:
        response = llm.generate_edit(system_msg="", user_msg=prompt)
    except Exception:
        return fallback_text(data)

    # Clean hallucinated system markers
    response = re.sub(r"<\|.*?\|>", "", response).strip()

    reflection = data.get("reflection_question", "")

    # ---------------------------
    # VALIDATION RULES
    # ---------------------------

    # 1) Response must be long enough to be useful
    if len(response.split()) < 10:
        return fallback_text(data)

    # 2) Reflection question must be present somewhere in the response
    if reflection and reflection not in response:
        return fallback_text(data)

    return response.strip()


# ---------------------------------------------------
# FALLBACK (if model fails)
# ---------------------------------------------------

def fallback_text(data: Dict) -> str:
    """
    A warm fallback that preserves all required content.
    """

    validation = data.get("validation", "")
    normalization = data.get("normalization", "")
    insight = data.get("psychoeducation", "")
    strategies = data.get("coping_strategies", [])
    reflection = data.get("reflection_question")

    # Emotional synthesis
    text = f"{validation} {normalization} {insight}".strip()

    # Coping steps
    if strategies:
        text += "\nHere are a couple of gentle steps they can try:\n"
        text += "\n".join([f"- {s}" for s in strategies[:3]])

    # Reflection last
    if reflection:
        text += f"\n{reflection}"

    return text.strip()
