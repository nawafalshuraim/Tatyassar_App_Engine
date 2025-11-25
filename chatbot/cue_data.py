"""
Bridge between dataset and neural network, prepares the data for ML.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json


# Data structures 
"""
Represents one row of training data.
"""
@dataclass
class CueExample:
    example_id: str
    # Client & Therapist
    text: str
    true_cues: List[str]
    label_indices: List[int]

class CueVocab:
    """
    Maps cue strings to integer indices and back.
    """
    def __init__(self, cues: List[str]):
        unique = sorted(set(cues))
        self.cues: List[str] = unique
        self.cue_to_id: Dict[str, int] = {c: i for i, c in enumerate(unique)}

    def encode(self, cue_list: List[str]) -> List[int]:
        return [self.cue_to_id[c] for c in cue_list if c in self.cue_to_id]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.cues[i] for i in indices]

    @property
    def size(self) -> int:
        return len(self.cues)


# JSON helpers 
def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_id_to_challenge(challenges_path: Path) -> Dict[str, Dict]:
    data = _load_json(challenges_path)
    return {item["id"]: item for item in data}


def _build_id_to_solution(solutions_path: Path) -> Dict[str, Dict]:
    data = _load_json(solutions_path)
    return {item["id"]: item for item in data}


def build_cue_vocab(solutions_path: Path) -> CueVocab:
    """
    Scan solutions.json to collect all valid cues.
    Automatically filters:
        - single characters
        - punctuation
        - empty strings
        - nonsense tokens
    """
    id_to_solution = _build_id_to_solution(solutions_path)
    all_cues: List[str] = []

    for rec in id_to_solution.values():
        out = rec.get("correct_output", {})
        cues = out.get("true_cues", [])
        all_cues.extend(cues)

    # CLEANING LOGIC 
    cleaned = []
    for c in all_cues:
        c = c.strip()

        if len(c) <= 2:
            continue  # remove: "a", "c", ",", etc.

        if not any(ch.isalpha() for ch in c):
            continue  # no letters? Remove: "[", "]", " " etc.

        cleaned.append(c)

    # Remove duplicates
    cleaned = sorted(set(cleaned))

    return CueVocab(cleaned)



def build_examples(
    challenges_path: Path,
    solutions_path: Path,
    cue_vocab: CueVocab | None = None,
) -> Tuple[List[CueExample], CueVocab]:
    """
    Align challenges.json with solutions.json and create raining objects - CueExample.
    """
    id_to_challenge = _build_id_to_challenge(challenges_path)
    id_to_solution = _build_id_to_solution(solutions_path)

    if cue_vocab is None:
        cue_vocab = build_cue_vocab(solutions_path)

    examples: List[CueExample] = []

    for ex_id, challenge in id_to_challenge.items():
        solution = id_to_solution.get(ex_id)
        if solution is None:
            # skip if no solution for this challenge
            continue

        text = challenge["input_text"]
        correct_out = solution.get("correct_output", {})
        true_cues: List[str] = correct_out.get("true_cues", [])
        label_indices = cue_vocab.encode(true_cues)

        examples.append(
            CueExample(
                example_id=ex_id,
                text=text,
                true_cues=true_cues,
                label_indices=label_indices,
            )
        )

    return examples, cue_vocab


# Debug helper 

def debug_print_stats(root: Path | None = None):
    if root is None:
        root = Path(__file__).resolve().parents[1]

    challenges_path = root / "challenges_solutions_data" / "challenges.json"
    solutions_path = root / "challenges_solutions_data" / "solutions.json"

    examples, vocab = build_examples(challenges_path, solutions_path)

    print(f"Total examples: {len(examples)}")
    print(f"Number of unique cues: {vocab.size}")
    print(f"Cues: {vocab.cues}")

    for ex in examples[:3]:
        print("-" * 40)
        print("ID:", ex.example_id)
        print("Snippet:", ex.text[:200], "...")
        print("True cues:", ex.true_cues)
        print("Label indices:", ex.label_indices)


if __name__ == "__main__":
    debug_print_stats()