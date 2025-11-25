import re

# ---------------- GIBBERISH CHECK ----------------

REPEATED = re.compile(r"(.)\1{6,}")       
KEYBOARD_SMASH = re.compile(r"^[bcdfghjklmnpqrstvwxyz]{7,}$", re.I)

def has_gibberish(text: str) -> bool:

    clean = text.replace(" ", "")

    # Very long character repetition
    if REPEATED.search(text):
        return True

    # Keyboard smash (only consonants, long)
    if KEYBOARD_SMASH.fullmatch(clean):
        return True

    # Too many symbols
    symbols = sum(not c.isalnum() and not c.isspace() for c in text)
    if len(text) > 0 and symbols / len(text) > 0.6:
        return True

    return False


# ---------------- MAIN VALIDATOR ----------------

def is_valid_input(text: str):

    if not text or len(text.strip()) < 2:
        return False, "Input is too short"

    if not re.search(r"[a-zA-Z]", text):
        return False, "Input must contain letters"

    if has_gibberish(text):
        return False, "Input looks like gibberish"

    return True, "OK"
