import re
from chatbot.preprocessor import clean_text, detect_crisis

# GIBBERISH SIGNALS
REPEATED = re.compile(r"(.)\1{8,}")  # allow emotional repetition
KEYBOARD_SMASH = re.compile(r"^[bcdfghjklmnpqrstvwxyz]{8,}$", re.I)

def has_gibberish(text: str) -> bool:
    clean = text.replace(" ", "")

    # insane repetition but allow emotion (e.g. "noooo", "stopppp")
    if REPEATED.search(text) and not re.search(r"[aeiou]", text, re.I):
        return True

    # keyboard smash (only consonants, no vowels)
    if KEYBOARD_SMASH.fullmatch(clean):
        return True

    # too many symbols (except safe emojis)
    symbols = sum(not c.isalnum() and not c.isspace() for c in text)
    if len(text) > 0 and symbols / len(text) > 0.7:
        return True

    return False


# MAIN VALIDATOR
def is_valid_input(raw_text: str):
    # Clean first
    text = clean_text(raw_text)

    if detect_crisis(text):
        return False, "This needs immediate crisis support, not a classifier"

    if not text or len(text.strip()) < 3:
        return False, "Input is too short"

    if not re.search(r"[a-zA-Z\u0600-\u06FF]", text):
        return False, "Input must contain letters"

    if has_gibberish(text):
        return False, "Input looks like gibberish"

    return True, "OK"
