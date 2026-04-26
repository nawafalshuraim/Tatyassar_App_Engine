# Tatyassar — IDSS Chatbot Backend

A FastAPI backend for an Intelligent Decision Support System (IDSS) chatbot focused on emotional support and mental wellness.

## What it does

The chatbot receives a user message, detects emotional cues, and returns a therapeutic response. The pipeline:

1. **Input validation** — rejects empty or malformed input
2. **Crisis detection** — keyword-based check; immediately returns emergency guidance if triggered
3. **Paraphrasing** — normalizes the input using a local LLM (Phi-3)
4. **Embedding** — encodes the paraphrased text with a sentence transformer
5. **Cue classification** — a custom PyTorch model predicts emotional cues (e.g., anxiety, loneliness)
6. **Narrative generation** — blends knowledge-base rules for detected cues into a therapeutic response
7. **Humanization** — Phi-3 refines the tone before returning the final reply

## Stack

- **FastAPI** — REST API
- **PyTorch** — cue classifier (`CueClassifier`)
- **Transformers / sentence-transformers** — text encoding
- **Phi-3 (local)** — paraphrasing and response humanization
- **Knowledge base** (`knowledge_base.json`) — rules and cue definitions

## Project structure

```
api.py                  # FastAPI app and /chat endpoint
chatbot/
  preprocessor.py       # NLPModel encoder + crisis detection
  cue_classifier_model.py
  local_llm.py          # Phi-3 wrapper
  llm_humanizer.py
  narrative_generator.py
  knowledge_base.py
  chatbot.py
  input_validator.py
models/
  cue_classifier/
    model.pt
    cue_vocab.json
    thresholds.json
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

## API

### `POST /chat`

**Request**
```json
{ "message": "I've been feeling really overwhelmed lately" }
```

**Response**
```json
{
  "reply": "...",
  "predicted_cues": ["overwhelm", "anxiety"],
  "paraphrased_input": "...",
  "is_crisis": false
}
```

If a crisis is detected, `is_crisis` is `true` and `reply` contains emergency guidance directing the user to contact real-world support.
