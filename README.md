# PII Redaction Tool

A full-stack tool for automatically detecting and redacting Personally Identifiable Information (PII) from PDF documents. Built as a Senior Capstone Project by Walker Moody, Blaire Wallace, Ash Sonar, Jace Fergusson, and Josiah Luke.

---

## How It Works

The core of the tool is `redact.py`, which uses a **hybrid detection pipeline** combining two complementary NLP approaches:

### 1. Presidio (Rule-Based + spaCy)
[Microsoft Presidio](https://microsoft.github.io/presidio/) provides the first layer of detection using a spaCy `en_core_web_lg` model and custom regex-based recognizers for:

- **Dates** — `MM/DD/YYYY`, ISO format, and written-out month formats
- **US Addresses** — multi-pattern regex covering numbered streets, directional prefixes, unit numbers, and state abbreviations
- **Medical Record Numbers** — explicit `MRN:` labels and generic numeric IDs with contextual hints

### 2. Transformer Model (XLM-R Fine-Tuned)
A fine-tuned XLM-RoBERTa token classification model (`xlmr_pii_ner/`) provides the second layer. It predicts BIO-tagged entity spans for:

| Model Label | Mapped Entity |
|---|---|
| `PERSON` | `PERSON` |
| `LOCATION` | `LOCATION` |
| `ADDRESS` | `US_ADDRESS` |
| `ZIP_CODE` | `ZIP_CODE` |
| `CREDIT_CARD` | `CREDIT_CARD` |
| `EMAIL_ADDRESS` | `EMAIL_ADDRESS` |
| `PHONE_NUMBER` | `PHONE_NUMBER` |
| `DATE_TIME` | `STRICT_DATE` |

The transformer runs on GPU if available, falling back to CPU automatically.

### 3. Span Merging
Results from both detectors are merged: overlapping spans of the same entity type are unioned, and non-overlapping spans from either source are kept. This means the transformer catches what regex misses and vice versa.

### 4. Header Name Detection
The first 8 lines of each page are scanned for likely name headers (e.g. `Dr. Jane Smith`) using heuristics: capitalized tokens, optional title prefix, no digits or colons, max 4 tokens. Detected header names are added to the redaction list even if not caught by the main detectors.

### 5. PDF Redaction
Detected PII terms are matched back to their visual positions in the PDF using [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/). Multi-word terms are matched line-by-line using a sliding window over word tokens. Single-word terms are matched by exact normalized token. Each match gets a filled black `add_redact_annot` box applied via `apply_redactions()`, permanently removing the underlying text.

---

## Supported Entity Types

| Entity | Description |
|---|---|
| `PERSON` | Full names and partial names |
| `EMAIL_ADDRESS` | Email addresses |
| `PHONE_NUMBER` | US phone numbers |
| `CREDIT_CARD` | Credit card numbers |
| `US_SSN` | Social Security Numbers |
| `ZIP_CODE` | 5-digit ZIP codes |
| `AGE` | Age references |
| `LOCATION` | Cities, states, countries |
| `US_ADDRESS` | Street addresses |
| `STRICT_DATE` | Dates in common formats |
| `MEDICAL_RECORD_NUMBER` | MRN identifiers |

---

## Project Structure

```
RedactionTool/
├── redact.py              # Core detection + redaction engine
├── api/
│   └── server.py          # FastAPI server exposing POST /api/redact
├── frontend/              # Angular 21 web interface
│   └── src/
│       └── app/
│           ├── app.component.ts
│           ├── app.component.html
│           └── app.component.css
├── xlmr_pii_ner/          # Fine-tuned XLM-R model weights
├── input/                 # Drop PDFs here for CLI usage
└── output/                # Redacted PDFs written here (CLI usage)
```

---

## Setup

### Requirements

- Python 3.10+
- conda environment named `redact`
- Node.js 18+ and npm (for frontend)

### Backend

```bash
conda activate redact
pip install fastapi uvicorn pymupdf presidio-analyzer spacy transformers torch
python -m spacy download en_core_web_lg
```

### Frontend

```bash
cd frontend
npm install
```

---

## Running

### Start the API server

```bash
conda run -n redact uvicorn api.server:app --reload --port 8000
```

### Start the frontend dev server

```bash
cd frontend
ng serve
```

Then open [http://localhost:4200](http://localhost:4200).

### CLI (batch redact without the UI)

Place PDFs in the `input/` folder and run:

```bash
conda run -n redact python redact.py
```

Redacted PDFs are written to `output/` with a `_REDACTED` suffix.

---

## API

**POST** `/api/redact`

| Field | Type | Description |
|---|---|---|
| `file` | File | PDF to redact |
| `entities` | JSON array | Entity types to redact (defaults to all) |
| `exclusions` | JSON array | Names to exclude from redaction |

Returns the redacted PDF as a binary file download.

### Example

```bash
curl -X POST http://localhost:8000/api/redact \
  -F "file=@document.pdf" \
  -F 'entities=["PERSON","EMAIL_ADDRESS","PHONE_NUMBER"]' \
  -F 'exclusions=["Acme Corp"]'
```

---

## Exclusions

Names passed in the `exclusions` field are protected from redaction. Matching strips punctuation and checks both full phrases and individual words — so excluding `"John Smith"` also prevents `"John"` and `"Smith"` from being redacted independently.
