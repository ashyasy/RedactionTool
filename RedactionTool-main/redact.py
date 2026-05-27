from __future__ import annotations

import re
from pathlib import Path
import shutil

import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import PatternRecognizer, Pattern


############################################################################################
# CONFIG
############################################################################################

DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "ZIP_CODE",
    "AGE",
    "LOCATION",
    "US_ADDRESS",
    "STRICT_DATE",
    "MEDICAL_RECORD_NUMBER",
]

NAME_TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "doctor", "professor"}
HEADER_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "doctor", "professor"}

ENTITY_MIN_SCORE = {
    "PERSON": 0.80,
    "STRICT_DATE": 0.60,
    "US_ADDRESS": 0.70,
    "MEDICAL_RECORD_NUMBER": 0.85,
}

MODEL_DIR = "xlmr_pii_ner"
TRANSFORMER_MAX_LENGTH = 256
TRANSFORMER_SCORE_THRESHOLD = 0.70

# Map trained model entity names to your old pipeline entity names
MODEL_TO_OLD_ENTITY = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "ADDRESS": "US_ADDRESS",
    "ZIP_CODE": "ZIP_CODE",
    "CREDIT_CARD": "CREDIT_CARD",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "DATE_TIME": "STRICT_DATE",
    # Uncomment only if you want companies/orgs redacted too:
    # "ORGANIZATION": "LOCATION",
}


############################################################################################
# PRESIDIO SETUP
############################################################################################

def add_strict_date_recognizer(analyzer: AnalyzerEngine) -> None:
    date_patterns = [
        Pattern(
            name="mm_dd_yyyy",
            score=0.95,
            regex=r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        ),
        Pattern(
            name="iso_date",
            score=0.95,
            regex=r"\b\d{4}-\d{2}-\d{2}\b",
        ),
        Pattern(
            name="month_day_year",
            score=0.95,
            regex=r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4}\b",
        ),
        Pattern(
            name="month.year",
            score=0.95,
            regex=r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\.?\s+\d{4}\b",
        ),
    ]
    recognizer = PatternRecognizer(
        supported_entity="STRICT_DATE",
        patterns=date_patterns,
    )
    analyzer.registry.add_recognizer(recognizer)


def add_mrn_recognizer(analyzer: AnalyzerEngine) -> None:
    mrn_patterns = [
        Pattern(
            name="mrn_with_label",
            score=0.99,
            regex=r"\bMRN[-:\s]?\d{4,10}\b",
        ),
        Pattern(
            name="generic_medical_id",
            score=0.85,
            regex=r"\b\d{6,10}\b",
        ),
    ]

    recognizer = PatternRecognizer(
        supported_entity="MEDICAL_RECORD_NUMBER",
        patterns=mrn_patterns,
        context=["medical record", "mrn", "record number"],
    )
    analyzer.registry.add_recognizer(recognizer)


def add_us_address_recognizer(analyzer: AnalyzerEngine) -> None:
    us_address_patterns = [
        Pattern(
            name="us_address_full",
            score=0.85,
            regex=r"\b\d{1,6}(?:\s+[\w.'-]+){1,10}\s+(?:St|Street|Square|Hall|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Broadway|Place|Ter|Terrace|Center)\b(?:\s*,?\s*(?:Apt|Apartment|Unit|Ste|Suite|#)\s*\w+)?(?:\s*,?\s*[A-Za-z .'-]{2,30})?(?:\s*,?\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY))?(?:\s+\d{5}(?:-\d{4})?)?\b",
        ),
        Pattern(
            name="us_address_numbered_street",
            score=0.90,
            regex=r"""
            \b
            \d{1,6}
            (?:\s+(?:N|S|E|W|NE|NW|SE|SW)\.?)?
            \s+
            (?:\d{1,5}(?:st|nd|rd|th)\b|[\w.'-]+)
            (?:\s+[\w.'-]+){0,5}
            \s+
            (?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Broadway|Center|Sq|Square)\.?
            (?:\s*,?\s*(?:Apt|Apartment|Unit|Ste|Suite|#)\s*[\w-]+)?
            (?:\s*,?\s*[A-Za-z .'-]{2,30})?
            (?:\s*,?\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY))?
            (?:\s+\d{5}(?:-\d{4})?)?
            \b
            """.strip(),
        ),
        Pattern(
            name="us_street_general",
            score=0.90,
            regex=r"""
            \b
            (?:
                \d{1,6}(?:\s+(?:N|S|E|W|NE|NW|SE|SW)\.?)?
                |
                (?:North|South|East|West|N|S|E|W)\.?
            )
            \s+
            (?:\d{1,5}\s*(?:st|nd|rd|th)\b|[\w.'-]+)
            (?:\s+[\w.'-]+){0,4}
            \s+
            (?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Broadway|Center|Sq|Square)\.?
            \b
            """.strip(),
        ),
        Pattern(
            name="numbered_street_optional_suffix",
            score=0.95,
            regex=r"""
            \b
            \d{1,6}
            (?:\s+(?:N|S|E|W|NE|NW|SE|SW)\.?)?
            \s+
            (?:\d{1,5}(?:st|nd|rd|th)\b|[\w.'-]+)
            (?:\s+[\w.'-]+){0,3}
            (?:
                \s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Sq|Square)\.?
            )?
            \b
            """.strip(),
        ),
    ]

    recognizer = PatternRecognizer(
        supported_entity="US_ADDRESS",
        patterns=us_address_patterns,
        context=["address", "st", "broadway", "street", "ave", "road", "blvd", "apt", "suite", "zip"],
    )
    analyzer.registry.add_recognizer(recognizer)


def build_presidio():
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
    )
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

    add_us_address_recognizer(analyzer)
    add_strict_date_recognizer(analyzer)
    add_mrn_recognizer(analyzer)

    return analyzer


ANALYZER = build_presidio()


############################################################################################
# TRANSFORMER SETUP
############################################################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    TRANSFORMER_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
    TRANSFORMER_MODEL = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    TRANSFORMER_MODEL.to(DEVICE)
    TRANSFORMER_MODEL.eval()
    TRANSFORMER_READY = True
    print(f"Loaded transformer model from: {MODEL_DIR}")
    print(f"Transformer device: {DEVICE}")
except Exception as e:
    TRANSFORMER_TOKENIZER = None
    TRANSFORMER_MODEL = None
    TRANSFORMER_READY = False
    print(f"Transformer model not loaded: {e}")
    print("Continuing with Presidio-only mode.")


############################################################################################
# ALLOWLIST / HELPERS
############################################################################################

ALLOWLIST_EXACT = {
    "inc", "llc", "ltd",
    "university", "department",
    "texas", "california", "lubbock",
    "usa", "united states", "weeks ago", "months ago", "minutes",
    "date", "date:", "birth", "email", "biographies",
    "dyspnea", "epigastric", "lumbosacral", "lumbo", "musculoskeletal",
    "the", "is", "in", "ist",
}

DATE_PATTERNS = [
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?$", re.I),
    re.compile(r"^\d{1,2},?$"),
    re.compile(r"^\d{4}$"),
]

_WORD_CH = re.compile(r"[A-Za-z0-9_]")


class SimpleResult:
    def __init__(self, start, end, entity_type, score):
        self.start = start
        self.end = end
        self.entity_type = entity_type
        self.score = score


def should_keep_term(term: str) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    if len(t) == 1 and t.isalpha():
        return False
    return True


def is_allowlisted(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    s_norm = re.sub(r"\s+", " ", s).lower()
    return s_norm in {x.lower() for x in ALLOWLIST_EXACT}


def split_and_filter_allowlisted(snippet: str) -> list[str]:
    s = re.sub(r"\s+", " ", (snippet or "")).strip()
    if not s:
        return []

    parts = s.split()
    kept = []
    for p in parts:
        p_clean = re.sub(r"^[^\w]+|[^\w]+$", "", p).strip()
        if not p_clean:
            continue
        if is_allowlisted(p_clean):
            continue
        kept.append(p_clean)

    if kept and (len(kept) != len(parts)):
        return kept

    return [s]


def is_whole_word_span(text: str, start: int, end: int) -> bool:
    if start < 0 or end > len(text) or start >= end:
        return False

    left_ok = (start == 0) or (not _WORD_CH.match(text[start - 1]))
    right_ok = (end == len(text)) or (not _WORD_CH.match(text[end]))
    return left_ok and right_ok


def _norm_token(s: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", (s or "")).lower()


def _normalize_names(names: list[str]) -> list[str]:
    seen = set()
    out = []
    for n in names:
        n = (n or "").strip()
        if not n:
            continue
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def _clean_person_tokens(raw: str) -> list[str]:
    out = []
    for token in raw.split():
        clean = re.sub(r"^[^\w]+|[^\w]+$", "", token).strip()
        if not clean:
            continue
        if is_allowlisted(clean):
            continue

        if len(clean) == 1 and clean.isalpha():
            out.append(clean)
            continue

        if not should_keep_term(clean):
            continue

        out.append(clean)
    return out


def _looks_like_header_name(line: str) -> bool:
    line = re.sub(r"\s+", " ", (line or "")).strip()
    if not line:
        return False

    if any(ch.isdigit() for ch in line):
        return False
    if "@" in line or ":" in line:
        return False
    if len(line) > 40:
        return False

    parts = [re.sub(r"^[^\w]+|[^\w]+$", "", p) for p in line.split()]
    parts = [p for p in parts if p]
    if not parts:
        return False

    if len(parts) > 4:
        return False

    lowered = [p.lower() for p in parts]

    if lowered[0] in HEADER_TITLES:
        parts = parts[1:]
        lowered = lowered[1:]

    if not parts:
        return False

    for p in parts:
        if len(p) < 2:
            return False
        if not re.match(r"^[A-Z][a-zA-Z.'-]*$", p):
            return False
        if is_allowlisted(p):
            return False

    return True


def extract_header_name_candidates(page_text: str, max_lines: int = 8) -> list[str]:
    out = []
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in page_text.splitlines()]
    lines = [ln for ln in lines if ln]

    for line in lines[:max_lines]:
        if _looks_like_header_name(line):
            out.append(line)

            parts = [re.sub(r"^[^\w]+|[^\w]+$", "", p) for p in line.split()]
            parts = [p for p in parts if p]
            if parts and parts[0].lower() in HEADER_TITLES and len(parts) >= 2:
                out.append(" ".join(parts[1:]))

    return _normalize_names(out)


############################################################################################
# DETECTION
############################################################################################

def detect_pii_spans(text: str, entities=DEFAULT_ENTITIES, default_min=0.85):
    results = ANALYZER.analyze(text=text, language="en")
    wanted = set(entities)

    kept = []
    for r in results:
        thresh = ENTITY_MIN_SCORE.get(r.entity_type, default_min)
        if r.entity_type in wanted and r.score >= thresh:
            kept.append(r)
    return kept


def detect_pii_transformer(text: str, score_threshold: float = TRANSFORMER_SCORE_THRESHOLD):
    if not TRANSFORMER_READY:
        return []

    inputs = TRANSFORMER_TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=TRANSFORMER_MAX_LENGTH,
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = TRANSFORMER_MODEL(**inputs)

    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1).tolist()
    pred_scores = torch.max(probs, dim=-1).values.tolist()
    id2label = TRANSFORMER_MODEL.config.id2label

    spans = []
    current_entity = None

    for pred_id, score, (start, end) in zip(pred_ids, pred_scores, offset_mapping):
        if start == end:
            continue

        label = id2label[pred_id]

        if label == "O" or score < score_threshold:
            if current_entity is not None:
                spans.append(current_entity)
                current_entity = None
            continue

        if "-" not in label:
            if current_entity is not None:
                spans.append(current_entity)
                current_entity = None
            continue

        prefix, entity_type = label.split("-", 1)
        mapped_entity = MODEL_TO_OLD_ENTITY.get(entity_type)

        if not mapped_entity:
            if current_entity is not None:
                spans.append(current_entity)
                current_entity = None
            continue

        if prefix == "B" or current_entity is None or current_entity["entity_type"] != mapped_entity:
            if current_entity is not None:
                spans.append(current_entity)

            current_entity = {
                "start": start,
                "end": end,
                "entity_type": mapped_entity,
                "score": score,
            }
        else:
            current_entity["end"] = end
            current_entity["score"] = max(current_entity["score"], score)

    if current_entity is not None:
        spans.append(current_entity)

    return spans


def merge_pii_results(presidio_results, transformer_results):
    merged = []

    for r in presidio_results:
        merged.append(SimpleResult(r.start, r.end, r.entity_type, r.score))

    for t in transformer_results:
        t_start = t["start"]
        t_end = t["end"]
        t_type = t["entity_type"]
        t_score = t["score"]

        overlap_found = False

        for m in merged:
            overlaps = not (t_end <= m.start or t_start >= m.end)
            same_type = (t_type == m.entity_type)

            if overlaps and same_type:
                m.start = min(m.start, t_start)
                m.end = max(m.end, t_end)
                m.score = max(m.score, t_score)
                overlap_found = True
                break

        if not overlap_found:
            merged.append(SimpleResult(t_start, t_end, t_type, t_score))

    merged.sort(key=lambda x: (x.start, x.end))
    return merged


def detect_pii_spans_hybrid(text: str, entities=DEFAULT_ENTITIES, default_min=0.85):
    presidio_results = detect_pii_spans(text, entities=entities, default_min=default_min)
    transformer_results = detect_pii_transformer(text)
    return merge_pii_results(presidio_results, transformer_results)


def print_pii(results, text):
    if not results:
        print("No PII detected.")
        return

    for r in results:
        snippet = text[r.start:r.end]
        print(
            f"[{r.entity_type}] '{snippet}' "
            f"(score={r.score:.2f}, span={r.start}-{r.end})"
        )


def redact_text(text: str, results, mask="[REDACTED]"):
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        text = text[:r.start] + mask + text[r.end:]
    return text


############################################################################################
# TERM EXTRACTION
############################################################################################

def extract_pii_terms_from_pdf(
    input_pdf: str | Path,
    *,
    entities=DEFAULT_ENTITIES,
    min_score=0.7,
    max_term_len=80,
) -> list[str]:
    input_pdf = Path(input_pdf)
    doc = fitz.open(str(input_pdf))

    terms: list[str] = []

    for page in doc:
        page_text = page.get_text("text") or ""

        # hybrid detector: old code + transformer
        results = detect_pii_spans_hybrid(page_text, entities=entities, default_min=min_score)

        # optional header-name boost from your old logic
        terms.extend(extract_header_name_candidates(page_text))

        for r in results:
            raw = page_text[r.start:r.end]

            if not is_whole_word_span(page_text, r.start, r.end):
                continue

            raw = re.sub(r"\s+", " ", raw).strip()

            if not raw or len(raw) > max_term_len:
                continue

            if is_allowlisted(raw):
                continue

            if r.entity_type in {
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "US_SSN",
                "STRICT_DATE",
                "MEDICAL_RECORD_NUMBER",
            }:
                terms.append(raw)
                continue

            if r.entity_type == "US_ADDRESS":
                addr_tokens = [_norm_token(t) for t in raw.split() if _norm_token(t)]
                if addr_tokens:
                    terms.append(" ".join(addr_tokens))
                continue

            if r.entity_type == "PERSON":
                person_tokens = _clean_person_tokens(raw)
                if not person_tokens:
                    continue

                full_person = " ".join(person_tokens)

                prefix_text = page_text[:r.start]
                m = re.search(r"(\b[A-Za-z]+\.?)\s*$", prefix_text)
                if m:
                    prefix_raw = m.group(1).strip()
                    prefix_clean = re.sub(r"^[^\w]+|[^\w]+$", "", prefix_raw).strip().lower()
                    if prefix_clean in NAME_TITLES:
                        full_person = f"{prefix_raw} {full_person}"

                terms.append(full_person)
                continue

            if r.entity_type in {"LOCATION", "AGE", "ZIP_CODE"}:
                terms.append(raw)
                continue

    doc.close()
    return _normalize_names(terms)


############################################################################################
# PDF REDACTION
############################################################################################

def _find_exact_token_rects(page, target: str):
    t = _norm_token(target)
    if not t:
        return []

    rects = []
    for w in page.get_text("words"):
        token = _norm_token(w[4])
        if token == t:
            rects.append(fitz.Rect(w[0], w[1], w[2], w[3]))
    return rects


def _find_name_rects_in_line(line_words, target_lower: str):
    tokens_and_rects = []
    cleaned = []

    for w in line_words:
        raw = (w[4] or "").strip()
        rect = fitz.Rect(w[0], w[1], w[2], w[3])
        tokens_and_rects.append((raw, rect))
        cleaned.append(_norm_token(raw))

    target_tokens = [t for t in (_norm_token(x) for x in re.split(r"\s+", target_lower.strip())) if t]
    if not target_tokens:
        return []

    rects = []
    n = len(target_tokens)

    for i in range(0, len(cleaned) - n + 1):
        window = cleaned[i:i + n]
        if window == target_tokens:
            rr = tokens_and_rects[i][1]
            for j in range(i + 1, i + n):
                rr |= tokens_and_rects[j][1]
            rects.append(rr)

    return rects


def redact_pdf_names(
    input_pdf: str | Path,
    output_pdf: str | Path,
    names: list[str],
    *,
    case_insensitive: bool = True,
    redact_color_rgb: tuple[int, int, int] = (0, 0, 0),
) -> dict:
    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    names = _normalize_names(names)
    names = [n for n in names if not is_allowlisted(n)]
    if not names:
        raise ValueError("No names provided to redact.")

    doc = fitz.open(str(input_pdf))

    total_matches = 0
    names_found: dict[str, int] = {n: 0 for n in names}
    fill = tuple(c / 255 for c in redact_color_rgb)

    for page in doc:
        for name in names:
            name = name.strip()
            if not name:
                continue

            if " " in name:
                rects = []
                words = page.get_text("words")
                words.sort(key=lambda w: (w[5], w[6], w[7]))
                target = name.lower() if case_insensitive else name

                current_line = None
                line_words = []

                for w in words + [None]:
                    if w is None:
                        if line_words:
                            rects += _find_name_rects_in_line(line_words, target)
                        break

                    line_id = (w[5], w[6])
                    if current_line is None:
                        current_line = line_id

                    if line_id != current_line:
                        rects += _find_name_rects_in_line(line_words, target)
                        line_words = []
                        current_line = line_id

                    line_words.append(w)
            else:
                rects = _find_exact_token_rects(page, name)

            for r in rects:
                rr = fitz.Rect(r)
                rr.x0 -= 0.5
                rr.y0 -= 0.5
                rr.x1 += 0.5
                rr.y1 += 0.5

                page.add_redact_annot(rr, fill=fill)
                total_matches += 1
                names_found[name] += 1

    for page in doc:
        page.apply_redactions()

    pages_count = doc.page_count
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf))
    doc.close()

    return {
        "pages": pages_count,
        "total_matches": total_matches,
        "names_found": {k: v for k, v in names_found.items() if v > 0},
        "output": str(output_pdf),
    }


############################################################################################
# MAIN
############################################################################################

if __name__ == "__main__":
    INPUT_DIR = Path("input")
    OUTPUT_DIR = Path("output")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {INPUT_DIR.resolve()}")

    for input_pdf in pdfs:
        print(f"Processing: {input_pdf.name}")

        output_pdf = OUTPUT_DIR / f"{input_pdf.stem}_REDACTED.pdf"
        pii_terms = extract_pii_terms_from_pdf(
            input_pdf,
            entities=DEFAULT_ENTITIES,
            min_score=0.6,
        )

        if not pii_terms:
            shutil.copy2(input_pdf, output_pdf)
            print(f"No PII terms found. Copied original to: {output_pdf}")
            continue

        stats = redact_pdf_names(
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            names=pii_terms,
            case_insensitive=True,
        )

        print("Output file:", stats["output"])
        print("Total redactions:", stats["total_matches"])

        if stats["names_found"]:
            print("PII found:")
            for name, count in stats["names_found"].items():
                print(f"  {name}: {count}")
        else:
            print("No visual matches found (possible layout mismatch)")