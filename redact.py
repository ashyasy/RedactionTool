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
    "US_ADDRESS",
    "STRICT_DATE",
    "MEDICAL_RECORD_NUMBER",
    "NPI",
    "INSURANCE",
]

MODEL_TO_OLD_ENTITY = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "ADDRESS": "US_ADDRESS",
    "ZIP_CODE": "ZIP_CODE",
    "CREDIT_CARD": "CREDIT_CARD",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
}

NAME_TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "doctor", "professor"}
HEADER_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "doctor", "professor"}

ENTITY_MIN_SCORE = {
    "PERSON": 0.80,
    "STRICT_DATE": 0.84,
    "US_ADDRESS": 0.70,
    "MEDICAL_RECORD_NUMBER": 0.85,
}

MODEL_DIR = "xlmr_pii_openpii_1m/checkpoint-70000"
TRANSFORMER_MAX_LENGTH = 512
TRANSFORMER_STRIDE = 128
TRANSFORMER_SCORE_THRESHOLD = 0.85

############################################################################################
# PRESIDIO SETUP
############################################################################################

def add_strict_date_recognizer(analyzer: AnalyzerEngine) -> None:
    date_patterns = [
        Pattern(
            name="dob_labeled",
            score=0.99,
            regex=r"(?i)(?:DOB|Date\s+of\s+Birth|Birthdate|Birthday)\s*:?\s*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}",
        ),
        Pattern(
            name="dob_labeled_written",
            score=0.99,
            regex=r"(?i)(?:DOB|Date\s+of\s+Birth|Birthdate|Birthday)\s*:?\s*(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\.?\s+\d{1,2},?\s+\d{4}",
        ),
        Pattern(
            name="mm_dd_yyyy",
            score=0.5,
            regex=r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        ),
        Pattern(
            name="iso_date",
            score=0.5,
            regex=r"\b\d{4}-\d{2}-\d{2}\b",
        ),
        Pattern(
            name="month_day_year",
            score=0.5,
            regex=r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4}\b",
        ),
        Pattern(
            name="month.year",
            score=0.5,
            regex=r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\.?\s+\d{4}\b",
        ),
    ]
    recognizer = PatternRecognizer(
        supported_entity="STRICT_DATE",
        patterns=date_patterns,
        context=["birth", "dob", "born", "birthdate", "birthday", "date of birth"],
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
            name="generic_medical_id1",
            score=0.85,
            regex=r"\b[A-Z]{3,5}-\d{4,10}\b"
        ),
        Pattern(
            name="Letter-Letter-Number",
            score=0.85,
            regex=r"\b[A-Z]{3,5}-\b[A-Z]{2,5}-\d{4,10}\b"
        ),
        Pattern(
            name="generic_medical_id2",
            score=0.5,
            regex=r"\b\d{7,10}\b",
        ),
    ]

    recognizer = PatternRecognizer(
        supported_entity="MEDICAL_RECORD_NUMBER",
        patterns=mrn_patterns,
        context=["medical record", "mrn", "record number"],
    )
    analyzer.registry.add_recognizer(recognizer)


def add_npi_recognizer(analyzer: AnalyzerEngine) -> None:
    npi_patterns = [
        Pattern(
            name="npi_with_label",
            score=0.99,
            regex=r"\bNPI[-:\s]?\d{10}\b",
        ),
        Pattern(
            name="npi_bare",
            score=0.5,
            regex=r"\b\d{10}\b",
        ),
    ]
    recognizer = PatternRecognizer(
        supported_entity="NPI",
        patterns=npi_patterns,
        context=["npi", "national provider", "provider id", "provider number"],
    )
    analyzer.registry.add_recognizer(recognizer)


def add_insurance_recognizer(analyzer: AnalyzerEngine) -> None:
    insurance_patterns = [
        Pattern(
            name="medicare_id",
            score=0.99,
            regex=r"\bMedicare\s*#?\s*[A-Z0-9]{1,4}(?:-[A-Z0-9]{2,4}){1,4}\b",
        ),
        Pattern(
            name="united_healthcare_id",
            score=0.99,
            regex=r"\bUnited\s*Healthcare\s*#?\s*[A-Z]{0,4}-?[A-Z0-9]{4,12}\b",
        ),
        Pattern(
            name="insurance_with_label",
            score=0.95,
            regex=r"\b(?:Aetna|Cigna|Humana|BlueCross|Blue\s+Cross|Blue\s+Shield|Anthem|Molina|Centene|WellCare|Tricare|Medicaid)\s*#?\s*[A-Z0-9]{3,}(?:-[A-Z0-9]{2,}){0,4}\b",
        ),
        Pattern(
            name="insurance_generic_label",
            score=0.95,
            regex=r"\b(?:Insurance|Ins|Policy|Member\s*ID|Member\s*#|Plan\s*ID)\s*[:#]?\s*[A-Z0-9]{3,}(?:-[A-Z0-9]{2,}){0,4}\b",
        ),
    ]

    recognizer = PatternRecognizer(
        supported_entity="INSURANCE",
        patterns=insurance_patterns,
        context=["insurance", "medicare", "medicaid", "policy", "member", "plan", "coverage", "payer"],
    )
    analyzer.registry.add_recognizer(recognizer)


def add_us_address_recognizer(analyzer: AnalyzerEngine) -> None:
    us_address_patterns = [
        Pattern(
            name="street_with_cardinal_direction",
            score=0.92,
            regex=r"\b\d{1,5}\s+(?:North|South|East|West|N|S|E|W|NE|NW|SE|SW)\.?\s+(?:\d{1,3}(?:st|nd|rd|th)|[A-Za-z][A-Za-z.'-]*)(?:\s+[A-Za-z][A-Za-z.'-]*){0,3}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Plaza|Parkway|Pl|Place|Ter|Terrace|Sq|Square)\.?(?:,|\b)",
        ),
        Pattern(
            name="directional_led_street",
            score=0.90,
            regex=r"\b(?:North|South|East|West)\s+(?:\d{1,3}(?:st|nd|rd|th)|[A-Za-z][A-Za-z.'-]*)(?:\s+[A-Za-z][A-Za-z.'-]*){0,3}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Plaza|Pl|Place|Ter|Terrace|Sq|Square)\.?(?:,|\b)",
        ),
        Pattern(
            name="us_address_full",
            score=0.85,
            regex=r"\b\d{1,6}(?:\s+[\w.'-]+){1,12}\s+(?:St|Street|Square|Hall|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Plaza|Parkway|Pl|Broadway|Place|Ter|Terrace|Center)\b(?:\s*,?\s*(?:Apt|Apartment|Unit|Ste|Suite|#)\s*\w+)?(?:\s*,?\s*[A-Za-z .'-]{2,30})?(?:\s*,?\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY))?(?:\s+\d{5}(?:-\d{4})?)?\b",
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
            (?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Plaza|Parkway|Pl|Place|Ter|Terrace|Broadway|Center|Sq|Square)\.?
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
            (?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Plaza|Parkway|Pl|Place|Ter|Terrace|Broadway|Center|Sq|Square)\.?
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
                \s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Plaza|Parkway|Pl|Place|Ter|Terrace|Sq|Square)\.?
            )?
            \b
            """.strip(),
        ),
        Pattern(
            name="numbered_broadway",
            score=0.95,
            regex=r"\b\d{1,6}\s+(?:Broadway|Boardwalk|Commons|Plaza|Promenade|Esplanade|Mall|Galleria|Highway|Freeway|Turnpike|Causeway|Expressway|Bypass)\b",
        ),
    ]

    recognizer = PatternRecognizer(
        supported_entity="US_ADDRESS",
        patterns=us_address_patterns,
        context=["address", "st", "broadway", "street", "ave", "road", "blvd", "apt", "suite", "zip"],
    )
    analyzer.registry.add_recognizer(recognizer)


def add_phone_recognizer(analyzer: AnalyzerEngine) -> None:
    recognizer = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[
            Pattern("us_phone_parens",  score=0.75, regex=r"\(\d{3}\)\s*\d{3}[-.\s]\d{4}\b"),
            Pattern("us_phone_dashes",  score=0.65, regex=r"\b\d{3}[-]\d{3}[-]\d{4}\b"),
            Pattern("us_phone_country", score=0.80, regex=r"\+1[\s\-]?\(?\d{3}\)?\s*\d{3}[-.\s]\d{4}\b"),
        ],
    )
    analyzer.registry.add_recognizer(recognizer)


def add_ssn_recognizer(analyzer: AnalyzerEngine) -> None:
    recognizer = PatternRecognizer(
        supported_entity="US_SSN",
        patterns=[
            Pattern("ssn_dashes",  score=0.99, regex=r"\b\d{3}-\d{2}-\d{4}\b"),
            Pattern("ssn_spaces",  score=0.95, regex=r"\b\d{3} \d{2} \d{4}\b"),
            Pattern("ssn_bare",    score=0.85, regex=r"\b\d{9}\b"),
        ],
        context=["ssn", "social security", "social security number"],
    )
    analyzer.registry.add_recognizer(recognizer)

def add_person_pattern_recognizer(analyzer: AnalyzerEngine) -> None:

    # Surname: 4+ chars, Title-Case — rules out short words like "Be", "In", "Per"
    surname = r"[A-Z][a-z'-]{3,}"
    month_firstname = r"(?:April|May|June|August)"
    recognizer = PatternRecognizer(
        supported_entity="PERSON",
        patterns=[
            Pattern("month_as_firstname", score=0.92,
                    regex=rf"\b{month_firstname}\s+{surname}\b"),
        ],
        context=[
            "patient", "name", "emergency contact", "next of kin",
            "guarantor", "referring", "attending", "provider", "author",
            "signed by", "ordered by", "contact", "subscriber", "insured",
        ],
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
    add_ssn_recognizer(analyzer)
    add_phone_recognizer(analyzer)
    add_mrn_recognizer(analyzer)
    add_npi_recognizer(analyzer)
    add_insurance_recognizer(analyzer)
    add_person_pattern_recognizer(analyzer)

    # Extend the existing SpacyRecognizer's context in-place so Presidio's
    # default entity mappings and settings are preserved.
    extra_context = [
        "emergency", "contact", "emergency contact", "next of kin",
        "guarantor", "subscriber", "insured", "guardian", "parent",
        "responsible party", "ordered by", "signed by", "referring",
        "referred", "clinician", "nurse", "staff",
    ]
    for rec in analyzer.registry.recognizers:
        if rec.__class__.__name__ == "SpacyRecognizer":
            rec.context = list(set((rec.context or []) + extra_context))
        if getattr(rec, "supported_entities", None) == ["PHONE_NUMBER"]:
            rec.context = list(set((rec.context or []) + ["tel", "telephone", "fax", "mobile", "cell", "contact"]))

    return analyzer


ANALYZER = build_presidio()


############################################################################################
# TRANSFORMER SETUP
############################################################################################

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    MODEL = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    MODEL.to(_DEVICE)
    MODEL.eval()
    TRANSFORMER_READY = True
    print(f"Loaded transformer model from: {MODEL_DIR}")
    print(f"Transformer device: {_DEVICE}")
except Exception as exc:
    TOKENIZER = None
    MODEL = None
    TRANSFORMER_READY = False
    print(f"Transformer model not loaded: {exc}")
    print("Continuing in Presidio-only mode.")


############################################################################################
# ALLOWLIST / HELPERS
############################################################################################

ALLOWLIST_EXACT = {
    "inc", "llc", "ltd", "md", "at", "needs",
    "university", "department",
    "texas", "california", "student",
    "usa", "united", "states", "united states", "united states of america",
    "new york", "new jersey", "new mexico", "new hampshire",
    "west virginia", "north carolina", "south carolina",
    "north dakota", "south dakota", "rhode island",
    "weeks ago", "months ago", "minutes",
    "date", "date:", "birth", "email", "biographies",
    "dyspnea", "epigastric", "lumbosacral", "lumbo", "musculoskeletal", "dysthymic",
    "dysthymia", "dysphoria", "dysphagia", "dysarthria", "dysuria", "dyslipidemia",
    "metoprolol", "metorolol", "glipizide", "morphine", "oxycodone", "hydrocodone",
    "ferrous", "sulfate", "ferrous sulfate", "ferric", "iron", "folic", "folate",
    "leopold", "leopold maneuver",
    "ibuprofen", "acetaminophen", "naproxen", "diltiazem", "verapamil", "digoxin",
    "mmse", "gcs", "moca", "phq", "gad", "bmi", "ekg", "ecg", "mri", "ct", "xray",
    "orthopedist", "cardiologist", "neurologist", "radiologist", "hospitalist",
    "pulmonologist", "rheumatologist", "endocrinologist", "nephrologist", "oncologist",
    "bone", "mineral", "density", "calcium", "phosphorus", "potassium", "sodium",
    "tamsulosin", "metformin", "lisinopril", "atorvastatin", "amlodipine",
    "omeprazole", "metoprolol", "albuterol", "gabapentin", "sertraline",
    "losartan", "furosemide", "pantoprazole", "escitalopram", "hydrochlorothiazide",
    "pneumovax", "clopidogrel", "warfarin", "prednisone", "insulin", "amoxicillin",
    "azithromycin", "levothyroxine", "aspirin", "kcl", "influenza", "tdap",
    "whipple", "appendectomy", "cholecystectomy", "colonoscopy", "bronchoscopy",
    "endoscopy", "biopsy", "catheterization", "intubation", "dialysis",
    "the", "is", "in", "ist", "name", "medical center", "ED", "ER", "hospital", "clinic",
    "PO", "SQ", "SpO2", "address",

    "st", "rd", "ave", "blvd", "ln", "ct", "pl", "pkwy", "ter", "cir", "sq", "way", "mg",

    "united", "healthcare", "united healthcare",
    "medicare", "medicaid", "aetna", "cigna", "humana", "anthem", "aetna student health",
    "insurance", "tricare", "molina", "primera",
    "quick reference", "quick", "reference",
    "table of contents", "executive summary", "summary", "overview",
    "introduction", "appendix", "glossary", "guide", "manual",
    "report", "form", "document", "section", "note", "notice",
    "instructions", "information", "disclaimer", "confidential",
    # medical document section headers
    "physical exam", "physical examination", "physical", "exam", "examination",
    "vital signs", "vital", "signs", "chief complaint", "chief",
    "history", "review", "systems", "systems review", "review of systems",
    "assessment", "plan", "impression", "findings", "results",
    "diagnosis", "diagnoses", "treatment", "procedure", "procedures",
    "medications", "medication", "allergies", "allergy",
    "social history", "family history", "past history", "medical history",
    "surgical history", "history of present illness", "present illness",
    "laboratory", "radiology", "imaging", "pathology",
    "discharge", "admission", "follow", "followup", "follow up",
    "objective", "subjective", "recommendation", "recommendations",
    "signature", "signed", "attending", "resident", "intern",
    "consultation", "referral", "orders", "order",
    "eliquis"
}

DATE_PATTERNS = [
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?$", re.I),
    re.compile(r"^\d{1,2},?$"),
    re.compile(r"^\d{4}$"),
]

_WORD_CH = re.compile(r"[A-Za-z0-9_]")


class SimpleResult:
    def __init__(self, start, end, entity_type, score, source="unknown"):
        self.start = start
        self.end = end
        self.entity_type = entity_type
        self.score = score
        self.source = source


def should_keep_term(term: str) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    if len(t) == 1 and t.isalpha():
        return False
    return True


_MEDICAL_SUFFIXES = re.compile(
    r"(oscopy|ectomy|otomy|ostomy|plasty|graphy|gram|ology|itis|osis|pathy|therapy|tomy|scopy|centesis|rrhaphy|pexy|desis|lysis|genesis|trophy|ulosin|emia|uria|algia|plegia|paresis|phobia|philia|mania)$",
    re.IGNORECASE,
)


def _is_medical_term(word: str) -> bool:
    return bool(_MEDICAL_SUFFIXES.search(word.strip()))


_MONTH_FIRSTNAMES = {"april", "may", "june", "august"}


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
    s = (s or "").strip()
    # If a label is fused with a name via colon (e.g. "Contact:Elaine"), take only the part after the last colon
    if ":" in s:
        s = s.split(":")[-1]
    return re.sub(r"^[^\w]+|[^\w]+$", "", s).lower()


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

        if _is_medical_term(clean):
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

    # Require at least 2 words — single words are section headers, not names
    if len(parts) < 2:
        return False

    for p in parts:
        if len(p) < 2:
            return False
        if not re.match(r"^[A-Z][a-z.'-]*$", p):
            return False
        if is_allowlisted(p):
            return False
        if _is_medical_term(p):
            return False
    return True

_LABEL_NAME_RE = re.compile(
    # Label must be at line start or after newline (case-insensitive label only)
    r"(?:^|\n)\s*(?i:patient(?:\s+name)?|provider|guarantor|guardian|parent(?:/guardian)?|"
    r"next\s+of\s+kin|emergency\s+contact|subscriber|insured|attending|referring|"
    r"ordered\s+by|signed\s+by|clinician|nurse|author|evaluator|therapist|"
    r"responsible\s+party)\s*[:/]\s*"
    # Name group: Title-Case words (supports hyphenated names like Soo-Yeon)
    r"((?:[A-Z][a-z]*(?:-[A-Z][a-z]*)*)(?:[^\S\n]+[A-Z][a-z]*(?:-[A-Z][a-z]*)*){0,3})",
    re.MULTILINE,
)


def extract_label_name_candidates(page_text: str) -> list[str]:
    """Extract names that immediately follow known PII label words."""
    out = []
    for m in _LABEL_NAME_RE.finditer(page_text):
        name = m.group(1).strip()
        if not name or is_allowlisted(name) or _is_medical_term(name.split()[-1]):
            continue
        out.append(name)
    return _normalize_names(out)


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


def _decode_transformer_chunk(
    offsets: list[tuple[int, int]],
    labels: list[int],
    scores: list[float],
    id2label: dict[int, str],
) -> list[SimpleResult]:
    spans: list[SimpleResult] = []
    current: SimpleResult | None = None

    for pred_id, score, (start, end) in zip(labels, scores, offsets):
        if start == end:
            continue

        label = id2label[int(pred_id)]
        if label == "O" or score < TRANSFORMER_SCORE_THRESHOLD:
            if current is not None:
                spans.append(current)
                current = None
            continue

        if "-" not in label:
            if current is not None:
                spans.append(current)
                current = None
            continue

        prefix, entity_type = label.split("-", 1)
        mapped = MODEL_TO_OLD_ENTITY.get(entity_type)
        if not mapped:
            if current is not None:
                spans.append(current)
                current = None
            continue

        if prefix == "B" or current is None or current.entity_type != mapped:
            if current is not None:
                spans.append(current)
            current = SimpleResult(start, end, mapped, float(score), source="transformer")
        else:
            current.end = end
            current.score = max(current.score, float(score))

    if current is not None:
        spans.append(current)

    return spans

def detect_pii_spans_transformer(text: str, entities: list[str] = DEFAULT_ENTITIES) -> list[SimpleResult]:
    if not TRANSFORMER_READY:
        return []

    # Only keep transformer results whose mapped entity type was requested
    wanted = set(entities)

    encoded = TOKENIZER(
        text,
        truncation=True,
        max_length=TRANSFORMER_MAX_LENGTH,
        stride=TRANSFORMER_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )

    offset_mappings = encoded.pop("offset_mapping")
    if "overflow_to_sample_mapping" in encoded:
        encoded.pop("overflow_to_sample_mapping")

    encoded = {k: v.to(_DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = MODEL(**encoded)

    probs = torch.softmax(outputs.logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
    pred_scores = torch.max(probs, dim=-1).values.cpu().tolist()
    id2label = MODEL.config.id2label

    all_spans: list[SimpleResult] = []
    for chunk_idx in range(len(offset_mappings)):
        offsets = [tuple(map(int, pair)) for pair in offset_mappings[chunk_idx].tolist()]
        chunk_spans = _decode_transformer_chunk(offsets, pred_ids[chunk_idx], pred_scores[chunk_idx], id2label)
        all_spans.extend(chunk_spans)

    return [s for s in all_spans if s.entity_type in wanted]


def merge_pii_results(presidio_results, transformer_results: list[SimpleResult]) -> list[SimpleResult]:
    merged: list[SimpleResult] = [
        SimpleResult(r.start, r.end, r.entity_type, r.score, source="presidio")
        for r in presidio_results
    ]

    for t in transformer_results:
        overlap_found = False
        for m in merged:
            overlaps = not (t.end <= m.start or t.start >= m.end)
            if overlaps and t.entity_type == m.entity_type:
                m.start = min(m.start, t.start)
                m.end = max(m.end, t.end)
                avg = (m.score + t.score) / 2
                m.score = min(1.0, avg + 0.15)
                m.source = "both"
                overlap_found = True
                break
        if not overlap_found:
            merged.append(SimpleResult(t.start, t.end, t.entity_type, t.score, source="transformer"))

    for m in merged:
        _log(f"  [merged/{m.source}] [{m.entity_type}] score={m.score:.2f}")
    
    merged.sort(key=lambda x: (x.start, x.end))
    return merged


def detect_pii_spans_hybrid(text: str, entities=DEFAULT_ENTITIES, default_min=0.85, USE_PRESIDIO=True, USE_TRANSFORMER=True) -> list[SimpleResult]:
    presidio_results = detect_pii_spans(text, entities=entities, default_min=default_min) if USE_PRESIDIO else []
    transformer_results = detect_pii_spans_transformer(text, entities=entities) if USE_TRANSFORMER else []

    _log("  [Presidio]")
    for r in presidio_results:
        _log(f"    [{r.entity_type}] score={r.score:.2f} raw={repr(text[r.start:r.end])}")
    _log("  [Transformer]")
    for t in transformer_results:
        _log(f"    [{t.entity_type}] score={t.score:.2f} raw={repr(text[t.start:t.end])}")

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

_DEBUG_LOG = Path("/tmp/redact_debug.log")


def _log(msg: str) -> None:
    with _DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def extract_pii_terms_from_pdf(
    input_pdf: str | Path,
    *,
    entities=DEFAULT_ENTITIES,
    min_score=0.7,
    max_term_len=80,
    use_transformer: bool = True,
) -> list[str]:
    input_pdf = Path(input_pdf)
    doc = fitz.open(str(input_pdf))

    _DEBUG_LOG.write_text("", encoding="utf-8")  # clear log at start of each run
    _log(f"=== {input_pdf.name} ===")

    terms: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text") or ""
        
        page_text = re.sub(r"\s*\|\s*", "\n", page_text)

        results = detect_pii_spans_hybrid(page_text, entities=entities, default_min=min_score, USE_TRANSFORMER=use_transformer)

        _log(f"\n--- Page {page_num} ({len(results)} detections) ---")

        # optional header-name boost — only when PERSON redaction is enabled
        if "PERSON" in set(entities):
            header_candidates = extract_header_name_candidates(page_text)
            for h in header_candidates:
                _log(f"  [HEADER]  term={h!r}")
            terms.extend(header_candidates)

            label_candidates = extract_label_name_candidates(page_text)
            for h in label_candidates:
                _log(f"  [LABEL]  term={h!r}")
            terms.extend(label_candidates)

        for r in results:
            raw = page_text[r.start:r.end]
            source = getattr(r, "source", "presidio")
            snippet = repr(raw[:60])

            trimmed_end = r.end - (len(raw) - len(raw.rstrip()))
            if not is_whole_word_span(page_text, r.start, trimmed_end):
                _log(f"  [SKIP not-whole-word] {r.entity_type} score={r.score:.2f} src={source} raw={snippet}")
                continue

            if "\n" in raw and r.entity_type != "STRICT_DATE":
                raw = raw.split("\n")[0].strip()
                if not raw:
                    _log(f"  [SKIP empty-after-newline-trim] {r.entity_type} score={r.score:.2f} src={source}")
                    continue

            raw = re.sub(r"\s+", " ", raw).strip()

            raw = re.sub(r"^[•·▪▸◦‣⁃–—\s]+|[•·▪▸◦‣⁃–—\s]+$", "", raw).strip()

            if not raw or len(raw) > max_term_len:
                _log(f"  [SKIP len] {r.entity_type} score={r.score:.2f} src={source} raw={snippet}")
                continue

            if is_allowlisted(raw):
                _log(f"  [SKIP allowlist] {r.entity_type} score={r.score:.2f} src={source} raw={repr(raw)}")
                continue


            if r.entity_type == "PERSON" and _is_medical_term(raw.split()[-1]):
                _log(f"  [SKIP medical-term] {r.entity_type} score={r.score:.2f} src={source} raw={repr(raw)}")
                continue

            if r.entity_type == "PERSON" and raw.strip().lower() in _MONTH_FIRSTNAMES:
                _log(f"  [SKIP standalone-month] {r.entity_type} score={r.score:.2f} src={source} raw={repr(raw)}")
                continue

            if r.entity_type == "PERSON":
                words = raw.strip().split()
                if words and words[0].lower() in _MONTH_FIRSTNAMES:
                    has_digit_word = any(w and w[0].isdigit() for w in words)
                    not_all_titlecase = not all(w[0].isupper() for w in words if w and w[0].isalpha())
                    if has_digit_word or not_all_titlecase:
                        _log(f"  [SKIP month-not-titlecase] {r.entity_type} score={r.score:.2f} src={source} raw={repr(raw)}")
                        continue

            if r.entity_type == "NPI":
                npi_match = re.search(r'\d{10}', raw)
                if npi_match:
                    _log(f"  [KEEP] NPI score={r.score:.2f} src={source} raw={repr(raw)} → term={repr(npi_match.group())}")
                    terms.append(npi_match.group())
                else:
                    _log(f"  [SKIP no-npi-in-span] NPI score={r.score:.2f} src={source} raw={repr(raw)}")
                continue

            if r.entity_type in {
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "US_SSN",
                "MEDICAL_RECORD_NUMBER",
            }:
                _log(f"  [KEEP] {r.entity_type} score={r.score:.2f} src={source} term={repr(raw)}")
                terms.append(raw)
                continue

            if r.entity_type == "INSURANCE":
                # Strip insurer name label — keep only the ID/number portion
                id_match = re.search(
                    r'(?:Medicare(?:\s*ID)?|United\s*Healthcare|Aetna|Cigna|Humana|Blue\s*Cross(?:\s*Blue\s*Shield)?|Anthem|Molina|Centene|WellCare|Tricare|Medicaid|Insurance|Ins|Policy|Member\s*(?:ID|#)|Plan\s*ID)\s*[:#]?\s*([A-Z0-9]{2,}(?:\s*-\s*[A-Z0-9]{2,})+|(?=[A-Z0-9]*\d)[A-Z0-9]{5,})',
                    raw,
                    re.IGNORECASE,
                )

                if id_match:
                    _log(f"  [KEEP] INSURANCE score={r.score:.2f} src={source} raw={repr(raw)} → term={repr(id_match.group(1))}")
                    terms.append(id_match.group(1))
                else:
                    _log(f"  [SKIP no-id-in-span] INSURANCE score={r.score:.2f} src={source} raw={repr(raw)}")
                continue

            if r.entity_type == "STRICT_DATE":
                # Strip any leading label (e.g. "DOB: ") — keep only the date value
                date_match = re.search(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\.?\s+\d{1,2},?\s+\d{4}|\b\d{4}-\d{2}-\d{2}", raw)
                if date_match:
                    _log(f"  [KEEP] STRICT_DATE score={r.score:.2f} src={source} raw={repr(raw)} → term={repr(date_match.group())}")
                    terms.append(date_match.group())
                else:
                    _log(f"  [SKIP no-date-in-span] STRICT_DATE score={r.score:.2f} src={source} raw={repr(raw)}")
                continue

            if r.entity_type == "US_ADDRESS":
                # Skip time strings like "30 pm", "00 am"
                if re.fullmatch(r"\d{1,2}\s*[aApP][mM]", raw.strip()):
                    _log(f"  [SKIP time-string] US_ADDRESS score={r.score:.2f} src={source} raw={repr(raw)}")
                    continue
                addr_tokens = [_norm_token(t) for t in raw.split() if _norm_token(t)]
                if addr_tokens:
                    term = " ".join(addr_tokens)
                    _log(f"  [KEEP] US_ADDRESS score={r.score:.2f} src={source} term={repr(term)}")
                    terms.append(term)
                else:
                    _log(f"  [SKIP empty-addr] US_ADDRESS score={r.score:.2f} src={source} raw={repr(raw)}")
                continue

            if r.entity_type == "PERSON":
                person_tokens = _clean_person_tokens(raw)
                if not person_tokens:
                    _log(f"  [SKIP empty-person] PERSON score={r.score:.2f} src={source} raw={repr(raw)}")
                    continue

                full_person = " ".join(person_tokens)

                prefix_text = page_text[:r.start]
                m = re.search(r"(\b[A-Za-z]+\.?)\s*$", prefix_text)
                if m:
                    prefix_raw = m.group(1).strip()
                    prefix_clean = re.sub(r"^[^\w]+|[^\w]+$", "", prefix_raw).strip().lower()
                    if prefix_clean in NAME_TITLES:
                        full_person = f"{prefix_raw} {full_person}"

                _log(f"  [KEEP] PERSON score={r.score:.2f} src={source} term={repr(full_person)}")
                terms.append(full_person)
                continue

            if r.entity_type in {"LOCATION", "AGE", "ZIP_CODE"}:
                _log(f"  [KEEP] {r.entity_type} score={r.score:.2f} src={source} term={repr(raw)}")
                terms.append(raw)
                continue

            _log(f"  [SKIP unhandled] {r.entity_type} score={r.score:.2f} src={source} raw={repr(raw)}")

    doc.close()
    result = _normalize_names(terms)
    _log(f"\n=== Final terms ({len(result)}) ===")
    for t in result:
        _log(f"  {repr(t)}")
    return result


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


def _find_phrase_rects_crossline(page, phrase: str) -> list[fitz.Rect]:
    """Match a phrase across line boundaries by comparing normalized token sequences."""
    target_tokens = [t for t in (_norm_token(x) for x in re.split(r"\s+", phrase.strip())) if t]
    if not target_tokens:
        return []

    all_words = sorted(page.get_text("words"), key=lambda w: (w[5], w[6], w[7]))
    cleaned = [(_norm_token(w[4]), fitz.Rect(w[0], w[1], w[2], w[3])) for w in all_words]
    cleaned = [(t, r) for t, r in cleaned if t]

    n = len(target_tokens)
    rects = []
    for i in range(len(cleaned) - n + 1):
        if [t for t, _ in cleaned[i:i + n]] == target_tokens:
            combined = cleaned[i][1]
            for _, r in cleaned[i + 1:i + n]:
                combined |= r
            rects.append(combined)
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
                rr.y0 -= 0.1
                rr.x1 += 0.5
                rr.y1 += 0.1

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


def redact_pdf_names_term(
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
                rects = _find_phrase_rects_crossline(page, name)
            else:
                rects = _find_exact_token_rects(page, name)

            for r in rects:
                rr = fitz.Rect(r)
                rr.x0 -= 0.5
                rr.y0 -= 0.1
                rr.x1 += 0.5
                rr.y1 += 0.1

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