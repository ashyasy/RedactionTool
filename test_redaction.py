"""
Redaction test suite.

Creates a synthetic medical PDF, runs the full redaction pipeline, then
verifies that PII is gone and non-PII is preserved.

Run:
    pytest test_redaction.py -v --disable-warnings -W ignore::DeprecationWarning
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import fitz
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(lines: list[str], path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 60
    for line in lines:
        if y > 740:
            page = doc.new_page(width=612, height=792)
            y = 60
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    doc.save(str(path))
    doc.close()


def _pdf_text(path: Path) -> str:
    doc = fitz.open(str(path))
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pipeline():
    from redact import extract_pii_terms_from_pdf, redact_pdf_names, DEFAULT_ENTITIES
    return extract_pii_terms_from_pdf, redact_pdf_names, DEFAULT_ENTITIES


# ---------------------------------------------------------------------------
# Test data — (id, pdf_line, term_to_check)
# ---------------------------------------------------------------------------

SHOULD_REDACT = [
    # ── Names ────────────────────────────────────────────────────────────────
    ("patient name",              "Patient: Jonathan Harker",                          "Jonathan Harker"),
    ("patient name spaced",       "Patient Name: Emily Clarke",                        "Emily Clarke"),
    ("dr prefix",                 "Attending: Dr. Marcus Webb",                        "Marcus Webb"),
    ("name with middle initial",  "Patient: Sarah J. Thompson",                        "Sarah"),
    ("two-word name inline",      "Referred by James Morrison MD",                     "James Morrison"),
    ("name after colon no space", "Provider:Rachel Green",                             "Rachel Green"),
    ("continuously missed name",              "April Murphy",                                     "April Murphy"),
    ("april in table",             "Name Date Changes April Murphy Aug. 2019 Extensive",        "April Murphy"),
    ("parent/guardian",           "Parent/Guardian: Soo-Yeon Park",                         "Soo-Yeon Park"),
   
    ("white name vs color", "Seen by Dr White",         "White"),
    ("stacked pii", "John Doe | 123-45-6789 | john@email.com", "John Doe"),
    ("stacked pii", "John Doe | 123-45-6789 | john@email.com", "123-45-6789"),
    ("stacked pii", "John Doe | 123-45-6789 | john@email.com", "john@email.com"),

    # ── Emergency / next of kin ───────────────────────────────────────────────
    ("emergency contact nospace", "Emergency Contact:Mina Murray",                     "Mina Murray"),
    ("emergency contact space",   "Emergency Contact: Arthur Holmwood",                "Arthur Holmwood"),
    ("next of kin",               "Next of Kin: Lucy Westenra",                        "Lucy Westenra"),
    ("guarantor",                 "Guarantor: Renfield Smith",                         "Renfield Smith"),

    # ── SSN ──────────────────────────────────────────────────────────────────
    ("ssn dashes",                "SSN: 123-45-6789",                                  "123-45-6789"),
    ("ssn spaces",                "Social Security: 234 56 7890",                      "234 56 7890"),

    # ── Phone ────────────────────────────────────────────────────────────────
    ("phone parens",              "Phone: (806) 555-0192",                             "555-0192"),
    ("phone dots",                "Cell: 806.555.0193",                                "806.555.0193"),
    ("fax number",                "Fax: (806) 555-0195",                               "555-0195"),
    ("tel prefix",                "Tel: (773) 555-0344",                               "555-0344"),
    ("phone pipe header",         "550 Maternity Lane, Suite 300, Nashville, TN 37203 | (615) 555-0630", "555-0630"),
    ("telephone prefix",          "Telephone: (312) 555-9900",                         "555-9900"),
    ("mobile prefix",             "Mobile: 214-555-7722",                              "555-7722"),

    # ── Email ────────────────────────────────────────────────────────────────
    ("email basic",               "Email: jharker@example.com",                        "jharker@example.com"),
    ("email subdomain",           "Contact: patient.doe@mail.hospital.org",            "patient.doe@mail.hospital.org"),

    # ── Date of birth ────────────────────────────────────────────────────────
    ("dob slash",                 "DOB: 03/15/1958",                                   "03/15/1958"),
    ("dob labeled written",       "Date of Birth: January 5, 1962",                   "January 5, 1962"),
    ("dob iso",                   "Birthdate: 1975-08-22",                             "1975-08-22"),
    ("dob table cell",            "DOB:\t07/04/1980",                                  "07/04/1980"),

    # ── Address ──────────────────────────────────────────────────────────────
    ("street address",            "Address: 452 Elm Street, Lubbock TX 79401",         "452 Elm Street"),
    ("avenue address",            "Home: 18 Oak Avenue, Dallas TX 75201",              "18 Oak Avenue"),
    ("apt address",               "Mailing: 301 Maple Dr Apt 4B, Austin TX",           "301 Maple Dr"),

    # ── MRN ──────────────────────────────────────────────────────────────────
    ("mrn labeled",               "MRN: 8834291",                                      "8834291"),
    ("mrn prefixed",              "Record: MRN-2291847",                               "2291847"),

    # ── Insurance ────────────────────────────────────────────────────────────
    ("medicare id",               "Medicare # 1EG4-TE5-MK72",                          "1EG4-TE5-MK72"),
    ("insurance member id",       "Member ID: XYZ987654",                              "XYZ987654"),
    ("hyphenated name",        "Patient: Anne-Marie Johnson",            "Anne-Marie Johnson"),
    ("apostrophe name",        "Patient: O'Connor Patrick",              "O'Connor Patrick"),
    ("all caps name",          "PATIENT: MICHAEL SCOTT",                 "MICHAEL SCOTT"),
    ("lowercase name",         "patient: jim halpert",                   "jim halpert"),
    ("name with suffix",       "Patient: Robert Downey Jr.",             "Robert Downey"),
    ("three part name",        "Patient: Mary Kate Olsen",               "Mary Kate Olsen"),
    ("name with title inline", "Seen by Dr John Watson today",           "John Watson"),

    # ── Phones (edge formats) ───────────────────────────────────────────────
    ("phone no separators",    "Phone: 8065551234",                      "8065551234"),
    ("phone with country",     "Phone: +1 (806) 555-7777",               "555-7777"),
    ("phone ext",              "Phone: 806-555-8888 ext 22",             "806-555-8888"),

    # ── Emails (edge cases) ─────────────────────────────────────────────────
    ("email plus",             "Email: test+alias@gmail.com",            "test+alias@gmail.com"),
    ("email caps",             "Email: JOHN.DOE@EXAMPLE.COM",            "JOHN.DOE@EXAMPLE.COM"),

    # ── Dates (harder) ──────────────────────────────────────────────────────
    ("dob dots",               "DOB: 03.15.1958",                        "03.15.1958"),
    ("dob short year",         "DOB: 03/15/58",                          "03/15/58"),

    # ── Addresses (messy) ───────────────────────────────────────────────────
    ("address lowercase",      "Address: 88 pine street houston tx",     "88 pine street"),
    ("address no comma",       "Address: 12 Sunset Blvd Los Angeles CA", "12 Sunset Blvd"),
    ("address unit",           "Address: 400 Main St Unit 12",           "400 Main St"),

    # ── MRN variants ────────────────────────────────────────────────────────
    ("mrn spaced",             "MRN : 4455667",                          "4455667"),
    ("mrn text inline",        "Patient MRN 9988776 recorded",           "9988776"),

    # ── Insurance variations ────────────────────────────────────────────────
    ("insurance dashed",       "Policy: ABC-123-XYZ",                    "ABC-123-XYZ"),
    ("insurance numeric",      "Insurance ID: 999888777",                "999888777"),

    # ── OCR-like spacing issues ─────────────────────────────────────────────
]


SHOULD_NOT_REDACT = [
    # ── Medications ──────────────────────────────────────────────────────────
    ("tamsulosin",          "Medications: Tamsulosin 0.4 mg daily",       "Tamsulosin"),
    ("metformin",           "Medications: Metformin 500 mg BID",          "Metformin"),
    ("lisinopril",          "Medications: Lisinopril 10 mg daily",        "Lisinopril"),
    ("atorvastatin",        "Medications: Atorvastatin 40 mg nightly",    "Atorvastatin"),
    ("albuterol",           "Medications: Albuterol inhaler PRN",         "Albuterol"),
    ("omeprazole",          "Medications: Omeprazole 20 mg daily",        "Omeprazole"),
    ("gabapentin",          "Medications: Gabapentin 300 mg TID",         "Gabapentin"),
    ("sertraline",          "Medications: Sertraline 50 mg daily",        "Sertraline"),
    ("furosemide",          "Medications: Furosemide 40 mg daily",        "Furosemide"),
    ("kcl",                 "Medications: KCl 20 mEq daily",              "KCl"),
    ("pneumovax",           "Vaccines: Pneumovax 23 administered",        "Pneumovax"),
    ("aspirin",             "Medications: Aspirin 81 mg daily",           "Aspirin"),
    ("warfarin",            "Medications: Warfarin 5 mg daily",           "Warfarin"),
    ("prednisone",          "Medications: Prednisone 10 mg daily",        "Prednisone"),
    ("insulin",             "Medications: Insulin glargine 20 units QHS", "Insulin"),
    ("amoxicillin",         "Medications: Amoxicillin 500 mg TID",        "Amoxicillin"),
    ("azithromycin",        "Medications: Azithromycin 250 mg daily",     "Azithromycin"),
    ("levothyroxine",       "Medications: Levothyroxine 50 mcg daily",    "Levothyroxine"),
    ("clopidogrel",         "Medications: Clopidogrel 75 mg daily",       "Clopidogrel"),
    ("amlodipine",          "Medications: Amlodipine 5 mg daily",         "Amlodipine"),
    ("metoprolol",          "Medications: Metoprolol 25 mg BID",          "Metoprolol"),
    ("glipizide",           "Medications: Glipizide 5 mg daily",          "Glipizide"),
    ("morphine",            "Medications: Morphine 2 mg IV PRN",          "Morphine"),
    ("oxycodone",           "Medications: Oxycodone 5 mg q4h PRN",        "Oxycodone"),
    ("ibuprofen",           "Medications: Ibuprofen 400 mg TID",          "Ibuprofen"),

    # ── Medical acronyms ─────────────────────────────────────────────────────
    ("mmse",                "Cognitive: MMSE score 28/30",                "MMSE"),
    ("bmi",                 "BMI: 24.5 kg/m2",                            "BMI"),
    ("ekg",                 "EKG: normal sinus rhythm",                   "EKG"),

    # ── Specialist titles (job roles, not names) ──────────────────────────────
    ("orthopedist role",    "Referred to Orthopedist for follow-up",      "Orthopedist"),
    ("cardiologist role",   "Seen by Cardiologist on 11/03/2023",         "Cardiologist"),

    # ── Time strings ─────────────────────────────────────────────────────────
    ("time am",             "Admitted at 8:30 am per nursing notes",      "30 am"),
    ("time pm",             "Discharge time: 2:00 pm",                    "00 pm"),

    # ── Vaccines ─────────────────────────────────────────────────────────────
    ("flu shot",            "Vaccines: Influenza vaccine given",          "Influenza"),
    ("tdap",                "Vaccines: Tdap booster administered",        "Tdap"),

    # ── Geographic / country names ────────────────────────────────────────────
    ("united states",       "Born in the United States",                  "United States"),
    ("new york",            "Relocated from New York to Texas",           "New York"),

    # ── Medical terms / symptoms ─────────────────────────────────────────────
    ("dyspnea",             "Chief Complaint: dyspnea on exertion",       "dyspnea"),
    ("epigastric",          "Pain: epigastric tenderness noted",          "epigastric"),
    ("musculoskeletal",     "MSK: musculoskeletal exam normal",           "musculoskeletal"),
    

    # ── Section headers ──────────────────────────────────────────────────────
    ("physical exam",       "Physical Exam:",                             "Physical Exam"),
    ("history header",      "History of Present Illness:",                "History"),
    ("assessment header",   "Assessment and Plan:",                       "Assessment"),
    ("allergies header",    "Allergies: Penicillin",                      "Allergies"),
    ("medications hdr",     "Current Medications:",                       "Medications"),
    ("vital signs",         "Vital Signs: BP 120/80",                     "Vital Signs"),
    ("review systems",      "Review of Systems:",                         "Review"),
    ("social history",      "Social History: non-smoker",                 "Social History"),
    ("family history",      "Family History: no known cardiac disease",   "Family History"),
    ("discharge summary",   "Discharge Summary",                          "Discharge"),
    ("laboratory",          "Laboratory Results:",                        "Laboratory"),
    ("radiology",           "Radiology: chest X-ray normal",              "Radiology"),
    ("diagnosis header",    "Diagnosis: Type 2 Diabetes Mellitus",        "Diagnosis"),

    # ── Facility / org names ─────────────────────────────────────────────────
    ("hospital name",       "Facility: University Medical Center",        "Medical Center"),
    ("clinic name",         "Referred to: University Health Clinic",      "University"),

    # ── Labels that precede PII (the label itself must survive) ──────────────
    ("label patient",       "Patient: Jonathan Harker",                   "Patient"),
    ("label ssn",           "SSN: 123-45-6789",                           "SSN"),
    ("label email",         "Email: jharker@example.com",                 "Email"),
    ("label dob",           "DOB: 03/15/1958",                            "DOB"),
    ("label mrn",           "MRN: 8834291",                               "MRN"),
    ("label phone",         "Phone: (806) 555-0192",                      "Phone"),
    ("label address",       "Address: 452 Elm Street, Lubbock TX 79401",  "Address"),
    ("label emerg contact", "Emergency Contact:Mina Murray",              "Emergency Contact"),

    # ── Non-DOB dates ────────────────────────────────────────────────────────
    ("procedure date lbl",  "Procedure Date: 03/15/2023",                 "Procedure Date"),
    ("visit date lbl",      "Visit Date: 04/10/2024",                     "Visit Date"),
    ("April",               "Follow up: April 2024",                     "April"),
    (   "month year",          "Birth •",                                   "Birth"),
    

    # ── Month words in normal prose (must NOT be redacted) ───────────────────
    ("may be",             "This may be a reaction to the medication",  "may be"),
    ("may include",        "Symptoms may include nausea and fatigue",   "may include"),
    ("june visit",         "Scheduled for June follow-up",              "June"),
    ("april date",         "Initial visit April 2019",                  "April 2019"),
    ("august note",        "Reviewed in August per protocol",           "August"),

    # ── Clinical values that must not be flagged ──────────────────────────────
    ("bp reading",          "Vital Signs: BP 120/80",                     "120/80"),
    ("o2 sat",              "SpO2: 98%",                                  "SpO2"),
    ("dosage mg",           "Dose: 500 mg",                               "500"),
    ("room number",         "Room: 214B",                                 "214B"),
    ("icd code",            "ICD-10: E11.9",                              "E11.9"),
    ("height weight",       "Ht: 5 ft 10 in  Wt: 185 lbs",              "185"),
    ("temp reading",        "Temp: 98.6 F",                               "98.6"),
    ("heart rate",          "HR: 72 bpm",                                 "72"),
    ("disease name",        "Diagnosis: Parkinson disease",          "Parkinson"),
    ("syndrome",            "Condition: Down syndrome",              "Down"),
    ("procedure name",      "Procedure: Whipple procedure",          "Whipple"),

    # ── Numbers that look like IDs ──────────────────────────────────────────
    ("glucose level",       "Glucose: 123 mg/dL",                    "123"),
    ("platelet count",      "Platelets: 250000",                     "250000"),

    # ── Dates that are NOT DOB ──────────────────────────────────────────────
    ("year only",           "History: 1998 surgery",                 "1998"),
    ("range",               "Hospitalized 2019-2020",               "2019-2020"),

    # ── Email-like but not email ────────────────────────────────────────────
    ("not email",           "Follow up at example.com site",         "example.com"),

    # ── Address-like but clinical ───────────────────────────────────────────
    ("unit reference",      "Patient in unit 5B",                    "5B"),
    
    # ── Names that are medications/brands ───────────────────────────────────
    ("insulin brand",       "Medication: Humalog insulin",           "Humalog"),
    ("device name",         "Device: Philips monitor",               "Philips"),

    # ── Month words edge cases ──────────────────────────────────────────────
    ("march verb",          "Patient will march forward with therapy","march"),
    ("may modal",           "Patient may return if symptoms worsen", "may"),
    ("brown color vs name", "Skin: brown discoloration","brown"),
    ("april false positive regression", "Follow up April 2024", "April"),

    # ── Abbreviations ───────────────────────────────────────────────────────
    ("dr abbreviation",     "Plan discussed with dr team",           "dr"),
    ("pt abbreviation",     "pt reports improvement",                "pt"),

    # ── Clinical shorthand ──────────────────────────────────────────────────
    ("bid",                 "Medication: take BID",                  "BID"),
    ("tid",                 "Medication: take TID",                  "TID"),
]


# ---------------------------------------------------------------------------
# Build one shared redacted PDF for all checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def redacted_text(pipeline, tmp_path_factory):
    extract_pii_terms_from_pdf, redact_pdf_names, DEFAULT_ENTITIES = pipeline

    tmp = tmp_path_factory.mktemp("redact_test")
    input_pdf = tmp / "input.pdf"
    output_pdf = tmp / "output.pdf"

    all_lines = list(dict.fromkeys(line for _, line, _ in SHOULD_REDACT + SHOULD_NOT_REDACT))
    _make_pdf(all_lines, input_pdf)

    terms = extract_pii_terms_from_pdf(input_pdf, entities=DEFAULT_ENTITIES, min_score=0.6)
    if terms:
        redact_pdf_names(input_pdf=input_pdf, output_pdf=output_pdf, names=terms)
    else:
        output_pdf.write_bytes(input_pdf.read_bytes())
    return _pdf_text(output_pdf)


# ---------------------------------------------------------------------------
# Parametrised tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,line,term", SHOULD_REDACT, ids=[x[0] for x in SHOULD_REDACT])
def test_pii_is_redacted(redacted_text, label, line, term):
    """PII terms must not appear verbatim in the redacted output."""
    assert term not in redacted_text, (
        f"[{label}] '{term}' should be redacted but still appears in output."
    )


@pytest.mark.parametrize("label,line,term", SHOULD_NOT_REDACT, ids=[x[0] for x in SHOULD_NOT_REDACT])
def test_non_pii_is_preserved(redacted_text, label, line, term):
    """Non-PII terms must survive redaction unchanged."""
    assert term in redacted_text, (
        f"[{label}] '{term}' should be preserved but was redacted."
    )


# ---------------------------------------------------------------------------
# Regex / structural checks
# ---------------------------------------------------------------------------


def test_ssn_pattern_gone(redacted_text):
    assert not re.search(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b", redacted_text), \
        "An SSN pattern is still present in output."


def test_email_pattern_gone(redacted_text):
    assert not re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", redacted_text), \
        "An email address is still present in output."


def test_credit_card_pattern_gone(redacted_text):
    assert not re.search(r"\b4\d{3}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b", redacted_text), \
        "A Visa card number is still present in output."


def test_known_patient_names_gone(redacted_text):
    for name in ("Jonathan Harker", "Emily Clarke", "Mina Murray",
                 "Arthur Holmwood", "Lucy Westenra", "Renfield Smith",
                 "Rachel Green", "Marcus Webb", "James Morrison"):
        assert name not in redacted_text, f"Patient name '{name}' still present in output."


def test_mrns_gone(redacted_text):
    assert "8834291" not in redacted_text
    assert "2291847" not in redacted_text


def test_dob_dates_gone(redacted_text):
    assert "03/15/1958" not in redacted_text
    assert "1975-08-22" not in redacted_text


def test_medications_all_present(redacted_text):
    for med in ("Tamsulosin", "Metformin", "Lisinopril", "Atorvastatin",
                "Albuterol", "Omeprazole", "Gabapentin", "Sertraline", "Furosemide"):
        assert med in redacted_text, f"Medication '{med}' was incorrectly redacted."


def test_section_headers_all_present(redacted_text):
    for h in ("Physical Exam", "Assessment", "Allergies", "Vital Signs",
              "Laboratory", "Discharge", "Social History", "Family History",
              "Radiology", "Diagnosis"):
        assert h in redacted_text, f"Section header '{h}' was incorrectly redacted."


def test_pii_labels_all_present(redacted_text):
    for lbl in ("Patient", "SSN", "Email", "DOB", "MRN",
                "Phone", "Address", "Emergency Contact"):
        assert lbl in redacted_text, f"PII label '{lbl}' was incorrectly redacted."


def test_clinical_values_preserved(redacted_text):
    for val in ("120/80", "SpO2", "98.6", "214B"):
        assert val in redacted_text, f"Clinical value '{val}' was incorrectly redacted."
def test_no_partial_name_leaks(redacted_text):
    """Ensure partial fragments of names don't leak (common bug)."""
    for frag in ("Jonathan", "Harker", "Emily", "Clarke"):
        assert frag not in redacted_text, f"Partial name fragment '{frag}' leaked."


def test_no_digit_sequences_like_ssn(redacted_text):
    """Catch near-SSN patterns."""
    assert not re.search(r"\b\d{3}[-\s]\d{2}[-\s]\d{3,4}\b", redacted_text)


def test_redaction_does_not_remove_entire_lines(redacted_text):
    """Ensure lines still exist after redaction (no over-deletion)."""
    lines = redacted_text.split("\n")
    assert any("Medications" in l for l in lines), "Content collapsed too aggressively."


def test_multiple_entities_same_line(redacted_text):
    """Ensure multiple PII on one line are all removed."""
    assert "John Doe" not in redacted_text
    assert "john@example.com" not in redacted_text
    assert "555-1234" not in redacted_text


def test_repeated_pii_all_removed(redacted_text):
    """Same PII appearing multiple times should be fully removed."""
    assert redacted_text.count("Jonathan Harker") == 0


# ---------------------------------------------------------------------------
# Entity-toggle tests — when an entity is disabled it must NOT be redacted
# ---------------------------------------------------------------------------

# (entity_to_disable, pdf_line, term_that_must_survive)
ENTITY_TOGGLE_CASES = [
    ("PERSON",               "Patient: Gordon Ramsay",                   "Gordon Ramsay"),
    ("EMAIL_ADDRESS",        "Email: toggle.test@example.com",           "toggle.test@example.com"),
    ("PHONE_NUMBER",         "Phone: (512) 555-9988",                    "(512) 555-9988"),
    ("US_SSN",               "SSN: 321-54-9876",                         "321-54-9876"),
    ("US_ADDRESS",           "Address: 77 Pine Street, Austin TX 78701", "77 Pine Street"),
    ("STRICT_DATE",          "DOB: 07/04/1980",                          "07/04/1980"),
    ("MEDICAL_RECORD_NUMBER","MRN: 1122334",                             "1122334"),
    ("INSURANCE",            "Medicare # 2EG5-TF6-NL83",                 "2EG5-TF6-NL83"),
]



def _run_pipeline_with_entities(tmp, entities, lines):
    from redact import extract_pii_terms_from_pdf, redact_pdf_names, DEFAULT_ENTITIES
    input_pdf = tmp / "input.pdf"
    output_pdf = tmp / "output.pdf"
    _make_pdf(lines, input_pdf)
    use_transformer = set(entities) == set(DEFAULT_ENTITIES)
    terms = extract_pii_terms_from_pdf(input_pdf, entities=entities, min_score=0.6, use_transformer=use_transformer)
    if terms:
        redact_pdf_names(input_pdf=input_pdf, output_pdf=output_pdf, names=terms)
    else:
        output_pdf.write_bytes(input_pdf.read_bytes())
    return _pdf_text(output_pdf)


@pytest.mark.parametrize("disabled_entity,line,term", ENTITY_TOGGLE_CASES,
                         ids=[x[0] for x in ENTITY_TOGGLE_CASES])
def test_disabled_entity_not_redacted(tmp_path_factory, disabled_entity, line, term):
    """When an entity type is disabled, its PII must survive in the output."""
    from redact import DEFAULT_ENTITIES
    entities = [e for e in DEFAULT_ENTITIES if e != disabled_entity]
    tmp = tmp_path_factory.mktemp(f"toggle_{disabled_entity.lower()}")
    text = _run_pipeline_with_entities(tmp, entities, [line])
    assert term in text, (
        f"[{disabled_entity} disabled] '{term}' should NOT be redacted but was removed."
    )

def test_large_document_performance():
    lines = ["Patient: John Doe"] * 10000
    import tempfile

    from redact import extract_pii_terms_from_pdf, redact_pdf_names, DEFAULT_ENTITIES

    tmp = Path(tempfile.mkdtemp(prefix="large_doc_"))
    input_pdf = tmp / "large_input.pdf"
    output_pdf = tmp / "large_output.pdf"

    _make_pdf(lines, input_pdf)
    terms = extract_pii_terms_from_pdf(input_pdf, entities=DEFAULT_ENTITIES, min_score=0.6)
    if terms:
        redact_pdf_names(input_pdf=input_pdf, output_pdf=output_pdf, names=terms)
    else:
        output_pdf.write_bytes(input_pdf.read_bytes())

    redacted_text = _pdf_text(output_pdf)
    assert "John Doe" not in redacted_text
    assert len(redacted_text) > 0
