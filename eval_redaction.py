"""
Quantitative evaluation of the redaction pipeline.

Computes Precision, Recall, F1, and Accuracy over the labeled test cases
defined in test_redaction.py.

Usage:
    python eval_redaction.py
    python eval_redaction.py --fail-below f1=0.90
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import fitz
from redact import extract_pii_terms_from_pdf, redact_pdf_names, DEFAULT_ENTITIES

# ---------------------------------------------------------------------------
# Test data (mirrors test_redaction.py — keep in sync)
# ---------------------------------------------------------------------------

SHOULD_REDACT = [
    ("patient name",              "Patient: Jonathan Harker",                          "Jonathan Harker"),
    ("patient name spaced",       "Patient Name: Emily Clarke",                        "Emily Clarke"),
    ("dr prefix",                 "Attending: Dr. Marcus Webb",                        "Marcus Webb"),
    ("name with middle initial",  "Patient: Sarah J. Thompson",                        "Sarah"),
    ("two-word name inline",      "Referred by James Morrison MD",                     "James Morrison"),
    ("name after colon no space", "Provider:Rachel Green",                             "Rachel Green"),
    ("continuously missed name",  "April Murphy",                                      "April Murphy"),
    ("april in table",            "Name Date Changes April Murphy Aug. 2019 Extensive","April Murphy"),
    ("parent/guardian",           "Parent/Guardian: Soo-Yeon Park",                    "Soo-Yeon Park"),
    ("white name vs color",       "Seen by Dr White",                                  "White"),
    ("stacked pii name",          "John Doe | 123-45-6789 | john@email.com",           "John Doe"),
    ("stacked pii ssn",           "John Doe | 123-45-6789 | john@email.com",           "123-45-6789"),
    ("stacked pii email",         "John Doe | 123-45-6789 | john@email.com",           "john@email.com"),
    ("emergency contact nospace", "Emergency Contact:Mina Murray",                     "Mina Murray"),
    ("emergency contact space",   "Emergency Contact: Arthur Holmwood",                "Arthur Holmwood"),
    ("next of kin",               "Next of Kin: Lucy Westenra",                        "Lucy Westenra"),
    ("guarantor",                 "Guarantor: Renfield Smith",                         "Renfield Smith"),
    ("ssn dashes",                "SSN: 123-45-6789",                                  "123-45-6789"),
    ("ssn spaces",                "Social Security: 234 56 7890",                      "234 56 7890"),
    ("phone parens",              "Phone: (806) 555-0192",                             "555-0192"),
    ("phone dots",                "Cell: 806.555.0193",                                "806.555.0193"),
    ("fax number",                "Fax: (806) 555-0195",                               "555-0195"),
    ("tel prefix",                "Tel: (773) 555-0344",                               "555-0344"),
    ("phone pipe header",         "550 Maternity Lane, Suite 300, Nashville, TN 37203 | (615) 555-0630", "555-0630"),
    ("telephone prefix",          "Telephone: (312) 555-9900",                         "555-9900"),
    ("mobile prefix",             "Mobile: 214-555-7722",                              "555-7722"),
    ("email basic",               "Email: jharker@example.com",                        "jharker@example.com"),
    ("email subdomain",           "Contact: patient.doe@mail.hospital.org",            "patient.doe@mail.hospital.org"),
    ("dob slash",                 "DOB: 03/15/1958",                                   "03/15/1958"),
    ("dob labeled written",       "Date of Birth: January 5, 1962",                    "January 5, 1962"),
    ("dob iso",                   "Birthdate: 1975-08-22",                             "1975-08-22"),
    ("dob table cell",            "DOB:\t07/04/1980",                                  "07/04/1980"),
    ("street address",            "Address: 452 Elm Street, Lubbock TX 79401",         "452 Elm Street"),
    ("avenue address",            "Home: 18 Oak Avenue, Dallas TX 75201",              "18 Oak Avenue"),
    ("apt address",               "Mailing: 301 Maple Dr Apt 4B, Austin TX",           "301 Maple Dr"),
    ("mrn labeled",               "MRN: 8834291",                                      "8834291"),
    ("mrn prefixed",              "Record: MRN-2291847",                               "2291847"),
    ("medicare id",               "Medicare # 1EG4-TE5-MK72",                          "1EG4-TE5-MK72"),
    ("insurance member id",       "Member ID: XYZ987654",                              "XYZ987654"),
    ("hyphenated name",           "Patient: Anne-Marie Johnson",                       "Anne-Marie Johnson"),
    ("apostrophe name",           "Patient: O'Connor Patrick",                         "O'Connor Patrick"),
    ("all caps name",             "PATIENT: MICHAEL SCOTT",                            "MICHAEL SCOTT"),
    ("lowercase name",            "patient: jim halpert",                              "jim halpert"),
    ("name with suffix",          "Patient: Robert Downey Jr.",                        "Robert Downey"),
    ("three part name",           "Patient: Mary Kate Olsen",                          "Mary Kate Olsen"),
    ("name with title inline",    "Seen by Dr John Watson today",                      "John Watson"),
    ("phone no separators",       "Phone: 8065551234",                                 "8065551234"),
    ("phone with country",        "Phone: +1 (806) 555-7777",                          "555-7777"),
    ("phone ext",                 "Phone: 806-555-8888 ext 22",                        "806-555-8888"),
    ("email plus",                "Email: test+alias@gmail.com",                       "test+alias@gmail.com"),
    ("email caps",                "Email: JOHN.DOE@EXAMPLE.COM",                       "JOHN.DOE@EXAMPLE.COM"),
    ("dob dots",                  "DOB: 03.15.1958",                                   "03.15.1958"),
    ("dob short year",            "DOB: 03/15/58",                                     "03/15/58"),
    ("address lowercase",         "Address: 88 pine street houston tx",                "88 pine street"),
    ("address no comma",          "Address: 12 Sunset Blvd Los Angeles CA",            "12 Sunset Blvd"),
    ("address unit",              "Address: 400 Main St Unit 12",                      "400 Main St"),
    ("mrn spaced",                "MRN : 4455667",                                     "4455667"),
    ("mrn text inline",           "Patient MRN 9988776 recorded",                      "9988776"),
    ("insurance dashed",          "Policy: ABC-123-XYZ",                               "ABC-123-XYZ"),
    ("insurance numeric",         "Insurance ID: 999888777",                           "999888777"),
]

SHOULD_NOT_REDACT = [
    ("tamsulosin",        "Medications: Tamsulosin 0.4 mg daily",       "Tamsulosin"),
    ("metformin",         "Medications: Metformin 500 mg BID",          "Metformin"),
    ("lisinopril",        "Medications: Lisinopril 10 mg daily",        "Lisinopril"),
    ("atorvastatin",      "Medications: Atorvastatin 40 mg nightly",    "Atorvastatin"),
    ("albuterol",         "Medications: Albuterol inhaler PRN",         "Albuterol"),
    ("omeprazole",        "Medications: Omeprazole 20 mg daily",        "Omeprazole"),
    ("gabapentin",        "Medications: Gabapentin 300 mg TID",         "Gabapentin"),
    ("sertraline",        "Medications: Sertraline 50 mg daily",        "Sertraline"),
    ("furosemide",        "Medications: Furosemide 40 mg daily",        "Furosemide"),
    ("kcl",               "Medications: KCl 20 mEq daily",              "KCl"),
    ("pneumovax",         "Vaccines: Pneumovax 23 administered",        "Pneumovax"),
    ("aspirin",           "Medications: Aspirin 81 mg daily",           "Aspirin"),
    ("warfarin",          "Medications: Warfarin 5 mg daily",           "Warfarin"),
    ("prednisone",        "Medications: Prednisone 10 mg daily",        "Prednisone"),
    ("insulin",           "Medications: Insulin glargine 20 units QHS", "Insulin"),
    ("amoxicillin",       "Medications: Amoxicillin 500 mg TID",        "Amoxicillin"),
    ("azithromycin",      "Medications: Azithromycin 250 mg daily",     "Azithromycin"),
    ("levothyroxine",     "Medications: Levothyroxine 50 mcg daily",    "Levothyroxine"),
    ("clopidogrel",       "Medications: Clopidogrel 75 mg daily",       "Clopidogrel"),
    ("amlodipine",        "Medications: Amlodipine 5 mg daily",         "Amlodipine"),
    ("metoprolol",        "Medications: Metoprolol 25 mg BID",          "Metoprolol"),
    ("glipizide",         "Medications: Glipizide 5 mg daily",          "Glipizide"),
    ("morphine",          "Medications: Morphine 2 mg IV PRN",          "Morphine"),
    ("oxycodone",         "Medications: Oxycodone 5 mg q4h PRN",        "Oxycodone"),
    ("ibuprofen",         "Medications: Ibuprofen 400 mg TID",          "Ibuprofen"),
    ("mmse",              "Cognitive: MMSE score 28/30",                "MMSE"),
    ("bmi",               "BMI: 24.5 kg/m2",                            "BMI"),
    ("ekg",               "EKG: normal sinus rhythm",                   "EKG"),
    ("orthopedist role",  "Referred to Orthopedist for follow-up",      "Orthopedist"),
    ("cardiologist role", "Seen by Cardiologist on 11/03/2023",         "Cardiologist"),
    ("time am",           "Admitted at 8:30 am per nursing notes",      "30 am"),
    ("time pm",           "Discharge time: 2:00 pm",                    "00 pm"),
    ("flu shot",          "Vaccines: Influenza vaccine given",          "Influenza"),
    ("tdap",              "Vaccines: Tdap booster administered",        "Tdap"),
    ("united states",     "Born in the United States",                  "United States"),
    ("new york",          "Relocated from New York to Texas",           "New York"),
    ("dyspnea",           "Chief Complaint: dyspnea on exertion",       "dyspnea"),
    ("epigastric",        "Pain: epigastric tenderness noted",          "epigastric"),
    ("musculoskeletal",   "MSK: musculoskeletal exam normal",           "musculoskeletal"),
    ("physical exam",     "Physical Exam:",                             "Physical Exam"),
    ("history header",    "History of Present Illness:",                "History"),
    ("assessment header", "Assessment and Plan:",                       "Assessment"),
    ("allergies header",  "Allergies: Penicillin",                      "Allergies"),
    ("medications hdr",   "Current Medications:",                       "Medications"),
    ("vital signs",       "Vital Signs: BP 120/80",                     "Vital Signs"),
    ("review systems",    "Review of Systems:",                         "Review"),
    ("social history",    "Social History: non-smoker",                 "Social History"),
    ("family history",    "Family History: no known cardiac disease",   "Family History"),
    ("discharge summary", "Discharge Summary",                          "Discharge"),
    ("laboratory",        "Laboratory Results:",                        "Laboratory"),
    ("radiology",         "Radiology: chest X-ray normal",              "Radiology"),
    ("diagnosis header",  "Diagnosis: Type 2 Diabetes Mellitus",        "Diagnosis"),
    ("hospital name",     "Facility: University Medical Center",        "Medical Center"),
    ("clinic name",       "Referred to: University Health Clinic",      "University"),
    ("label patient",     "Patient: Jonathan Harker",                   "Patient"),
    ("label ssn",         "SSN: 123-45-6789",                           "SSN"),
    ("label email",       "Email: jharker@example.com",                 "Email"),
    ("label dob",         "DOB: 03/15/1958",                            "DOB"),
    ("label mrn",         "MRN: 8834291",                               "MRN"),
    ("label phone",       "Phone: (806) 555-0192",                      "Phone"),
    ("label address",     "Address: 452 Elm Street, Lubbock TX 79401",  "Address"),
    ("label emerg contact","Emergency Contact:Mina Murray",             "Emergency Contact"),
    ("procedure date lbl","Procedure Date: 03/15/2023",                 "Procedure Date"),
    ("visit date lbl",    "Visit Date: 04/10/2024",                     "Visit Date"),
    ("April",             "Follow up: April 2024",                      "April"),
    ("month year",        "Birth •",                                     "Birth"),
    ("may be",            "This may be a reaction to the medication",   "may be"),
    ("may include",       "Symptoms may include nausea and fatigue",    "may include"),
    ("june visit",        "Scheduled for June follow-up",               "June"),
    ("april date",        "Initial visit April 2019",                   "April 2019"),
    ("august note",       "Reviewed in August per protocol",            "August"),
    ("bp reading",        "Vital Signs: BP 120/80",                     "120/80"),
    ("o2 sat",            "SpO2: 98%",                                  "SpO2"),
    ("dosage mg",         "Dose: 500 mg",                               "500"),
    ("room number",       "Room: 214B",                                 "214B"),
    ("icd code",          "ICD-10: E11.9",                              "E11.9"),
    ("height weight",     "Ht: 5 ft 10 in  Wt: 185 lbs",               "185"),
    ("temp reading",      "Temp: 98.6 F",                               "98.6"),
    ("heart rate",        "HR: 72 bpm",                                 "72"),
    ("disease name",      "Diagnosis: Parkinson disease",               "Parkinson"),
    ("syndrome",          "Condition: Down syndrome",                   "Down"),
    ("procedure name",    "Procedure: Whipple procedure",               "Whipple"),
    ("glucose level",     "Glucose: 123 mg/dL",                        "123"),
    ("platelet count",    "Platelets: 250000",                          "250000"),
    ("year only",         "History: 1998 surgery",                      "1998"),
    ("range",             "Hospitalized 2019-2020",                     "2019-2020"),
    ("not email",         "Follow up at example.com site",              "example.com"),
    ("unit reference",    "Patient in unit 5B",                         "5B"),
    ("insulin brand",     "Medication: Humalog insulin",                "Humalog"),
    ("device name",       "Device: Philips monitor",                    "Philips"),
    ("march verb",        "Patient will march forward with therapy",    "march"),
    ("may modal",         "Patient may return if symptoms worsen",      "may"),
    ("brown color vs name","Skin: brown discoloration",                 "brown"),
    ("april false positive regression","Follow up April 2024",          "April"),
    ("dr abbreviation",   "Plan discussed with dr team",                "dr"),
    ("pt abbreviation",   "pt reports improvement",                     "pt"),
    ("bid",               "Medication: take BID",                       "BID"),
    ("tid",               "Medication: take TID",                       "TID"),
]


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


def run_pipeline(lines: list[str], tmp: Path) -> str:
    input_pdf = tmp / "input.pdf"
    output_pdf = tmp / "output.pdf"
    _make_pdf(lines, input_pdf)
    terms = extract_pii_terms_from_pdf(input_pdf, entities=DEFAULT_ENTITIES, min_score=0.6)
    if terms:
        redact_pdf_names(input_pdf=input_pdf, output_pdf=output_pdf, names=terms)
    else:
        output_pdf.write_bytes(input_pdf.read_bytes())
    return _pdf_text(output_pdf)


def evaluate() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="eval_redact_"))
    all_lines = list(dict.fromkeys(line for _, line, _ in SHOULD_REDACT + SHOULD_NOT_REDACT))

    print("Running redaction pipeline...", flush=True)
    text = run_pipeline(all_lines, tmp)

    tp_cases, fn_cases, fp_cases, tn_cases = [], [], [], []

    for label, _line, term in SHOULD_REDACT:
        if term not in text:
            tp_cases.append((label, term))
        else:
            fn_cases.append((label, term))

    for label, _line, term in SHOULD_NOT_REDACT:
        if term not in text:
            fp_cases.append((label, term))
        else:
            tn_cases.append((label, term))

    tp, fn, fp, tn = len(tp_cases), len(fn_cases), len(fp_cases), len(tn_cases)
    total = tp + fn + fp + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    return dict(
        tp=tp, fn=fn, fp=fp, tn=tn,
        n_pii=len(SHOULD_REDACT), n_non_pii=len(SHOULD_NOT_REDACT),
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
        fn_cases=fn_cases, fp_cases=fp_cases,
    )


def print_report(r: dict) -> None:
    print()
    print("=" * 45)
    print("  Redaction Evaluation Report")
    print("=" * 45)
    print(f"  TP  correctly redacted PII:   {r['tp']:3d} / {r['n_pii']}")
    print(f"  FN  missed PII:               {r['fn']:3d} / {r['n_pii']}")
    print(f"  FP  over-redacted non-PII:    {r['fp']:3d} / {r['n_non_pii']}")
    print(f"  TN  correctly preserved:      {r['tn']:3d} / {r['n_non_pii']}")
    print()
    print(f"  Precision : {r['precision']:.3f}")
    print(f"  Recall    : {r['recall']:.3f}")
    print(f"  F1 Score  : {r['f1']:.3f}")
    print(f"  Accuracy  : {r['accuracy']:.3f}")

    if r['fn_cases']:
        print()
        print(f"--- Missed PII (False Negatives: {len(r['fn_cases'])}) ---")
        for label, term in r['fn_cases']:
            print(f"  [{label}]  \"{term}\"")

    if r['fp_cases']:
        print()
        print(f"--- Over-redacted non-PII (False Positives: {len(r['fp_cases'])}) ---")
        for label, term in r['fp_cases']:
            print(f"  [{label}]  \"{term}\"")

    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fail-below",
        metavar="METRIC=THRESHOLD",
        help="Exit with code 1 if a metric is below threshold. E.g. f1=0.90",
    )
    args = parser.parse_args()

    results = evaluate()
    print_report(results)

    if args.fail_below:
        metric, threshold = args.fail_below.split("=")
        value = results[metric]
        if value < float(threshold):
            print(f"FAIL: {metric}={value:.3f} is below threshold {threshold}")
            sys.exit(1)
        else:
            print(f"PASS: {metric}={value:.3f} >= {threshold}")


if __name__ == "__main__":
    main()
