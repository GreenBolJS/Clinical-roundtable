from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.vector.chroma import add_documents, get_collection

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MAX_PER_TOPIC = 15
API_DELAY = 0.5

TOPICS_BY_CONDITION = {
    "Cardiovascular": [
        "acute coronary syndrome",
        "hypertensive emergency",
        "hypertensive encephalopathy",
        "ischemic stroke",
        "intracerebral hemorrhage",
        "cerebral venous sinus thrombosis",
        "heart failure management",
        "atrial fibrillation",
        "aortic dissection",
        "pulmonary embolism",
        "deep vein thrombosis",
        "cardiac arrhythmia",
        "pericarditis",
        "myocarditis",
        "endocarditis",
    ],
    "Respiratory": [
        "COPD exacerbation",
        "pneumonia treatment",
        "asthma exacerbation",
        "pulmonary fibrosis",
        "pneumothorax",
        "pleural effusion",
        "respiratory failure",
        "tuberculosis diagnosis",
        "lung cancer screening",
        "sleep apnea",
    ],
    "Neurological": [
        "transient ischemic attack",
        "meningitis diagnosis",
        "encephalitis treatment",
        "epilepsy management",
        "migraine treatment",
        "Parkinson disease",
        "multiple sclerosis",
        "Guillain-Barre syndrome",
        "myasthenia gravis",
        "brain tumor diagnosis",
    ],
    "Endocrine/Metabolic": [
        "diabetic ketoacidosis",
        "hyperosmolar hyperglycemic state",
        "hypoglycemia management",
        "thyroid storm",
        "hypothyroidism treatment",
        "adrenal crisis",
        "Cushing syndrome",
        "hyperkalemia management",
        "hyponatremia treatment",
        "metabolic acidosis",
    ],
    "Gastrointestinal": [
        "acute appendicitis",
        "gastrointestinal bleeding",
        "peptic ulcer disease",
        "pancreatitis management",
        "cholecystitis treatment",
        "inflammatory bowel disease",
        "liver cirrhosis",
        "hepatitis diagnosis",
        "bowel obstruction",
        "peritonitis",
    ],
    "Renal": [
        "acute kidney injury",
        "chronic kidney disease",
        "nephrotic syndrome",
        "urinary tract infection",
        "kidney stones",
        "renal failure dialysis",
        "glomerulonephritis",
        "polycystic kidney disease",
        "renal artery stenosis",
        "electrolyte imbalance",
    ],
    "Infectious Disease": [
        "sepsis diagnosis",
        "septic shock management",
        "COVID-19 treatment",
        "influenza management",
        "malaria diagnosis",
        "dengue fever",
        "HIV management",
        "bacterial meningitis",
        "cellulitis treatment",
        "osteomyelitis",
    ],
    "Hematology/Oncology": [
        "anemia diagnosis",
        "sickle cell crisis",
        "leukemia treatment",
        "lymphoma diagnosis",
        "coagulation disorders",
        "thrombocytopenia",
        "multiple myeloma",
        "bone marrow failure",
        "tumor lysis syndrome",
        "chemotherapy complications",
    ],
    "Musculoskeletal": [
        "rheumatoid arthritis",
        "systemic lupus erythematosus",
        "gout management",
        "osteoporosis treatment",
        "septic arthritis",
        "rhabdomyolysis",
        "compartment syndrome",
        "osteoarthritis management",
        "ankylosing spondylitis",
        "polymyalgia rheumatica",
    ],
    "Pharmacology/Drug Interactions": [
        "CYP3A4 drug interactions",
        "anticoagulant drug interactions",
        "statin myopathy",
        "ACE inhibitor side effects",
        "NSAID renal toxicity",
        "beta blocker overdose",
        "serotonin syndrome",
        "neuroleptic malignant syndrome",
        "drug induced liver injury",
        "antibiotic resistance",
    ],
    "Psychiatry": [
        "major depressive disorder",
        "bipolar disorder management",
        "schizophrenia treatment",
        "anxiety disorder",
        "panic attack diagnosis",
        "delirium management",
        "dementia diagnosis",
        "substance withdrawal",
        "suicidal ideation assessment",
        "PTSD treatment",
    ],
    "Emergency/Critical Care": [
        "anaphylaxis management",
        "trauma resuscitation",
        "burns management",
        "drowning resuscitation",
        "hypothermia treatment",
        "heat stroke management",
        "poisoning management",
        "shock diagnosis",
        "mechanical ventilation",
        "CPR guidelines",
    ],
}

TOPICS = [
    {"condition": condition, "topic": topic}
    for condition, topics in TOPICS_BY_CONDITION.items()
    for topic in topics
]
TOTAL_TOPICS = len(TOPICS)


def search_pubmed(term: str, retmax: int = MAX_PER_TOPIC) -> list[str]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "relevance",
    }
    response = requests.get(ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_pubmed_abstracts(pmids: list[str]) -> list[dict[str, Any]]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    response = requests.get(EFETCH_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    records: list[dict[str, Any]] = []
    for article in root.findall("PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

        article_info = medline.find("Article")
        if article_info is None:
            continue

        title_el = article_info.find("ArticleTitle")
        title = title_el.text.strip() if title_el is not None and title_el.text else "Untitled"

        abstract_texts: list[str] = []
        abstract = article_info.find("Abstract")
        if abstract is not None:
            for section in abstract.findall("AbstractText"):
                text = section.text or ""
                label = section.attrib.get("Label")
                if label:
                    abstract_texts.append(f"{label}: {text.strip()}")
                else:
                    abstract_texts.append(text.strip())
        abstract_text = "\n".join(t for t in abstract_texts if t)

        journal = "unknown"
        journal_el = article_info.find("Journal/Title")
        if journal_el is not None and journal_el.text:
            journal = journal_el.text.strip()
        elif article_info.find("Journal/ISOAbbreviation") is not None:
            journal = article_info.find("Journal/ISOAbbreviation").text or journal

        year = "unknown"
        journal_issue = article_info.find("Journal/JournalIssue")
        if journal_issue is not None:
            pub_date_el = journal_issue.find("PubDate")
            if pub_date_el is not None:
                year_el = pub_date_el.find("Year")
                medline_date_el = pub_date_el.find("MedlineDate")
                if year_el is not None and year_el.text:
                    year = year_el.text.strip()
                elif medline_date_el is not None and medline_date_el.text:
                    year = medline_date_el.text.strip()

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract_text,
            "year": year,
            "journal": journal,
        })

    return records


def seed_chroma() -> None:
    collection = get_collection()
    try:
        collection.delete(where={"topic": {"$ne": ""}})
        print("Cleared existing ChromaDB collection.")
    except Exception as exc:
        print(f"Warning: failed to clear collection with topic filter: {exc}")

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    seed_log: list[dict[str, Any]] = []

    for index, entry in enumerate(TOPICS, start=1):
        condition = entry["condition"]
        topic = entry["topic"]
        print(f"Seeding topic {index}/{TOTAL_TOPICS}: {topic} — fetching PMIDs")

        try:
            pmids = search_pubmed(topic)
        except Exception as exc:
            print(f"  Error searching PubMed for '{topic}': {exc}")
            seed_log.append({"condition": condition, "topic": topic, "fetched": 0, "notes": "search_error"})
            time.sleep(API_DELAY)
            continue

        time.sleep(API_DELAY)

        if not pmids:
            print(f"  Skipping topic {index}/{TOTAL_TOPICS}: '{topic}' returned no PubMed results")
            seed_log.append({"condition": condition, "topic": topic, "fetched": 0})
            continue

        pmids = pmids[:MAX_PER_TOPIC]
        try:
            records = fetch_pubmed_abstracts(pmids)
        except Exception as exc:
            print(f"  Error fetching abstracts for '{topic}': {exc}")
            seed_log.append({"condition": condition, "topic": topic, "fetched": 0, "notes": "fetch_error"})
            time.sleep(API_DELAY)
            continue

        time.sleep(API_DELAY)

        topic_count = 0
        for record in records:
            if not record["abstract"]:
                continue

            documents.append(f"{record['title']}\n\n{record['abstract']}")
            metadatas.append(
                {
                    "condition": condition,
                    "topic": topic,
                    "pmid": record["pmid"],
                    "year": record["year"],
                    "journal": record["journal"],
                }
            )
            topic_count += 1

        print(f"Seeding topic {index}/{TOTAL_TOPICS}: {topic} — fetched {topic_count} abstracts")
        seed_log.append({"condition": condition, "topic": topic, "fetched": topic_count})

    if documents:
        print(f"Adding {len(documents)} documents to ChromaDB...")
        add_documents(documents=documents, metadatas=metadatas)
        print("ChromaDB seeding complete.")
    else:
        print("No documents were loaded into ChromaDB.")

    log_path = Path(__file__).resolve().parent / "seed_log.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(seed_log, handle, indent=2)

    total_seeded = sum(entry.get("fetched", 0) for entry in seed_log)
    print(f"Total documents seeded: {total_seeded}")
    print(f"Seed log written to: {log_path}")


if __name__ == "__main__":
    seed_chroma()
