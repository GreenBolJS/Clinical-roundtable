# The Clinical Roundtable

A **Recursive Multi-Agent Clinical Intelligence System** built with FastAPI, LangGraph, Groq, ChromaDB, and PostgreSQL.

Specialist agents — Diagnostic Pathologist, PharmacogenomicsSpecialist, Chief Medical Officer, Adversarial Auditor, Literature Critic, Bio-Sensor Analyst, and Citation Agent — collaborate in parallel to produce comprehensive clinical reports with confidence scoring and human-in-the-loop escalation.

---

## Architecture Overview

```
POST /api/v1/consult
        │
        ▼
  PII Redaction (spaCy/scispaCy)
        │
        ▼
  [LangGraph Pipeline]
  triage_node
        │ (Send API fan-out)
        ├──→ pathologist_node (Groq 70b)
        ├──→ pharmaco_node    (Groq 70b)
        ├──→ literature_node  (Gemini Flash + ChromaDB)
        └──→ biosensor_node   (Gemini Flash)
                │
                ▼
          citation_node (Gemini Flash)
                │
                ▼
           cmo_node (Groq 70b)
                │
                ▼
         auditor_node (Groq 70b)
                │
                ▼
       confidence_node (Pure Python)
                │
          ┌─────┴───────────────────┐
     unanswered                  continue
     challenges                     │
     + loop < 1                     ▼
          │                    hitl_node
          └──→ loop_increment       │
               → cmo_node           ▼
                              output_node
                                    │
                                    ▼
                             ClinicalReport
```

---

## Prerequisites

- Python 3.11+
- Redis (for Celery)
- A PostgreSQL database
- Groq API key (free tier at console.groq.com)
- Google Gemini API key (free tier at aistudio.google.com)

---

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo>
cd clinical_roundtable
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install scispaCy model

The PII redactor uses `en_core_sci_md` from scispaCy for named entity recognition:

```bash
# Install the model wheel directly from GitHub releases
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

> If that URL is outdated, check https://github.com/allenai/scispacy for the latest release.

Verify installation:
```bash
python -c "import spacy; nlp = spacy.load('en_core_sci_md'); print('scispaCy loaded OK')"
```

### 5. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your actual keys
```

Required variables:
- `GROQ_API_KEY` — from console.groq.com
- `GEMINI_API_KEY` — from aistudio.google.com
- `DATABASE_URL` — your PostgreSQL connection string
- `REDIS_URL` — Redis connection string (default: `redis://localhost:6379/0`)
- `HITL_WEBHOOK_URL` — URL that receives HITL notifications
- `CHROMA_PERSIST_PATH` — local directory for ChromaDB persistence

### 6. Start Redis (if not already running)

```bash
# Docker
docker run -d -p 6379:6379 redis:alpine

# Or on Ubuntu/Debian
sudo service redis-server start
```

---

## Running the Application

### Development server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Start Celery worker (for background tasks)

```bash
celery -A app.tasks worker --loglevel=info --concurrency=4
```

---

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Sample API Calls

### Health Check

```bash
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
```

### POST /api/v1/consult — Full Clinical Consultation

```bash
curl -s -X POST http://localhost:8000/api/v1/consult \
  -H "Content-Type: application/json" \
  -d '{
    "patient_token": "PATIENT_001",
    "symptoms": [
      "severe chest pain radiating to left arm",
      "diaphoresis",
      "shortness of breath",
      "nausea"
    ],
    "history": "67-year-old male with hypertension, type 2 diabetes, and hyperlipidemia. Previous cardiac catheterization 5 years ago showing 40% LAD stenosis. No prior MI. Non-smoker.",
    "current_medications": [
      "metformin 1000mg twice daily",
      "lisinopril 10mg daily",
      "atorvastatin 40mg daily",
      "aspirin 81mg daily"
    ],
    "biosensor_data": {
      "heart_rate": 112,
      "blood_pressure_systolic": 158,
      "blood_pressure_diastolic": 94,
      "spo2": 96,
      "ecg_rhythm": "sinus tachycardia with ST elevation in leads II, III, aVF",
      "temperature": 37.1,
      "respiratory_rate": 22
    }
  }' | python3 -m json.tool
```

### GET /api/v1/session/{session_id} — Retrieve Past Report

```bash
curl -s http://localhost:8000/api/v1/session/<SESSION_ID> | python3 -m json.tool
```

### POST /api/v1/patient — Create Patient Record

```bash
curl -s -X POST http://localhost:8000/api/v1/patient \
  -H "Content-Type: application/json" \
  -d '{
    "patient_token": "PATIENT_001",
    "age_range": "65-70",
    "notes": "Cardiology patient, annual review"
  }' | python3 -m json.tool
```

### POST /api/v1/webhook/hitl-response — Doctor Override

```bash
curl -s -X POST http://localhost:8000/api/v1/webhook/hitl-response \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<SESSION_ID>",
    "doctor_id": "DR_SMITH_001",
    "override_decision": "Confirmed STEMI. Initiating cath lab activation. AI differential is correct.",
    "revised_diagnosis": "ST-Elevation Myocardial Infarction (STEMI) - Inferior",
    "approved": true
  }' | python3 -m json.tool
```

---

## Seeding ChromaDB with Medical Literature

To populate the vector store with medical journal embeddings:

```python
from app.vector.chroma import add_documents

# Example: add PubMed abstracts
documents = [
    "Aspirin inhibits platelet aggregation by irreversibly blocking COX-1...",
    "In patients with STEMI, primary PCI within 90 minutes reduces mortality...",
]
metadatas = [
    {"title": "Antiplatelet Therapy Review", "evidence_level": "meta-analysis", "year": 2023},
    {"title": "STEMI Management Guidelines", "evidence_level": "RCT", "year": 2022},
]
add_documents(documents=documents, metadatas=metadatas)
```

---

## Confidence Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| ≥ 85  | High confidence | Automatic report delivery |
| 70–84 | Moderate confidence | HITL notification sent |
| < 70  | Low confidence | HITL notification + `escalate_immediately: true` |

---

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq Cloud API key | ✅ |
| `GEMINI_API_KEY` | Google Gemini API key | ✅ |
| `DATABASE_URL` | PostgreSQL connection URL | ✅ |
| `REDIS_URL` | Redis connection URL | ✅ |
| `HITL_WEBHOOK_URL` | Webhook URL for HITL alerts | ✅ |
| `CHROMA_PERSIST_PATH` | ChromaDB persistence directory | Optional (default: `./chroma_db`) |

---

## Project Structure

```
clinical_roundtable/
├── app/
│   ├── main.py                    # FastAPI app, lifespan, CORS
│   ├── config.py                  # pydantic-settings configuration
│   ├── tasks.py                   # Celery task queue
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── clinical.py            # All Pydantic v2 models
│   ├── providers/
│   │   ├── router.py              # ProviderRouter with semaphores
│   │   ├── groq_client.py         # Groq async client
│   │   └── gemini_client.py       # Gemini async wrapper
│   ├── agents/
│   │   ├── triage.py              # Groq 8b — routing
│   │   ├── pathologist.py         # Groq 70b — differential diagnosis
│   │   ├── pharmaco.py            # Groq 70b — drug interactions
│   │   ├── literature.py          # Gemini Flash — evidence review
│   │   ├── biosensor.py           # Gemini Flash — sensor data
│   │   ├── citation.py            # Gemini Flash — citation enrichment
│   │   ├── cmo.py                 # Groq 70b — synthesis
│   │   └── auditor.py             # Groq 70b — adversarial audit
│   ├── graph/
│   │   ├── state.py               # GraphState TypedDict
│   │   └── graph.py               # LangGraph builder
│   ├── confidence/
│   │   └── engine.py              # ConfidenceEngine
│   ├── pii/
│   │   └── redactor.py            # PIIRedactor (spaCy + regex)
│   ├── db/
│   │   └── supabase.py            # asyncpg patient ledger
│   ├── vector/
│   │   └── chroma.py              # ChromaDB client
│   └── api/
│       └── routes.py              # All FastAPI routes
├── .env.example
├── requirements.txt
└── README.md
```

---

