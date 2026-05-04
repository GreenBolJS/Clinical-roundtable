# Clinical Roundtable
### Multi-Agent Clinical Intelligence System

---

## What It Is

Clinical Roundtable is a FastAPI backend that runs multiple AI agents in parallel to analyze patient data and produce a structured clinical report. Each agent has a specialized role. Their outputs are synthesized, challenged by an adversarial auditor, and scored for confidence before a final report is returned.

All inference runs on free-tier Groq (llama-3.3-70b-versatile and llama-3.1-8b-instant). No paid APIs required.

---

## Architecture

The system uses LangGraph to orchestrate a cyclic graph of agents. A triage agent decides which specialists to activate, they run in parallel, and their outputs flow into synthesis, auditing, and confidence scoring before the final report is assembled.

```
User Query
    ↓
PII Redactor (local, spaCy)
    ↓
Triage Agent (Groq 8b)
    ↓ (parallel)
Pathologist | Literature Critic | Bio-Sensor Streamer | PharmacoGenomics
    ↓
Citation Agent
    ↓
CMO Agent (synthesis)
    ↓
Adversarial Auditor
    ↓
Confidence Engine
    ↓
[HITL gate if score < 85]
    ↓
Final Clinical Report
```

---

## Agents

**Triage Agent** — Groq llama-3.1-8b-instant
Receives the patient query and decides which specialist agents need to be activated. Always activates pharmaco if medications are present.

**Diagnostic Pathologist** — Groq llama-3.3-70b-versatile
Analyzes symptoms, history, and patient context to produce a ranked differential diagnosis list with ICD-10 codes and probability scores.

**PharmacoGenomics Agent** — Groq llama-3.3-70b-versatile
Checks every possible pair of medications including OTC drugs, vitamins, and supplements for drug-drug interactions. Returns severity (none/mild/moderate/severe) and mechanism for each flagged pair.

**Literature Critic** — Groq llama-3.3-70b-versatile
Queries ChromaDB (seeded with ~2900 PubMed abstracts across 195 medical topics) and returns the best-evidenced diagnosis from the differential, corroborating or challenging the pathologist.

**Bio-Sensor Streamer** — Groq llama-3.1-8b-instant
Parses real-time wearable/biosensor data (heart rate, blood pressure, O2 saturation, temperature, glucose, respiratory rate) and identifies which differential diagnoses the vitals support or contradict.

**Citation Agent** — Groq llama-3.1-8b-instant
Links every clinical finding to a source with title, PubMed URL, paragraph excerpt, and evidence level (RCT / meta-analysis / cohort / case-report / expert-opinion).

**CMO Agent (Chief Medical Officer)** — Groq llama-3.3-70b-versatile
Synthesizes all specialist outputs into a unified clinical report with a narrative summary and structured confidence sub-scores.

**Adversarial Auditor** — Groq llama-3.3-70b-versatile
Acts as devil's advocate. Reviews the CMO report, raises specific clinical challenges, and scores how many the CMO actually answered. Returns a pass/fail verdict.

---

## Confidence Scoring

The system computes a weighted confidence score from four signals:

Signal 1 — CMO self-report (30%)
The CMO outputs structured sub-scores: diagnosis_confidence, evidence_strength, contraindication_risk (inverted), differential_completeness.

Signal 2 — Inter-agent agreement (25%)
Compares the top differentials from the Pathologist, Literature Critic, and Bio-Sensor agent using fuzzy substring matching. Higher agreement = higher confidence.

Signal 3 — Citation density (20%)
Scored from citation count, recency of sources, and ratio of RCT/meta-analysis citations vs case reports and expert opinions.

Signal 4 — Adversarial audit (25%)
Penalizes unanswered challenges. Each unanswered challenge reduces this signal by 8 points.

Final score = weighted sum × 100.

If score < 85: HITL required (human doctor sign-off flagged).
If score < 70 AND unanswered challenges > 3: escalate immediately.
Critical vitals (BP < 90 systolic, O2 < 92%) trigger immediate escalation regardless of score.

---

## PII Redaction

All patient data passes through a local spaCy redactor (en_core_sci_md from scispaCy) before any cloud API call. Names, MRNs, and dates of birth are replaced with tokens like [PATIENT_001]. No real patient identifiers ever reach Groq's servers.

---

## Medical Knowledge Base

ChromaDB is seeded with approximately 2,900 PubMed abstracts across 195 medical topics covering cardiovascular, neurological, respiratory, renal, infectious, psychiatric, endocrine, gastrointestinal, oncology, obstetric, emergency, and pharmacology domains. This gives the Literature Critic and Citation Agent real retrieved content instead of relying on hallucinated references.

Run once before starting the server:
    python scripts/seed_chroma.py

---

## Tech Stack

- FastAPI — API framework and route management
- LangGraph — cyclic graph orchestration for agent flow
- Groq (llama-3.3-70b-versatile, llama-3.1-8b-instant) — all LLM inference
- ChromaDB — local vector store for medical literature
- spaCy + scispaCy (en_core_sci_md) — local PII redaction
- Pydantic v2 — strict schema validation for all agent outputs
- asyncio Semaphore — rate limiting (max 2 concurrent Groq calls)
- structlog — structured JSON logging for every agent call

---

## API Endpoints

POST /api/v1/consult
Accepts a ClinicalQuery and runs the full multi-agent pipeline. Returns a ClinicalReport.

GET /api/v1/session/{session_id}
Retrieves a past consultation by session ID.

POST /api/v1/patient
Creates a new tokenized patient record.

GET /api/v1/health
Returns connectivity status for Groq providers.

POST /api/v1/webhook/hitl-response
Endpoint for a human doctor to submit an override decision.

---

## Input Schema

```json
{
  "patient_token": "PATIENT_001",
  "symptoms": ["chest pain", "shortness of breath"],
  "history": "45 year old male, smoker for 20 years",
  "current_medications": ["aspirin", "metformin"],
  "biosensor_data": {
    "heart_rate": 98,
    "blood_pressure": "145/92",
    "oxygen_saturation": 96,
    "temperature": 37.1,
    "respiratory_rate": 18,
    "blood_glucose": 114
  }
}
```

---

## Output Schema (abbreviated)

```json
{
  "session_id": "...",
  "patient_token": "PATIENT_001",
  "differentials": [...],
  "drug_interactions": [...],
  "cited_findings": [...],
  "audit_report": {...},
  "confidence_signals": {...},
  "confidence_score": 77.62,
  "hitl_required": true,
  "escalate_immediately": false,
  "biosensor_summary": "...",
  "synthesis_narrative": "...",
  "loop_count": 0
}
```

---

## Setup

1. Clone the repo and navigate to the backend folder.

2. Create and activate a virtual environment:
    python -m venv venv
    venv\Scripts\activate  (Windows)
    source venv/bin/activate  (Mac/Linux)

3. Install dependencies:
    pip install -r requirements.txt

4. Install the medical NLP model:
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

5. Copy .env.example to .env and fill in your API keys:
    GROQ_API_KEY=your_groq_key
    GEMINI_API_KEY=optional_not_used
    DATABASE_URL=optional_supabase_url
    HITL_WEBHOOK_URL=http://localhost:8000/webhook/hitl-response
    CHROMA_PERSIST_PATH=./chroma_db

6. Seed ChromaDB (run once, takes 15-20 minutes):
    python scripts/seed_chroma.py

7. Start the server:
    uvicorn app.main:app --reload

8. Open the interactive docs:
    http://localhost:8000/docs

---

## Free Tier Limits

Groq free tier: 100,000 tokens per day, 30 requests per minute.
Each full consultation uses approximately 8,000-12,000 tokens across all agents.
You can run approximately 8-12 consultations per day on the free tier before hitting the daily limit.
To reset: create a new free Groq API key.

---

## Project Structure

```
backend/
├── app/
│   ├── agents/          # All 8 agent functions
│   ├── api/             # FastAPI routes
│   ├── confidence/      # Confidence scoring engine
│   ├── db/              # Database layer (optional)
│   ├── graph/           # LangGraph state and graph builder
│   ├── pii/             # spaCy PII redactor
│   ├── providers/       # Groq client and provider router
│   ├── schemas/         # Pydantic clinical schemas
│   ├── vector/          # ChromaDB client
│   ├── config.py
│   └── main.py
├── scripts/
│   └── seed_chroma.py   # PubMed seeding script
├── requirements.txt
└── .env.example
```

---

## Performance Benchmarks (free tier Groq)

Average response time per consultation: 8-15 seconds
Confidence scores observed across test cases: 64-80
Drug interactions caught: correctly identifies CYP3A4 interactions, ACE inhibitor combinations, beta blocker + insulin, warfarin + amiodarone
Cases tested: chest pain, hypertensive emergency, kidney stones, acute heart failure, bacterial meningitis, bipolar mania, ruptured AAA, preeclampsia, elderly polypharmacy

---

Built by Daksh Chawla
