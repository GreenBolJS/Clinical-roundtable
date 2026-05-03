from __future__ import annotations
import asyncpg
import json
import structlog
from app.config import get_settings
from app.schemas.clinical import ClinicalReport, PatientRecord

logger = structlog.get_logger(__name__)
settings = get_settings()


async def _connect() -> asyncpg.Connection:
    return await asyncpg.connect(settings.database_url)


async def get_pool() -> asyncpg.Connection:
    return await _connect()


async def init_db() -> None:
    """Create tables if they don't exist."""
    conn = await _connect()
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_token TEXT PRIMARY KEY,
                age_range TEXT,
                notes TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clinical_sessions (
                session_id TEXT PRIMARY KEY,
                patient_token TEXT REFERENCES patients(patient_token),
                report JSONB NOT NULL,
                hitl_required BOOLEAN DEFAULT FALSE,
                escalate_immediately BOOLEAN DEFAULT FALSE,
                confidence_score FLOAT,
                doctor_override JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
    finally:
        await conn.close()
    logger.info("db_tables_initialized")


async def create_patient(record: PatientRecord) -> PatientRecord:
    conn = await _connect()
    try:
        await conn.execute(
            """
            INSERT INTO patients (patient_token, age_range, notes)
            VALUES ($1, $2, $3)
            ON CONFLICT (patient_token) DO UPDATE
                SET age_range = EXCLUDED.age_range,
                    notes = EXCLUDED.notes
            """,
            record.patient_token,
            record.age_range,
            record.notes,
        )
    finally:
        await conn.close()
    logger.info("patient_created", patient_token=record.patient_token)
    return record


async def upsert_clinical_report(report: ClinicalReport) -> None:
    report_json = json.loads(report.model_dump_json())
    conn = await _connect()
    try:
        await conn.execute(
            """
            INSERT INTO patients (patient_token)
            VALUES ($1)
            ON CONFLICT (patient_token) DO NOTHING
            """,
            report.patient_token,
        )
        await conn.execute(
            """
            INSERT INTO clinical_sessions
                (session_id, patient_token, report, hitl_required, escalate_immediately, confidence_score, updated_at)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, NOW())
            ON CONFLICT (session_id) DO UPDATE
                SET report = EXCLUDED.report,
                    hitl_required = EXCLUDED.hitl_required,
                    escalate_immediately = EXCLUDED.escalate_immediately,
                    confidence_score = EXCLUDED.confidence_score,
                    updated_at = NOW()
            """,
            report.session_id,
            report.patient_token,
            json.dumps(report_json),
            report.hitl_required,
            report.escalate_immediately,
            report.confidence_score,
        )
    finally:
        await conn.close()
    logger.info("clinical_report_upserted", session_id=report.session_id)


async def get_clinical_report(session_id: str) -> ClinicalReport | None:
    conn = await _connect()
    try:
        row = await conn.fetchrow(
            "SELECT report FROM clinical_sessions WHERE session_id = $1",
            session_id,
        )
    finally:
        await conn.close()
    if row is None:
        return None
    return ClinicalReport.model_validate(json.loads(row["report"]))


async def update_doctor_override(
    session_id: str,
    override_payload: dict,
) -> None:
    conn = await _connect()
    try:
        await conn.execute(
            """
            UPDATE clinical_sessions
            SET doctor_override = $1::jsonb, updated_at = NOW()
            WHERE session_id = $2
            """,
            json.dumps(override_payload),
            session_id,
        )
    finally:
        await conn.close()
    logger.info("doctor_override_saved", session_id=session_id)


async def close_pool() -> None:
    """No pooled resources to close with direct asyncpg connections."""
    return None
