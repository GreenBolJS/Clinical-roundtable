from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)

# Module-level initialization
_SPACY_AVAILABLE = False
_nlp = None

try:
    import spacy
    try:
        _nlp = spacy.load("en_core_sci_md")
        _SPACY_AVAILABLE = True
        logger.info("pii_redactor_mode", mode="en_core_sci_md")
    except OSError:
        try:
            _nlp = spacy.load("en_core_web_sm")
            _SPACY_AVAILABLE = True
            logger.warning("pii_redactor_fallback", mode="en_core_web_sm", message="en_core_sci_md unavailable")
        except OSError:
            _SPACY_AVAILABLE = False
            logger.warning("pii_redactor_mode", mode="regex", message="No spaCy model found")
except ImportError:
    _SPACY_AVAILABLE = False
    logger.warning("pii_redactor_mode", mode="regex", message="spaCy not installed")


@dataclass
class PIISession:
    """Per-session mapping of token -> real value and real value -> token."""
    session_id: str
    _token_to_real: dict[str, str] = field(default_factory=dict)
    _real_to_token: dict[str, str] = field(default_factory=dict)
    _patient_counter: int = 0
    _dob_counter: int = 0
    _mrn_counter: int = 0
    _name_counter: int = 0

    def _next_patient_token(self) -> str:
        self._patient_counter += 1
        return f"[PATIENT_{self._patient_counter:03d}]"

    def _next_dob_token(self) -> str:
        self._dob_counter += 1
        return f"[DOB_{self._dob_counter:03d}]"

    def _next_mrn_token(self) -> str:
        self._mrn_counter += 1
        return f"[MRN_{self._mrn_counter:03d}]"

    def _next_name_token(self) -> str:
        self._name_counter += 1
        return f"[NAME_{self._name_counter:03d}]"

    def get_or_create_token(self, real_value: str, token_type: str = "name") -> str:
        if real_value in self._real_to_token:
            return self._real_to_token[real_value]
        if token_type == "patient":
            token = self._next_patient_token()
        elif token_type == "dob":
            token = self._next_dob_token()
        elif token_type == "mrn":
            token = self._next_mrn_token()
        else:
            token = self._next_name_token()
        self._token_to_real[token] = real_value
        self._real_to_token[real_value] = token
        return token

    def restore(self, text: str) -> str:
        """Replace all tokens back to real values."""
        for token, real in self._token_to_real.items():
            text = text.replace(token, real)
        return text


# Global in-memory store of sessions keyed by session_id
_sessions: dict[str, PIISession] = {}


def get_or_create_session(session_id: str) -> PIISession:
    if session_id not in _sessions:
        _sessions[session_id] = PIISession(session_id=session_id)
    return _sessions[session_id]


# ── Regex patterns ─────────────────────────────────────────────────────────────
# Date of birth patterns: MM/DD/YYYY, YYYY-MM-DD, Month DD YYYY, etc.
_DOB_PATTERNS = [
    re.compile(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})\b"),
    re.compile(r"\b(\d{4})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b"),
    re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(0?[1-9]|[12]\d|3[01]),?\s+(\d{4})\b",
        re.IGNORECASE,
    ),
]

# Medical Record Number patterns
_MRN_PATTERNS = [
    re.compile(r"\bMRN[:\s#]*(\d{5,12})\b", re.IGNORECASE),
    re.compile(r"\bMedical Record[:\s#]*(\d{5,12})\b", re.IGNORECASE),
    re.compile(r"\bPatient ID[:\s#]*(\d{5,12})\b", re.IGNORECASE),
]

# SSN pattern (should not appear in clinical notes but guard anyway)
_SSN_PATTERN = re.compile(r"\b(\d{3}-\d{2}-\d{4})\b")

# Phone numbers
_PHONE_PATTERN = re.compile(r"\b(\+?1?\s*[-.]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b")

# Email
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


def _regex_redact(text: str, session: PIISession) -> str:
    """Apply regex-based PII redaction."""
    # DOB
    for pattern in _DOB_PATTERNS:
        def _dob_replacer(m: re.Match) -> str:
            return session.get_or_create_token(m.group(0), "dob")
        text = pattern.sub(_dob_replacer, text)

    # MRN
    for pattern in _MRN_PATTERNS:
        def _mrn_replacer(m: re.Match) -> str:
            return session.get_or_create_token(m.group(0), "mrn")
        text = pattern.sub(_mrn_replacer, text)

    # SSN → treat as patient token
    def _ssn_replacer(m: re.Match) -> str:
        return session.get_or_create_token(m.group(0), "patient")
    text = _SSN_PATTERN.sub(_ssn_replacer, text)

    # Phone → name token
    def _phone_replacer(m: re.Match) -> str:
        return session.get_or_create_token(m.group(0), "name")
    text = _PHONE_PATTERN.sub(_phone_replacer, text)

    # Email → name token
    def _email_replacer(m: re.Match) -> str:
        return session.get_or_create_token(m.group(0), "name")
    text = _EMAIL_PATTERN.sub(_email_replacer, text)

    return text


def _spacy_redact(text: str, session: PIISession) -> str:
    """Use scispaCy NER to redact PERSON entities."""
    doc = _nlp(text)
    # Collect spans to replace (process in reverse to preserve offsets)
    spans = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON",):
            spans.append((ent.start_char, ent.end_char, ent.text))

    if not spans:
        return text

    # Replace from right to left
    chars = list(text)
    for start, end, ent_text in sorted(spans, key=lambda x: x[0], reverse=True):
        token = session.get_or_create_token(ent_text, "name")
        chars[start:end] = list(token)

    return "".join(chars)


class PIIRedactor:
    """
    Orchestrates PII redaction using scispaCy (if available) + regex.
    Always returns (redacted_text, session).
    """

    def redact(self, text: str, session_id: str) -> tuple[str, PIISession]:
        session = get_or_create_session(session_id)

        # 1. Apply regex patterns first (catches structured PII like MRN, DOB)
        redacted = _regex_redact(text, session)

        # 2. Apply spaCy NER if available
        if _SPACY_AVAILABLE and _nlp is not None:
            try:
                redacted = _spacy_redact(redacted, session)
            except Exception as exc:
                logger.warning("spacy_redact_error", error=str(exc))

        return redacted, session

    def redact_dict(self, data: dict, session_id: str) -> tuple[dict, PIISession]:
        """Recursively redact string values in a dict."""
        session = get_or_create_session(session_id)
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                redacted, _ = self.redact(value, session_id)
                result[key] = redacted
            elif isinstance(value, list):
                result[key] = [
                    self.redact(item, session_id)[0] if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                result[key], _ = self.redact_dict(value, session_id)
            else:
                result[key] = value
        return result, session


# Module-level singleton
_redactor: PIIRedactor | None = None


def get_redactor() -> PIIRedactor:
    global _redactor
    if _redactor is None:
        _redactor = PIIRedactor()
    return _redactor
