from __future__ import annotations
import structlog
from app.schemas.clinical import ConfidenceSignals

logger = structlog.get_logger(__name__)


class ConfidenceEngine:
    """
    Computes a composite confidence score from four weighted signals.

    Signal weights:
        Signal 1 — CMO Structured Self-Report:    30%
        Signal 2 — Inter-Agent Agreement:          25%
        Signal 3 — Citation Density:               20%
        Signal 4 — Adversarial Audit:              25%

    Final score is 0–100.
    """

    # Top-level signal weights
    WEIGHT_CMO_SELF_REPORT = 0.30
    WEIGHT_INTER_AGENT = 0.25
    WEIGHT_CITATION = 0.20
    WEIGHT_ADVERSARIAL = 0.25

    # Signal 1 sub-weights
    CMO_SUB_DIAGNOSIS_CONFIDENCE = 0.35
    CMO_SUB_EVIDENCE_STRENGTH = 0.30
    CMO_SUB_CONTRAINDICATION_RISK = 0.20  # inverted
    CMO_SUB_DIFFERENTIAL_COMPLETENESS = 0.15

    # Signal 3 sub-weights
    CITATION_SUB_DENSITY = 0.50
    CITATION_SUB_RECENCY = 0.30
    CITATION_SUB_RCT_RATIO = 0.20

    def compute(self, signals: ConfidenceSignals) -> float:
        """
        Compute the composite confidence score.
        Returns a float in the range [0.0, 100.0].
        """

        # ── Signal 1: CMO Self-Report ──────────────────────────────────────────
        cmo = signals.cmo_self_report
        contraindication_inverted = 1.0 - cmo.contraindication_risk
        signal1 = (
            cmo.diagnosis_confidence * self.CMO_SUB_DIAGNOSIS_CONFIDENCE
            + cmo.evidence_strength * self.CMO_SUB_EVIDENCE_STRENGTH
            + contraindication_inverted * self.CMO_SUB_CONTRAINDICATION_RISK
            + cmo.differential_completeness * self.CMO_SUB_DIFFERENTIAL_COMPLETENESS
        )

        # ── Signal 2: Inter-Agent Agreement ───────────────────────────────────
        signal2 = signals.inter_agent_agreement.agreement_score

        # ── Signal 3: Citation Density ─────────────────────────────────────────
        cd = signals.citation_density
        density_score = min(cd.citation_count / 10.0, 1.0)
        signal3 = (
            density_score * self.CITATION_SUB_DENSITY
            + cd.recency_score * self.CITATION_SUB_RECENCY
            + cd.rct_ratio * self.CITATION_SUB_RCT_RATIO
        )

        # ── Signal 4: Adversarial Audit ───────────────────────────────────────
        adv = signals.adversarial
        unanswered_count = max(0, adv.challenges_raised - adv.challenges_answered)
        signal4 = max(0.0, 1.0 - unanswered_count * 0.08)

        # ── Weighted Final Score ──────────────────────────────────────────────
        raw_score = (
            signal1 * self.WEIGHT_CMO_SELF_REPORT
            + signal2 * self.WEIGHT_INTER_AGENT
            + signal3 * self.WEIGHT_CITATION
            + signal4 * self.WEIGHT_ADVERSARIAL
        )

        final_score = raw_score * 100.0

        logger.info(
            "confidence_computed",
            signal1_cmo=round(signal1, 4),
            signal2_agreement=round(signal2, 4),
            signal3_citation=round(signal3, 4),
            signal4_adversarial=round(signal4, 4),
            final_score=round(final_score, 2),
        )

        return round(final_score, 2)


def compute_inter_agent_agreement(
    pathologist_conditions: list[str],
    literature_conditions: list[str],
    biosensor_conditions: list[str],
) -> float:
    path = [c.lower().strip() for c in pathologist_conditions]
    lit = [c.lower().strip() for c in literature_conditions]
    bio = [c.lower().strip() for c in biosensor_conditions]

    if not path:
        return 0.5

    scores = []

    # Check if literature top pick is in pathologist list (substring match)
    if lit:
        lit_match = any(
            any(l in p or p in l for p in path)
            for l in lit
        )
        scores.append(1.0 if lit_match else 0.2)

    # Check if biosensor picks overlap with pathologist list
    if bio:
        bio_matches = sum(
            1 for b in bio
            if any(b in p or p in b for p in path)
        )
        scores.append(min(1.0, bio_matches / len(bio)))

    return sum(scores) / len(scores) if scores else 0.5


def compute_citation_density_signals(cited_findings) -> tuple[int, float, float]:
    """
    Compute citation density sub-signals from a list of CitedFinding objects.
    Returns (citation_count, recency_score, rct_ratio).
    """
    if not cited_findings:
        return 0, 0.0, 0.0

    citation_count = len(cited_findings)

    # Recency score: based on evidence level proxy (RCTs tend to be more recent)
    level_recency = {
        "RCT": 1.0,
        "meta-analysis": 0.9,
        "cohort": 0.7,
        "case-report": 0.5,
        "expert-opinion": 0.4,
    }
    recency_scores = [level_recency.get(f.evidence_level, 0.5) for f in cited_findings]
    recency_score = sum(recency_scores) / len(recency_scores)

    # RCT ratio
    rct_count = sum(1 for f in cited_findings if f.evidence_level in ("RCT", "meta-analysis"))
    rct_ratio = rct_count / citation_count

    return citation_count, recency_score, rct_ratio
