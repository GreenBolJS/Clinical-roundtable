from __future__ import annotations
import structlog
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import router

# ── Structured logging setup ─────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()
settings = get_settings()


# ── Lifespan: startup / shutdown ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup, clean up on shutdown."""
    logger.info("startup_begin", app="clinical_roundtable")

    # Initialize database tables
    try:
        from app.db.supabase import init_db
        await init_db()
        logger.info("startup_db_initialized")
    except Exception as exc:
        logger.warning("startup_db_init_failed", error=str(exc), message="Continuing without DB")

    # Pre-warm ChromaDB connection
    try:
        from app.vector.chroma import get_collection
        col = get_collection()
        logger.info("startup_chroma_ready", document_count=col.count())
    except Exception as exc:
        logger.warning("startup_chroma_failed", error=str(exc))

    # Pre-compile LangGraph
    try:
        from app.graph.graph import get_graph
        get_graph()
        logger.info("startup_graph_compiled")
    except Exception as exc:
        logger.warning("startup_graph_failed", error=str(exc))

    logger.info("startup_complete")
    yield

    # Cleanup
    try:
        from app.db.supabase import close_pool
        await close_pool()
        logger.info("shutdown_db_pool_closed")
    except Exception as exc:
        logger.warning("shutdown_db_close_failed", error=str(exc))

    logger.info("shutdown_complete")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="The Clinical Roundtable",
    description=(
        "Recursive Multi-Agent Clinical Intelligence System. "
        "Parallel specialist agents, LangGraph orchestration, confidence scoring, and HITL escalation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "The Clinical Roundtable",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
