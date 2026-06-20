import os
import time
import logging
import traceback
import hashlib
import json
import uuid
import tempfile
from typing import List

import redis.asyncio as redis_async

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from dotenv import load_dotenv

load_dotenv()
from backend.report_gen import generate_docx_report
from backend.ingestion import AegisIngestor
from backend.engine import AegisEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL")

redis_client = redis_async.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if REDIS_URL:
        try:
            await redis_client.ping()
            logger.info("✅ Connected to Redis cache successfully.")
            await FastAPILimiter.init(redis_client)
            logger.info("🛡️ Rate Limiter initialized.")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed. Caching/Limiting disabled. Error: {e}")
    else:
        logger.warning("⚠️ REDIS_URL not found in environment. Caching disabled.")
    
    yield
    await redis_client.close()

app = FastAPI(title="Aegis-auditor", version="2.0", lifespan=lifespan)

ALLOWED_ORIGINS = [
    "http://localhost:3000", 
    "https://aegis-audit-acr.vercel.app", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class AuditRequest(BaseModel):
    session_id: str

class LogoutRequest(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    query: str
    history: List[str]

# --- Health Check (For Render & UptimeRobot) ---
@app.get("/")
async def health_check():
    """Provides a 200 OK status for UptimeRobot and Render health checks."""
    return {"status": "online", "service": "Aegis-Audit API", "version": "2.0"}

@app.post("/api/v1/chat", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def chat_with_docs(request: ChatRequest):
    """Handles real-time streaming chat."""
    try:
        engine = AegisEngine(api_key=os.getenv("GEMINI_API_KEY"))
        
        return StreamingResponse(
            engine.run_query(request.query, request.session_id, request.history),
            media_type="text/plain"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/export")
async def export_report(request: Request):
    """Generates the .docx binary from the audit JSON."""
    try:
        data = await request.json()
        session_id = data.get("session_id") or "general"
        report_data = data.get("report_data")
        
        if not report_data:
            raise ValueError("No report data provided from the frontend.")

        metadata = {
            "law_name": "Target Law Document",
            "policy_name": "Internal Policy Document"
        }

        doc_buffer = generate_docx_report(report_data, metadata)
        
        return StreamingResponse(
            doc_buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename=Aegis_Audit_{session_id}.docx"
            }
        )
            
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export crashed: {str(e)}")


@app.post("/api/v1/upload", dependencies=[Depends(RateLimiter(times=3, seconds=60))])
async def upload_documents(
    law_file: UploadFile = File(...), 
    policy_file: UploadFile = File(...)
):
    """Ingests documents into Pinecone and generates a caching signature."""
    if not law_file.filename.endswith('.pdf') or not policy_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    try:
        start_time = time.time()
        
        law_bytes = await law_file.read()
        policy_bytes = await policy_file.read()

        MAX_FILE_SIZE_MB = 10 
        MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
        
        if len(law_bytes) > MAX_BYTES or len(policy_bytes) > MAX_BYTES:
            raise HTTPException(
                status_code=413, 
                detail=f"Payload Too Large. Maximum allowed file size is {MAX_FILE_SIZE_MB}MB."
            )

        logger.info(f"[{session_id}] Uploaded Law: {len(law_bytes) / (1024*1024):.2f} MB | Policy: {len(policy_bytes) / (1024*1024):.2f} MB")

        combined_hash = hashlib.sha256(law_bytes + policy_bytes).hexdigest()
        if REDIS_URL:
            try:
                await redis_client.setex(f"session_hash:{session_id}", 86400, combined_hash)
            except Exception as e:
                logger.error(f"Redis write error on upload: {e}")

        law_ingestor = AegisIngestor(namespace_name=f"{session_id}_LAW", api_key=os.getenv("GEMINI_API_KEY"))
        law_ingestor.process_pdf(law_bytes, law_file.filename)
        
        policy_ingestor = AegisIngestor(namespace_name=f"{session_id}_POLICY", api_key=os.getenv("GEMINI_API_KEY"))
        policy_ingestor.process_pdf(policy_bytes, policy_file.filename)
        
        end_time = time.time()
        logger.info(f"[{session_id}] Total Ingestion Time: {end_time - start_time:.2f} seconds")
        
        return {
            "status": "success", 
            "message": "Documents indexed successfully.",
            "session_id": session_id 
        }
    except Exception as e:
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/v1/audit", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def run_audit(request: AuditRequest):
    """Executes the Agentic 8-Pillar Gap Analysis with Redis Caching bypass."""
    try:
        combined_hash = None
        
        if REDIS_URL:
            try:
                combined_hash = await redis_client.get(f"session_hash:{request.session_id}")
                if combined_hash:
                    cached_report = await redis_client.get(f"aegis_cache:{combined_hash}")
                    if cached_report:
                        logger.info(f"⚡ [CACHE HIT] Bypassing LLM execution for session: {request.session_id}")
                        return {"status": "success", "report": json.loads(cached_report), "cached": True}
            except Exception as e:
                logger.error(f"Redis read error on audit: {e}")

        logger.info(f"⚙️ [CACHE MISS] Running full 8-pillar LLM execution for: {request.session_id}")
        
        engine = AegisEngine(api_key=os.getenv("GEMINI_API_KEY"))
        report = engine.run_compliance_audit(request.session_id)
        
        if REDIS_URL and combined_hash:
            try:
                await redis_client.setex(f"aegis_cache:{combined_hash}", 604800, json.dumps(report))
            except Exception as e:
                logger.error(f"Redis write error on audit: {e}")

        return {"status": "success", "report": report, "cached": False}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Audit Crash: {str(e)}")


@app.post("/api/v1/logout")
async def logout(request: LogoutRequest, background_tasks: BackgroundTasks):
    """Wipes user data from Pinecone instantly."""
    law_ingestor = AegisIngestor(namespace_name=f"{request.session_id}_LAW", api_key=os.getenv("GEMINI_API_KEY"))
    policy_ingestor = AegisIngestor(namespace_name=f"{request.session_id}_POLICY", api_key=os.getenv("GEMINI_API_KEY"))

    background_tasks.add_task(law_ingestor.scrub_session_data)
    background_tasks.add_task(policy_ingestor.scrub_session_data)
    
    return {"status": "success", "message": "Session data queued for deletion."}