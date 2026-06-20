# Aegis-Audit: High-Speed Advanced RAG Compliance Engine

## 1. Problem Statement
In enterprise environments, executing a comprehensive gap analysis between internal corporate policies and evolving statutory laws is a grueling, manual process. Compliance officers spend weeks cross-referencing hundreds of pages of legalese. While standard Large Language Models (LLMs) attempt to solve this, they suffer from two fatal flaws in legal contexts: **hallucination** (inventing clauses that do not exist) and **latency** (taking minutes to process massive document contexts). 

Aegis-Audit was built to solve this. We needed a system that was strictly grounded in the provided text, capable of parallelized multi-pillar reasoning, and optimized for sub-second retrieval.

## 2. Solution Overview
Aegis-Audit is an automated, Advanced Retrieval-Augmented Generation (RAG) compliance engine. It allows users to upload a Target Law PDF and an Internal Policy PDF, and instantly runs an 8-pillar legal gap analysis (Capacity, Consent, Consideration, Legality, Documentation, Breach, Termination, Jurisdiction). 

The system utilizes an advanced Hybrid Retrieval pipeline, strict system prompts to prevent hallucination, and a distributed caching layer to achieve enterprise-grade speeds and high-fidelity grounding.

## 3. Architecture & Design Decisions
The architecture was designed with a strict separation of concerns, moving from a standard monolithic script to a cloud-native, asynchronous pipeline:

* **Ingestion:** PDFs are parsed, chunked, and embedded into a session-isolated Pinecone Vector namespace to prevent cross-contamination of user data.
* **Retrieval (Advanced RAG):** A bespoke Hybrid Retriever merges dense semantic search (Pinecone) with sparse exact-keyword matching (BM25) to ensure hyper-specific legal terms are never missed.
* **Execution:** The 8-pillar audit does not run sequentially. It utilizes a `ThreadPoolExecutor` to run all 8 prompts concurrently against the LLM, drastically cutting down total generation time.
* **Caching:** An Upstash Redis layer intercepts identical queries via document hashing. If a document hash matches a previous audit, the LLM execution is bypassed entirely.

## 4. Tech Stack
* **Backend Framework:** `FastAPI` — Chosen for its native asynchronous capabilities and Pydantic validation, allowing non-blocking I/O during heavy LLM network calls.
* **LLM Engine:** `Google Gemini 2.5 Flash` — Selected for its massive context window and rapid generation speeds, ideal for parsing long legal chunks.
* **Vector Database:** `Pinecone` — Serverless architecture with incredibly low-latency approximate nearest neighbor (ANN) search.
* **Caching & Throttling:** `Upstash Redis` — Chosen for its serverless REST API compatibility, handling both semantic caching and distributed rate-limiting.
* **Evaluation (CI/CD):** `Ragas` — Integrated to scientifically measure hallucination rates (Faithfulness) using an LLM-as-a-Judge architecture (Llama-3.3 70B via Groq).
* **Frontend:** `React + Vite` — Blazing fast build tooling and HMR, deployed independently to Vercel's edge network.

## 5. Results & Metrics
* **Latency Reduction:** By implementing the Upstash Redis caching layer, repeated document audits were reduced from an average of **27.4 seconds** to **<0.5 seconds** (a ~6,400% performance increase).
* **Hallucination Defense:** The pipeline was rigorously evaluated against a 30-document ground-truth dataset using the Ragas framework, achieving an **82.14% Faithfulness score**. This proves the model relies strictly on the provided PDFs in over 4 out of 5 interactions without injecting outside knowledge.
* **Parallel Execution:** Moving from sequential to threaded generation reduced the full 8-pillar audit time by roughly **65%**.

## 6. Key Features
* **Dynamic Hybrid Retrieval:** Tunable `MAX_KEYWORD_COUNT` environment variable allowing MLOps adjustments to the EnsembleRetriever (70% Semantic / 30% Keyword).
* **Enterprise Safeguards:** Global Redis-backed rate limiting (e.g., 5 audits/minute) to prevent API exhaustion and DDoS attacks.
* **Strict CORS Shielding:** Backend API is locked to the specific Vercel production frontend.
* **Streaming UI:** Chat interface streams tokens in real-time, handling `finish_reason: 1` edge cases gracefully to prevent frontend crashes.
* **Automated Data Scrubbing:** Secure `/logout` endpoint instantly fires background tasks to wipe session namespaces from Pinecone.

## 7. Installation & Setup

**Prerequisites:** Python 3.12+, Node.js.

1.  **Clone & Environment:**
    ```bash
    git clone [https://github.com/ShivabasavaM/AEGIS-Auditor-BuddyAI_Auditor-.git](https://github.com/ShivabasavaM/AEGIS-Auditor-BuddyAI_Auditor-.git)
    cd AEGIS-Auditor-BuddyAI_Auditor-
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables (`.env`):**
    ```env
    GEMINI_API_KEY="your_google_key"
    PINECONE_API_KEY="your_pinecone_key"
    REDIS_URL="rediss://default:password@url.upstash.io:6379"
    MAX_KEYWORD_COUNT="50"
    ```

4.  **Run Locally:**
    ```bash
    # Terminal 1: Backend
    uvicorn backend.main:app --reload

    # Terminal 2: Frontend
    cd frontend
    npm install
    npm run dev
    ```

## 8. Usage Examples
* **Upload:** POST to `/api/v1/upload` with a `law_file` and `policy_file`. Returns a unique `session_id`.
* **Audit:** POST to `/api/v1/audit` with the `session_id`. The engine calculates a document hash; if found in Redis, it instantly returns the cached JSON report. If not, it executes the parallel 8-pillar LLM calls.
* **Chat:** POST to `/api/v1/chat`. The engine routes between general conversation and Hybrid RAG document lookup based on the query.

## 9. Performance & Reliability
* **Infrastructure:** The frontend is hosted on Vercel's edge network for 99.99% UI uptime. The heavy FastAPI backend is hosted persistently on Render.
* **Continuous Availability:** Because free-tier Render instances spin down after 15 minutes of inactivity, an UptimeRobot heartbeat monitor is configured to hit the backend's `/` health-check endpoint every 5 minutes, ensuring cold-starts (which can take >50s) never impact the end-user.
* **Fault Tolerance:** The `run_pillar_analysis` handles individual thread crashes gracefully, returning an "ERROR" JSON block for a specific pillar rather than crashing the entire 8-pillar HTTP response.

## 10. Design Trade-offs
* **Serverless vs. Persistent Hosting:** Initially, deploying the Python backend to Vercel was considered. However, Vercel's strict 10-second serverless timeout would forcefully kill the execution during a cache-miss audit (which takes ~27s). The trade-off was accepting the slightly higher maintenance overhead of a persistent Render instance to guarantee execution completion.
* **Caching Strategy:** The Redis TTL is set to 24 hours for session hashes and 7 days for the actual audit reports. This balances cloud storage costs against the compute costs of re-running the Gemini API.

## 11. Limitations & Future Work
* **Context Window Bleed:** While Gemini 2.5 Flash has a large context window, feeding 100+ pages of retrieved legal text can still cause attention degradation ("lost in the middle" phenomenon). 
* **Future - Agentic Routing (LangGraph):** The current system relies on a linear, Advanced RAG pipeline. Version 3.0 will implement `LangGraph` to create a true Agentic state machine, allowing the LLM to self-grade its retrieved documents, decide if it needs to rewrite the user's query, and iteratively loop back to the vector database before generating a final answer.

## 12. References
* [Ragas Framework Documentation](https://docs.ragas.io/)
* [Google Gemini API Docs](https://ai.google.dev/docs)
* [Pinecone Hybrid Search Guide](https://docs.pinecone.io/guides/data/understanding-hybrid-search)
* [LangChain EnsembleRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble)
