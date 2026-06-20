import json
import os
import datetime
import concurrent.futures
import google.generativeai as genai
from pydantic import BaseModel, Field

from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langsmith import traceable

MAX_KEYWORD_COUNT = int(os.getenv("MAX_KEYWORD_COUNT", "50"))

class AuditFinding(BaseModel):
    pillar: str = Field(description="The name of the legal pillar being analyzed.")
    rating: str = Field(description="Strictly one of: 'Low', 'Medium', 'High', 'Critical', 'ERROR'.")
    finding: str = Field(description="A detailed explanation of compliance risk.")
    remediation: str = Field(description="Actionable steps to resolve the conflict.")
    citation: str = Field(description="The exact document filename and clause referenced.")

class AegisEngine:
    def __init__(self, api_key):
        load_dotenv() 
        secure_key = api_key or os.getenv("GEMINI_API_KEY")
        if not secure_key:
            raise ValueError("CRITICAL: API Key is missing! Check your .env file spelling.")
            
        genai.configure(api_key=secure_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.chat_model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2", 
            google_api_key=secure_key
        )

    def _log_eval_trace(self, pillar, context, generated_finding):
        """Silently logs the execution trace for RAGAS evaluation."""
        try:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "question": f"Perform a gap analysis between the internal policy and target law for the legal pillar: {pillar}",
                "contexts": [context], 
                "answer": generated_finding.get("finding", "ERROR"),
                "remediation": generated_finding.get("remediation", "ERROR"),
                "risk_rating": generated_finding.get("rating", "ERROR")
            }
            
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            file_path = os.path.join(root_dir, "ragas_eval_dataset.jsonl")
            
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            print(f"✅ RAGAS Trace Logged -> {file_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to log evaluation trace: {e}")

    def _format_error_msg(self, e):
        """Translates ugly cloud errors into clean UI messages."""
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
            return "Google API Quota Exceeded. Please wait a moment before trying again."
        return f"AI Processing Error: {str(e)}"

    def _get_hybrid_docs(self, query, session_id, doc_type, k_val=5):
        """
        Industry-Standard Hybrid Retriever:
        Combines Dense Semantic Vector Search (Pinecone) with Sparse Keyword Matching (BM25).
        """
        try:
            # 1. Access the dense vector store namespace
            vector_store = PineconeVectorStore(
                index_name="aegis-audit-index", 
                embedding=self.embeddings, 
                namespace=f"{session_id}_{doc_type}" 
            )
            
            # 2. Fetch all local session documents to construct the corpus for BM25
            all_session_docs = vector_store.similarity_search("", k=100)
            
            if not all_session_docs:
                return vector_store.similarity_search(query, k=k_val)

            # 3. Instantiate local in-memory BM25 Retriever
            bm25_retriever = BM25Retriever.from_documents(all_session_docs)
            bm25_retriever.k = k_val

            # 4. Instantiate semantic Pinecone Retriever
            pinecone_retriever = vector_store.as_retriever(search_kwargs={"k": k_val})

            # 5. Build Ensemble Hybrid Framework
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, pinecone_retriever],
                weights=[0.3, 0.7]
            )
            
            return ensemble_retriever.invoke(query)
            
        except Exception as e:
            print(f"⚠️ Hybrid retrieval bottleneck hit, falling back to pure vector search: {e}")
            vector_store = PineconeVectorStore(
                index_name="aegis-audit-index", 
                embedding=self.embeddings, 
                namespace=f"{session_id}_{doc_type}" 
            )
            return vector_store.similarity_search(query, k=k_val)

    @traceable(run_type="chain", name="Pillar_Analysis")
    def run_pillar_analysis(self, pillar, session_id):
        law_docs = self._get_hybrid_docs(pillar, session_id, "LAW", k_val=5)
        pol_docs = self._get_hybrid_docs(pillar, session_id, "POLICY", k_val=5)
        
        context = f"LAW: {[d.page_content for d in law_docs]}\nPOLICY: {[d.page_content for d in pol_docs]}"
        
        prompt = f"""
        You are a highly literal, strict Legal Compliance Auditor. 
        PILLAR TO ANALYZE: "{pillar}"
        
        CONTEXT PROVIDED:
        {context}
        
        STRICT GROUNDING INSTRUCTIONS:
        1. You must base your findings ONLY and EXACTLY on the text provided in the CONTEXT above.
        2. ZERO EXTERNAL KNOWLEDGE: Do not introduce external legal principles, standard industry practices, or common sense assumptions. 
        3. NO INFERENCE: If the context does not explicitly mention concepts related to the pillar (e.g., if the text says "Background Checks" but never mentions the word "Consent"), do NOT assume consent is implied. 
        4. If the provided POLICY context is completely silent on the LAW's requirements, your 'finding' must explicitly state: "The policy lacks explicit provisions for..." and base the Risk Rating on that exact omission.
        
        OUTPUT REQUIREMENTS:
        1. Compare the POLICY against the LAW focusing on '{pillar}'.
        2. Assess Risk Rating (Critical, High, Medium, Low).
        3. Write a professional 'finding' explaining the gap (strictly based on text).
        4. Write an actionable 'remediation' plan.
        5. Extract specific section numbers for the 'citation'.
        """
        
        try:
            import time
            start = time.time()

            res = self.model.generate_content(
                prompt, 
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=AuditFinding
                )
            )
            
            print(f"⏱️ [{pillar}] Reasoning time: {time.time() - start:.2f} seconds")

            finding = json.loads(res.text)
            self._log_eval_trace(pillar, context, finding)
            return finding
            
        except Exception as e:
            clean_error = self._format_error_msg(e)
            print(f"Error in {pillar}: {str(e)}") 
            return {
                "pillar": pillar, 
                "rating": "ERROR", 
                "finding": clean_error, 
                "remediation": "N/A", 
                "citation": "System Error"
            }

    @traceable(run_type="chain", name="Full_Compliance_Audit")
    def run_compliance_audit(self, session_id):
        pillars = ["Capacity", "Consent", "Consideration", "Legality", "Documentation", "Breach", "Termination", "Jurisdiction"]
        final_report = []

        print("🚀 Launching Parallel Audit Threads...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_pillar = {
                executor.submit(self.run_pillar_analysis, pillar, session_id): pillar 
                for pillar in pillars
            }
            
            for future in concurrent.futures.as_completed(future_to_pillar):
                pillar = future_to_pillar[future]
                try:
                    finding = future.result()
                    final_report.append(finding)
                except Exception as e:
                    print(f"❌ Thread crash on {pillar}: {e}")
                    final_report.append({
                        "pillar": pillar, 
                        "rating": "ERROR", 
                        "finding": "Thread Execution Failed", 
                        "remediation": "N/A", 
                        "citation": "System Error"
                    })
                    
        print("✅ All threads completed successfully!")
        return final_report

    def run_query(self, user_query, session_id, history):
        """Unified State-Aware Chat Router (Streaming Enabled)."""
        
        if session_id == "general_chat":
            prompt = f"You are Buddy, a helpful AI. Answer conversationally.\nHistory: {history}\nUser: {user_query}"
            try:
                response = self.chat_model.generate_content(prompt, stream=True)
                for chunk in response:
                    # 🛡️ SAFETY FIX 1: Check for valid parts before yielding text
                    if chunk.candidates and chunk.candidates[0].content.parts:
                        yield chunk.text
                    else:
                        yield "I'm not quite sure how to respond to that."
                return
            except Exception as e:
                yield self._format_error_msg(e)
                return

        # Chat now benefits from hybrid search capabilities
        law_docs = self._get_hybrid_docs(user_query, session_id, "LAW", k_val=4)
        pol_docs = self._get_hybrid_docs(user_query, session_id, "POLICY", k_val=4)
        
        context = ""
        for d in law_docs + pol_docs:
            src = d.metadata.get('source', 'Document')
            context += f"[{src}]: {d.page_content}\n"
        
        hybrid_prompt = f"""
        Role: You are Buddy, a Senior AI Legal & Compliance Consultant.
        
        INSTRUCTIONS:
        1. You have access to retrieved snippets from the user's uploaded Law and Policy documents below. Use them as your primary source of truth.
        2. YOU ARE A CONSULTANT, NOT JUST A SEARCH BAR. Synthesize the context and provide intelligent, professional advice. Use bullet points and clean spacing for readability.
        3. Answer general knowledge questions normally.
        4. Only say "I cannot find it in the documents" if they explicitly ask a factual question about the document's contents that truly isn't there.
        
        DOCUMENT CONTEXT: 
        {context}
        
        PREVIOUS CHAT HISTORY: 
        {history}
        
        USER QUESTION: {user_query}
        """
        
        try:
            response = self.model.generate_content(hybrid_prompt, stream=True)
            for chunk in response:
                if chunk.candidates and chunk.candidates[0].content.parts:
                    yield chunk.text
                else:
                    yield "I'm not quite sure how to respond to that."
        except Exception as e:
            yield self._format_error_msg(e)