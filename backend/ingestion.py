import os
import tempfile
from langchain_core.documents import Document
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class AegisIngestor:
    def __init__(self, namespace_name, api_key):
        """
        namespace_name will map to the session ID to isolate user data
        """
        self.namespace = namespace_name

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2", 
            google_api_key=api_key
        )

        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            num_workers= True,
            verbose=True
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "aegis-audit-index"
        self.index = self.pc.Index(self.index_name)

    def process_pdf(self, file_bytes, filename):
            """
            Processes PDF, parses layout, and pushes directly to isolated Pinecone namespace.
            """
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes) # Write the raw bytes from FastAPI
                temp_path = tmp.name
                
            try:

                parsed_docs = self.parser.load_data(temp_path)
                raw_text = "\n\n".join([doc.text for doc in parsed_docs])      
                
                chunks = self.splitter.split_text(raw_text)
                cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
                docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in cleaned_chunks]

                vector_db = PineconeVectorStore.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    index_name=self.index_name,
                    namespace=self.namespace
                )
                return vector_db

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def scrub_session_data(self):
        """
        Deletes all vectors for this specific namespace.
        Call this when the user logs out or ends the session.
        """ 
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            print(f"🧹 Successfully wiped ephemeral data for namespace: {self.namespace}")

        except Exception as e:
            print(f"⚠️ Failed to scrub data for {self.namespace}: str{e}")