import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

evaluator_llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.0, 
    api_key=os.getenv("GROQ_API_KEY")
)

