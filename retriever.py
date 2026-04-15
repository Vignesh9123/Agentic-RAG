from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
retriever = Chroma(
    collection_name="agentic-rag",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory='./.chroma'
).as_retriever(
    search_kwargs={"k":3}
)