from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent",
]

loaded_docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [doc for docs in loaded_docs for doc in docs]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250, chunk_overlap=50
)

splitted_docs = text_splitter.split_documents(docs_list)

vector_store = Chroma.from_documents(
    documents=splitted_docs,
    collection_name="agentic-rag",
    embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory='./.chroma'
)
print("Ingestion doneee")
