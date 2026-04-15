from typing import List, TypedDict

from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    generated_content: str
    web_search: bool
    documents: List[Document]