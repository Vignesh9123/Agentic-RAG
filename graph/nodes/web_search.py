from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

from graph.state import GraphState

web_search_tool = TavilySearch(
    max_results=3
)

def web_search(state: GraphState) -> GraphState:
    question = state['question']
    documents = state['documents']

    web_results = web_search_tool.invoke({
        "query":question
    })
    search_content = "\n".join([
        web_result["content"] for web_result in web_results["results"]
    ])
    search_doc = Document(search_content)
    if documents is not None:
        documents.append(search_doc)
    else:
        documents = [search_doc]
    
    return {"documents":documents}
