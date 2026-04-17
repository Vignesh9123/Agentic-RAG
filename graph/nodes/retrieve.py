from typing import Dict

from graph.state import GraphState
from retriever import retriever

def retrieve(state: GraphState) -> Dict[str, any]:
    print("IN retrieve node")
    question = state['question']
    documents = retriever.invoke(question)
    return {"documents":documents}