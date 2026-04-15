from typing import Dict

from graph.state import GraphState
from retriever import retriever

def retrieve(state: GraphState) -> Dict[str, any]:
    question = state['question']
    documents = retriever.invoke(question)
    return {"documents":documents}