from typing import Dict
from graph.chains.retrieval_grader import grader_chain
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, any]:
    print("IN grade_documents node")
    question = state['question']
    documents = state['documents']

    relevant_docs = []
    web_search = False
    for doc in documents:
        response = grader_chain.invoke({
            "question":question,
            "document": doc
        })
        if response.binary_score.lower() == "yes":
            print("Document relevant")
            relevant_docs.append(doc)
        else:
            print("Document not relevant", doc.page_content)
            web_search = True
    
    return {"documents": relevant_docs, "web_search":web_search}