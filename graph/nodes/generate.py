from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> GraphState:
    print("IN generate node")
    question = state['question']
    documents = state['documents']

    context = "\n".join(doc.page_content for doc in documents)

    generated_content = generation_chain.invoke({
        "question":question,
        "context":context
    })
    return {"generated_content": generated_content}


