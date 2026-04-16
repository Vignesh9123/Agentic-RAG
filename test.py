from retriever import retriever
from graph.chains.retrieval_grader import GradeDocuments, grader_chain

def test_relevancy_yes():
    question = "What are the types of memory?"
    relevant_docs = retriever.invoke(question)
    doc_text = "\n".join([doc.page_content for doc in relevant_docs])
    response: GradeDocuments = grader_chain.invoke({
        "question": question,
        "document":doc_text
    })
    if response.binary_score.lower() == "yes":
        print("POSITIVE TEST SUCCESSFUL")
    else:
        print("POSITIVE TEST FAILED")

def test_relevancy_no():
    question = "What is OpenClaw?"
    relevant_docs = retriever.invoke(question)
    doc_text = "\n".join([doc.page_content for doc in relevant_docs])
    response: GradeDocuments = grader_chain.invoke({
        "question": question,
        "document":doc_text
    })
    if response.binary_score.lower() == "no":
        print("NEGATIVE TEST SUCCESSFUL")
    else:
        print("NEGATIVE TEST FAILED")


test_relevancy_yes()
test_relevancy_no()