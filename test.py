from dotenv import load_dotenv
load_dotenv()
from retriever import retriever
from graph.chains.retrieval_grader import GradeDocuments, grader_chain
from graph.chains.generation import generation_chain

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
    
def test_generation():
    question = "What are the types of Agent memory?"
    relevant_docs = retriever.invoke(question)
    doc_text = "\n".join([doc.page_content for doc in relevant_docs])
    response = generation_chain.invoke({
        "question":question,
        "context":doc_text
    })
    print("GENERATION TEST SUCCESSFUL")




test_relevancy_yes()
test_relevancy_no()
test_generation()