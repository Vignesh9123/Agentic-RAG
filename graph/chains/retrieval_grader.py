from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()

class GradeDocuments(BaseModel):
    """Binary score to check relevance of the given document to the given question"""
    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)
llm_with_structured_output = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system),
    HumanMessagePromptTemplate.from_template("Retrieved document:\n\n{document}\n\nUser question: {question}")
])

grader_chain = prompt_template | llm_with_structured_output