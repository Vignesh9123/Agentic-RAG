
from graph.constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from langgraph.graph import StateGraph, END

from graph.state import GraphState

from graph.nodes.retrieve import retrieve
from graph.nodes.grade_documents import grade_documents
from graph.nodes.web_search import web_search
from graph.nodes.generate import generate

def make_grade_to_next_edge(state: GraphState) -> GraphState:
    if state['web_search']:
        return WEB_SEARCH
    else:
        return GENERATE

graph_builder = StateGraph(GraphState)

graph_builder.add_node(RETRIEVE, retrieve)
graph_builder.add_node(GRADE_DOCUMENTS, grade_documents)
graph_builder.add_node(WEB_SEARCH, web_search)
graph_builder.add_node(GENERATE, generate)

graph_builder.set_entry_point(RETRIEVE)

graph_builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)
graph_builder.add_conditional_edges(
    GRADE_DOCUMENTS,
    make_grade_to_next_edge
)

graph_builder.add_edge(WEB_SEARCH, GENERATE)
graph_builder.add_edge(GENERATE, END)

agentic_rag_graph = graph_builder.compile()