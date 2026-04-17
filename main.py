from dotenv import load_dotenv
load_dotenv()

from graph.graph import agentic_rag_graph

output = agentic_rag_graph.invoke({
    "question":"What are the types of agent memory?"
})

print("GRAPH output", output)