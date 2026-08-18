from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from typing import TypedDict
# from langchain.agents import Create_Agent

llm = ChatOllama(model='gemma4:12b')

class IdeaState(TypedDict):
    department_name: str
    company_name: str
    ml_ideas: str
    genai_ideas: str
    agentic_ideas: str
    functions: str
    final_ideas: str

def department_functions(state: IdeaState):
    print("\nUnderstanding Department Functions\n")
    department_name = state['department_name']

    prompt = f"""List all the funtions,Sub-Functions,Tools & Systems Used within this department: {department_name}"""

    response = llm.invoke(prompt)
    print("\nIdentified below Department Functions\n: ",response.content)

    return ({"functions":response.content})


def ML_ideas(state: IdeaState):
    print("\nSearching for ML solutions\n")
    department_name = state['department_name']

    prompt = f"""Fetch all the Machine Learning ideas or solutions that are being implemented in the 
    organizations across the world in the {department_name} department."""

    response = llm.invoke(prompt)
    print("\nML Solutions:\n",response.content)

    return {"ml_ideas":response.content}

def genai_ideas(state: IdeaState):
    print("\nSearching for genai solutions\n")
    department_name = state['department_name']

    prompt = f"""Fetch all the generative ai ideas or solutions that are being implemented in the 
    organizations across the world in the {department_name} department."""

    response = llm.invoke(prompt)
    print("\nGen ai solutions\n",response.content)

    return {"genai_ideas":response.content}

def agentic_ideas(state: IdeaState):
    print("\nSearching for the agentic solutions\n")
    department_name = state['department_name']

    prompt = f"""Fetch all the agentic ai ideas or solutions that are being implemented in the 
    organizations across the world in the {department_name} department."""

    response = llm.invoke(prompt)
    print("\nAgentic Solutions\n",response.content)

    return {"agentic_ideas":response.content}

def idea_aggregator(state: IdeaState):
    print("\nAggregate the Ideas\n")
    ml = state['ml_ideas']
    genai = state['genai_ideas']
    agentic = state['agentic_ideas']

    department_name = state['department_name']

    prompt = f"""Below are all the generative ai, agentic ai and ML ideas/solutions that are being implemented in the 
    organizations in the {department_name} department. I want you to aggregate each idea and create a single comphresensive 
    list.\n
    ml ideas: {ml}\n
gen ai ideas: {genai}\n
agentic: {agentic}"""

    response = llm.invoke(prompt)
    print("\nFinal Ideas\n",response.content)

    return {"final_ideas":response.content}

graph = StateGraph(IdeaState)

graph.add_node("ML_ideas",ML_ideas)
graph.add_node("genai_ideas",genai_ideas)
graph.add_node("agentic_ideas",agentic_ideas)
graph.add_node("department_functions",department_functions)
graph.add_node("idea_aggregator",idea_aggregator)

graph.add_edge(START,"department_functions")
graph.add_edge("department_functions","ML_ideas")
graph.add_edge("department_functions","genai_ideas")
graph.add_edge("department_functions","agentic_ideas")

graph.add_edge("ML_ideas","idea_aggregator")
graph.add_edge("genai_ideas","idea_aggregator")
graph.add_edge("agentic_ideas","idea_aggregator")
graph.add_edge("idea_aggregator",END)


workflow = graph.compile()

user_input = input("Enter the department name: \n")

initial_state = {"department_name":user_input}

workflow.invoke(initial_state)