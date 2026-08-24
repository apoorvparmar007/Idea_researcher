from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from typing import TypedDict
from langchain_community.tools import DuckDuckGoSearchRun
import json
# from ddgs import DDGS
from langchain.agents import create_agent
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ddgs.exceptions import DDGSException
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


ddg = DuckDuckGoSearchRun()


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(DDGSException),
    reraise=True,
)
def _ddg_search_with_retry(query: str) -> str:
    return ddg.invoke(query)


@tool
def ddg_search(query: str) -> str:
    """Search DuckDuckGo for the given query and return the results."""
    try:
        return _ddg_search_with_retry(query)
    except DDGSException as e:
        return f"Search failed after retries: {e}"


tools = [ddg_search]

llm = ChatOllama(model='ornith-1.5:9b', num_ctx=16384)
# llm_with_tools = llm.bind_tools(tools)

evaluator_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")




class IdeaState(TypedDict):
    department_name: str
    company_name: str
    generated_ideas: str
    functions: str
    final_ideas: str
    evaluated_ideas: str



def department_functions(state: IdeaState):
    print("\nUnderstanding Department Functions\n")
    department_name = state['department_name']

    prompt = f"""List all the funtions,Sub-Functions,Tools & Systems Used within this 
    department: {department_name}"""

    response = llm.invoke(prompt)
    print("\nIdentified below Department Functions\n: ",response.content)

    return ({"functions":response.content})

def idea_generator(state: IdeaState):

    print("\nGenerating Ideas\n")
    department_name = state['department_name']
    functions = state['functions']

    # search_results = ddg.invoke(f"Agentic AI use cases and solutions in {department_name} department")

    prompt = f"""Basis the below department functions, Fetch all the agentic ai, gen ai and ML ideas or 
    solutions that are being implemented in the
    organizations across the world in the {department_name} department for each of these functions.\n
    Department Functions:\n {functions}"""

    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. Use the ddg_search tool sparingly: "
            "run at most 2 rounds of searches (a handful of queries total), "
            "then stop searching and write your final synthesized answer."
        ),
        debug=True
    )

    # 3. Invoke the agent loop
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        config={"recursion_limit": 10},
    )

    print("Result-0\n",result)

    last_message = result["messages"][-1]
    print("Last message type:", type(last_message).__name__)
    print("Last message content:", repr(last_message.content))
    print("Last message additional_kwargs:", last_message.additional_kwargs)
    print("Last message response_metadata:", getattr(last_message, "response_metadata", None))

    print("Result\n",last_message.content)

    return ({"generated_ideas":last_message.content})

def idea_evaluator(state: IdeaState):
    print("\nEvaluate the Ideas\n")

    department_name = state['department_name']
    generated_ideas = state['generated_ideas']

    prompt = f"""Below are all the generative ai, agentic ai and ML ideas/solutions that are being implemented in the 
    organizations in the {department_name} department. Evaluator all the ideas on feasibility and impact on score
     of 10 each. 
     Ideas:\n {generated_ideas}"""

    agent = create_agent(
            model=evaluator_llm,
            tools=tools,
            system_prompt=(
                "You are a helpful assistant. Use the ddg_search tool sparingly: "
                "run at most 2 rounds of searches (a handful of queries total), "
                "then stop searching and write your final synthesized answer."
            )
        )
    
        # 3. Invoke the agent loop
    result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            config={"recursion_limit": 10},
        )

    print("\evaluated_ideas\n",result["messages"][-1].content)

    return {"evaluated_ideas":result["messages"][-1].content}

graph = StateGraph(IdeaState)


graph.add_node("idea_generator",idea_generator)
graph.add_node("department_functions",department_functions)
graph.add_node("idea_evaluator",idea_evaluator)

graph.add_edge(START,"department_functions")
graph.add_edge("department_functions","idea_generator")
graph.add_edge("idea_generator","idea_evaluator")
# graph.add_edge("idea_evaluator","idea_evaluator")

# graph.add_edge("ML_ideas","idea_aggregator")
# graph.add_edge("genai_ideas","idea_aggregator")
# graph.add_edge("agentic_ideas","idea_aggregator")
graph.add_edge("idea_evaluator",END)


workflow = graph.compile()

user_input = input("Enter the department name: \n")

initial_state = {"department_name":user_input}

final_state = workflow.invoke(initial_state)

with open("final_state.txt", "w", encoding="utf-8") as f:
    # Use json.dumps for a neat, readable string format of the state dict
    f.write(json.dumps(final_state, indent=4, default=str))