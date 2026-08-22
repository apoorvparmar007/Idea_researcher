# Idea Researcher

A LangGraph-powered agent that researches AI/automation opportunities for a given company department. Given a department name, it maps out the department's functions, then generates and aggregates ML, generative AI, and agentic AI use cases relevant to that department.

## How it works

The workflow is a LangGraph `StateGraph` with the following nodes:

1. **department_functions** — asks the LLM to list the functions, sub-functions, tools, and systems used within the given department.
2. **ML_ideas**, **genai_ideas**, **agentic_ideas** — run in parallel off the department functions, each asking the LLM (with web search tooling available) for real-world ML, generative AI, and agentic AI solutions used in that department across organizations.
3. **idea_aggregator** — merges the three idea sets into a single, comprehensive list.

```
START -> department_functions -> ML_ideas -----+
                                -> genai_ideas ---+--> idea_aggregator -> END
                                -> agentic_ideas --+
```

The LLM is served locally via [Ollama](https://ollama.com/) (`qwen3:8b`) through `langchain-ollama`, and is bound to a DuckDuckGo search tool (`langchain-community`) so it can look up current information.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) running locally with the `qwen3:8b` model pulled:
  ```bash
  ollama pull qwen3:8b
  ```
- Python packages:
  ```bash
  pip install langgraph langchain-ollama langchain-community duckduckgo-search pydantic
  ```

## Usage

```bash
python main.py
```

You'll be prompted to enter a department name, e.g.:

```
Enter the department name:
Human Resources
```

The script prints progress for each step and writes the final aggregated state (including all intermediate results) to `final_state.txt` as formatted JSON.

## Output

`final_state.txt` contains the full final state of the graph, including:

- `department_name` — the input department
- `functions` — identified department functions/sub-functions/tools
- `ml_ideas` — ML use cases for the department
- `genai_ideas` — generative AI use cases for the department
- `agentic_ideas` — agentic AI use cases for the department
- `final_ideas` — the aggregated, comprehensive idea list
