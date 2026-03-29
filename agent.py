"""
PostgreSQL Database Agent - Runs with Local Llama 3.
Uses the llama3 model running on Ollama.
"""

import os
import logging
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from tools import sql_tool, tables_tool, schema_tool

# Logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME      = os.getenv("OLLAMA_MODEL", "llama3")

# --- LLM ---
llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0,          # For deterministic output
    num_predict=1024,       # Max tokens
)

# --- Tools ---
tools = [tables_tool, schema_tool, sql_tool]

# --- Prompt ---
# ReAct (Reason + Act) template optimized for Llama 3
REACT_TEMPLATE = """You are a PostgreSQL database expert.
Analyze the user's natural language question, create the appropriate SQL query, and execute it.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important Rules:
- First, use 'get_tables' to see which tables exist.
- Then, use 'get_schema' to check the columns of the relevant table.
- Finally, use 'run_sql' to execute the query.
- Always use 'LIMIT 100' in SQL queries unless specified otherwise.
- Be careful about SQL injection; do not directly insert user values into SQL.
- If you encounter an error, explain it and try an alternative approach.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(REACT_TEMPLATE).partial(
    tools="\n".join([f"- {t.name}: {t.description}" for t in tools]),
    tool_names=", ".join([t.name for t in tools]),
)

# --- Agent ---
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # Step-by-step output
    max_iterations=10,      # Prevents infinite loops
    handle_parsing_errors=True,
    return_intermediate_steps=False,
)


def run_agent(question: str) -> str:
    """Runs the agent with the given question and returns the answer."""
    logger.info(f"Question: {question}")
    try:
        result = agent_executor.invoke({"input": question})
        answer = result.get("output", "No answer received.")
        logger.info(f"Answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return f"An error occurred while running the agent: {e}"


# --- CLI ---
if __name__ == "__main__":
    print("=" * 60)
    print(" PostgreSQL Agent (Local Llama 3)")
    print(" Type 'exit' or 'quit' to close")
    print("=" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down agent...")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "logout"):
            print("Goodbye!")
            break

        answer = run_agent(question)
        print(f"\n💡 Answer:\n{answer}")