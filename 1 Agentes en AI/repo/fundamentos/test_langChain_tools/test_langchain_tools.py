import os
import sqlite3

from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langgraph.prebuilt import create_react_agent

load_dotenv()
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
    max_tokens=100)


def get_engine_for_chinook_db():
    """Load SQL file from local directory, populate in-memory database, and create engine."""
    with open("sql_scripts/Chinook_Sqlite.sql", "r", encoding="utf-8") as file:
        sql_script = file.read()

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

# Create the engine and initialize the SQL database
engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# tools = toolkit.get_tools()
# for tool in tools:
#     print(tool)

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

assert len(prompt_template.messages) == 1
print(prompt_template.input_variables)

system_message = prompt_template.format(dialect="SQLite", top_k=5)

agent_executor = create_react_agent(
    llm, toolkit.get_tools(), state_modifier=system_message
)

example_query = "Which country's customers spent the most?"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()