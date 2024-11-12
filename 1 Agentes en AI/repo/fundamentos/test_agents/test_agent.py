import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')
tavily_key = os.getenv('TAVILY_API_KEY')

# Initialize a dictionary to store chat histories for each session
store = {}

# Function to retrieve or create session-specific chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
    max_tokens=100)

search = TavilySearchResults(max_results=2)

answer = search.invoke('¿Que es openai o1-mini')

loader = WebBaseLoader('https://docs.smith.langchain.com/overview')
docs = loader.load()

documents = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(docs)

vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

txt= '''Busca información sobre LangSmith.
Para cualquier pregunta sobre LangSmith
debes utilizar esta herramienta
'''

retriever_tool = create_retriever_tool(
    retriever,
    'langsmith_search',
    txt
)

tools = [search, retriever_tool]
model_with_tools = llm.bind_tools(tools)

# response = model_with_tools.invoke(
#     [HumanMessage('What is the weather in Bogota?')]
# )
#
# print(f'ContentSting: {response.content}')
# print(f'ToolCalls: {response.tool_calls}')

prompt = hub.pull('hwchase17/openai-functions-agent')

agent = create_tool_calling_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools)
#
# query = 'How to install LangSmith?'
# print(agent_executor.invoke({'input': query}))

# Create a conversational retrieval-augmented generation (RAG) chain
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

agent_with_chat_history.invoke(
    {'input': "what's my name?"},
    config={'configurable': {'session_id': 'abc123'}}
)

agent_with_chat_history.invoke(
    {'input': "My name is Bob"},
    config={'configurable': {'session_id': 'abc123'}}
)

answer = agent_with_chat_history.invoke(
    {'input': "what's my name?"},
    config={'configurable': {'session_id': 'abc123'}}
)
print(answer)