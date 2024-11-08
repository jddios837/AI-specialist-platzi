import os
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sympy import content

load_dotenv()
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')
llm = ChatOpenAI(model='gpt-4o', temperature=0)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant. Answer as best a you can.'),
        MessagesPlaceholder(variable_name='messages')
    ]
)

chain = prompt | llm
response = chain.invoke({
    'messages': [HumanMessage(content='Hi, I am Juan')]
})
print(response.content)

with_message_history = RunnableWithMessageHistory(chain, get_session_history)
config = {'configurable': {'session_id': 'abc2'}}

response = with_message_history.invoke(
    [HumanMessage(content='What is your name?')],
    config
)