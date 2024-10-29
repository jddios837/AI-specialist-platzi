import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')
model = ChatOpenAI(model='gpt-4o')

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {'configurable': {'session_id': 'abc2'}}

response = with_message_history.invoke(
    [HumanMessage(content='Hola, mi nombre es Juan de Dios')],
    config=config
)

print(response.content)
print(store)


# memory = [
#     HumanMessage(content='Hola, mi nombre es Juan de Dios'),
#     AIMessage(content='¡Hola Juan de Dios! ¿En qué puedo ayudarte hoy?'),
#     HumanMessage('¿Cual es mi nombre?')
# ]

# query = input('Preguntame algo: ')
# memory.append(HumanMessage(content='query'))

# print(model.invoke(memory).content)

