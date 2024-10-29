import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

# Load the model
llm = ChatOpenAI(model='gpt-4o', temperature=0)

chat_history = []

if not chat_history:
    system_message = SystemMessage(content='Eres un asistente util')
    chat_history.append(system_message)

query = input('Haz una pregunta: ')
chat_history.append(HumanMessage(content='query'))

response = llm.invoke(chat_history).content
chat_history.append(AIMessage(content=response))

print(response)

for message in chat_history:
    print(message)
