import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

# Load the model
llm = ChatOpenAI(model='gpt-4o', temperature=0)

# PromptTemplate example
prompt_template = PromptTemplate.from_template('Dime un chiste de {topic}')
print(prompt_template.invoke({'topic':'gatos'}))

# ChatPromptTemplate example, using SystemMessage, HumanMessage as objects
# cause that the replacement doesn't work
# HumanMessage(content='Dime un chiste de {topic}') does not replace topic later
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'Eres un asistente util'),
    ('user', 'Dime un chiste de {topic}')
])
print(chat_template.invoke({'topic':'gatos'}))

prompt_placeholder = ChatPromptTemplate.from_messages([
    ('system', 'Eres un asistente útil'),
    MessagesPlaceholder('msgs')
])

print(prompt_placeholder.invoke({'msgs': [
    HumanMessage('Hola'),
    HumanMessage('Adios')
]}))