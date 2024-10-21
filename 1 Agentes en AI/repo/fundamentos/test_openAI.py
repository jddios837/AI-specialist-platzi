import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

# Use the API key in your application
# Use the API key in your application
if api_key:
    print("open_api_key=" + api_key)  # or use it in your API calls
else:
    print("API key not found.")

llm = ChatOpenAI(model='gpt-4o', temperature=0)

messages =[
    ('system','Eres un pintor profesional, y ayudas a nuevos aprendices a mejorar sus tenicas'),
    ('human', 'Quiero aprender a ser un pintor, como puedo comenzar')
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)