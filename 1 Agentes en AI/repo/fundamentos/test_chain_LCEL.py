import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

# Load the model
llm = ChatOpenAI(model='gpt-4o', temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ('system', "You are a translation assistant."),
    ('human', "Translate the following text to {language}: {text}")
])

parser = StrOutputParser()

# this is a chain
chain = prompt_template | llm | parser

response = chain.invoke({'language': 'Italian', 'text': 'Mi nombre es Juan'})

print(response)
