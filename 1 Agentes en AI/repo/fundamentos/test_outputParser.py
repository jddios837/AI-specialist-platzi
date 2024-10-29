import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

# Load the model
model = ChatOpenAI()

# Define the query to be sent to the model
joke_query = 'Tell me a joke'

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Joke)

# Create a prompt template with format instructions and the user query
prompt = PromptTemplate(
    template = 'Answer the user query. \n {format_instructions}\n{query}',
    input_variables = ['query'],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

print(parser.get_format_instructions())
# print(prompt)
# Chain the prompt, model, and parser together
chain = prompt | model | parser

# Invoke the chain with the joke query and print the response
# response = chain.invoke({'query': joke_query})
# print(response)

# Stream the response from the chain invocation with the joke query.
# Each chunk of the response is printed as it is received.
for s in chain.stream({"query": joke_query}):
    print(s)
