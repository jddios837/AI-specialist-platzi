import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

# Load the model
llm = ChatOpenAI(model='gpt-4o', temperature=0)

# Define a sequence of operations using RunnableLambda.
# The first operation increments the input by 1.
# The result is then passed to the next operation which multiplies the result by 2.
sequence1 = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)

# Define a sequence of operations using RunnableLambda.
# The first operation increments the input by 1.
# The result is then passed to a dictionary of operations:
# - 'index_1': Multiplies the result by 2.
# - 'index_2': Multiplies the result by 5.
sequence = RunnableLambda(lambda x: x + 1) | {
    'index_1': RunnableLambda(lambda x: x * 2),
    'index_2': RunnableLambda(lambda x: x * 5)
}

print(sequence.invoke(10))

