import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

# Load the model
llm = ChatOpenAI(model='gpt-4o', temperature=0)

examples = [
    {'input':'2 🦜 2', 'output':'4'},
    {'input':'2 🦜 3', 'output':'5'},
]

example_prompt = ChatPromptTemplate(
    [('human', '{input}'),
     ('ai', '{output}')]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

print(few_shot_prompt.invoke({}).to_messages)

main_prompt = ChatPromptTemplate.from_messages(
    [('system', 'Eres un mago de las matematicas.'),
     few_shot_prompt,
     ('human', '{input}')
     ]
)

chain = main_prompt | llm

print(chain.invoke({'input': 'Cuanto es 2 🦜 9'}).content)