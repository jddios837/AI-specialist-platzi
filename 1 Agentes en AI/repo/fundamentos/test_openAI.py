import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages


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

# messages =[
#     ('system','Eres un pintor profesional, y ayudas a nuevos aprendices a mejorar sus tenicas'),
#     ('human', 'Quiero aprender a ser un pintor, como puedo comenzar')
# ]
#
# ai_msg = llm.invoke(messages)
# print(ai_msg.content)

# using message type to let know to the model about it
messagesTypes = [
    SystemMessage(content='Eres un asistente util.'),
    HumanMessage(content='me ayudas a organizar las tareas del dia'),
    AIMessage(content='Claro! que tareas necesitas completar hoy?'),
    HumanMessage(content='Necesito, programar la interfaz del nuevo producto, hacer ejercicio, almorzar con mi madre y estudiar para el examen de ciberseguridad'),
    AIMessage(content="Aquí tienes tu lista de tareas. 1. Haz ejercicio a primera hora, 2. Trabaja y programa la nueva interfaz del producto, 3. Almuerza con tu madre, 4. En la noche estudia para el examen de ciberseguridad")
]

trimmed_messages = trim_messages(
    messagesTypes,
    max_tokens=45,
    strategy='last',
    token_counter=ChatOpenAI(model='gpt-4o')
)

response = llm.invoke(messagesTypes)
print(response.content)

# Print the trimmed messages
for msg in trimmed_messages:
    print(f"{msg.__class__.__name__}: {msg.content}")