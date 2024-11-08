import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# RAG Retrieval-augmented generation

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

llm = ChatOpenAI(model='gpt-4o', temperature=0)

embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(
    [
        'Hola!',
        'Hola, como estas?',
        'Cual es tu nombre?'
        'Me llamo Daniel',
        'Hola Daniel'
    ]
)
embeddings_query = embeddings_model.embed_query('Cual es el nombre mencionado en la conversación?')

print(len(embeddings))
# print(embeddings)


# ¿Por qué utilizar Embeddings?
# El uso de embeddings es crucial cuando necesitamos personalizar la información que un modelo de lenguaje maneja. Los embeddings permiten convertir un documento o fragmento de texto en una representación numérica, facilitando la recuperación de información específica y relevante. Esto es útil cuando trabajamos con datos que no están en los modelos preentrenados, como:
#
# Información confidencial de la empresa (datos internos).
# Información actualizada o específica que debe consultarse en tiempo real.
# Datos almacenados en sistemas de gestión empresarial como ERP o CRM.
# Ejemplos de Aplicación
# Cargar un documento y convertirlo a vectores: Imagina que tienes una serie de documentos internos de tu empresa (por ejemplo, PDF, CSV, HTML) que quieres que el modelo de lenguaje entienda y utilice para responder preguntas. Primero, esos documentos deben ser vectorizados.
# Realizar consultas semánticas: Una vez que tienes los textos convertidos a embeddings, puedes realizar consultas a esos vectores. Por ejemplo, si un usuario pregunta “¿Cuál es el nombre mencionado en la conversación?”, el sistema buscará en el espacio vectorial los textos que tengan una representación semánticamente cercana a la pregunta.