# pip install langchain-pinecone
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

from fundamentos.test_text_splitters import text_splitter

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')
pine_key = os.getenv('PINECONE_API_KEY')

llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
    max_tokens=100)

index_name = 'documents'
loader = TextLoader('files/example.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

docs = text_splitter.split_documents(documents)

# LOAD IN PINECONE CLOUD
embedding = OpenAIEmbeddings(model='text-embedding-3-small')

# vector_store_from_docs = PineconeVectorStore.from_documents(
#     docs,
#     index_name= index_name,
#     embedding=embedding)

vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding)

query = 'Execution Phase'
print(vector_store.similarity_search(query))


