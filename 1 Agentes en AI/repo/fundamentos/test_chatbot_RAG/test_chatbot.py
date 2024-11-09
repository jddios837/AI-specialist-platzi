from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

loader = DirectoryLoader(
    "pdfs",
    glob='**/*.pdf',
    loader_cls=PyPDFLoader
)

# load documents
pages = loader.load()

# split configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# create chunks
splits = text_splitter.split_documents(pages)

vector_store = Chroma.from_documents(
    documents=splits,
    embeddings=OpenAIEmbeddings()
)

retriever = vector_store.as_retriever()

system_prompt = (
    'Eres un asistente que devuelve información de múltiples PDFS, ademas incluye emojis a cada una de las respuestas. Tienes el siguiente {context}'
)

llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
    max_tokens=150,
)
