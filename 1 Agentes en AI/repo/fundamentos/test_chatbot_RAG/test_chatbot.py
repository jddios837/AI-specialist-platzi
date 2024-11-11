# Import necessary classes and functions from the langchain library
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)

# Initialize a dictionary to store chat histories for each session
store = {}

# Function to retrieve or create session-specific chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Initialize a loader to load all PDF documents from the "pdfs" directory
loader = DirectoryLoader(
    "pdfs",
    glob='**/*.pdf',
    loader_cls=PyPDFLoader
)

# Load all documents from the specified directory
pages = loader.load()

# Set up configuration for text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Define maximum chunk size of text
    chunk_overlap=50  # Set overlap between chunks
)

# Split the loaded documents into smaller chunks
splits = text_splitter.split_documents(pages)

# Create a vector store from document chunks using embeddings
vector_store = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

# Initialize a retriever to search through the vector store
retriever = vector_store.as_retriever()

# Define the system prompt that will guide the assistant's responses
system_prompt = (
    'Eres un asistente que devuelve información de múltiples PDFS, ' +
    'ademas incluye emojis a cada una de las respuestas. Tienes el siguiente {context}'
)

# Initialize a language model (LLM) with OpenAI's API
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,      # Set response variability
    max_tokens=150      # Set token limit for each response
)

# Define a system prompt for context-based, history-aware responses
contextualize_q_system_prompt = (
    "Responde segun el historial de chat y la ultima pregunta del usuario " +
    "si no esta en el historial de chat o en el contexto. NO respondas la pregunta" +
    "Ademas responde de manera profesional a la pregunta del usuario"
)

# Configure the prompt template with chat history and input placeholders
contextualize_q_system = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ]
)

# Create a retriever that can use chat history for context
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_system
)

# Set up a prompt template for the question-answering chain
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ]
)

# Create a document-combining chain that uses the LLM and prompt for responses
question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

# Create a retrieval chain that combines history-aware retriever with the Q&A chain
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

# Create a conversational retrieval-augmented generation (RAG) chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)

# Test the conversational chain with a sample question
test = conversational_rag_chain.invoke(
    {
        'input': '¿Que es la alimentación?'   # Sample question for testing
    },
    config={
        'configurable': {'session_id': 'abc134'}   # Specify session ID
    }
)['answer']

# Print the result of the test question
print(test)
