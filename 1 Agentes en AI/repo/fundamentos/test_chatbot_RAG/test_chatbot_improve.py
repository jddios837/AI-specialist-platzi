# Import necessary modules
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory


# Step 1: Define a singleton DocumentLoader class to load documents only once.
class DocumentLoader:
    _instance = None
    _documents = None

    def __new__(cls, path="pdfs", glob_pattern='**/*.pdf', loader_cls=PyPDFLoader):
        if cls._instance is None:
            cls._instance = super(DocumentLoader, cls).__new__(cls)
            cls._documents = cls._load_documents(path, glob_pattern, loader_cls)
        return cls._instance

    @classmethod
    def _load_documents(cls, path, glob_pattern, loader_cls):
        loader = DirectoryLoader(path, glob=glob_pattern, loader_cls=loader_cls)
        return loader.load()

    def get_documents(self):
        return self._documents


# Step 2: Create a TextSplitter class responsible for splitting the documents.
class TextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, documents):
        return self.text_splitter.split_documents(documents)


# Step 3: Define a VectorStoreHandler to manage vector storage and retrieval.
class VectorStoreHandler:
    def __init__(self, documents):
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings()
        )

    def get_retriever(self):
        return self.vector_store.as_retriever()


# Step 4: Define a ChatSessionManager class to manage chat sessions.
class ChatSessionManager:
    def __init__(self):
        self.store = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]


# Step 5: Define the main ConversationalRAGChain class with injected dependencies.
class ConversationalRAGChain:
    def __init__(self, retriever, session_manager, llm_model="gpt-4o"):
        self.retriever = retriever
        self.session_manager = session_manager
        self.llm = ChatOpenAI(model=llm_model, temperature=0, max_tokens=150)

        # Define the prompt templates
        self.system_prompt = ChatPromptTemplate.from_messages([
            ('system',
             'Eres un asistente que devuelve información de múltiples PDFS, además incluye emojis en cada respuesta. Tienes el siguiente {context}'),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])

        self.history_aware_prompt = ChatPromptTemplate.from_messages([
            ('system', "Responde según el historial de chat y la última pregunta del usuario."),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])

        # Set up the history-aware retriever and question-answering chain
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.history_aware_prompt
        )
        self.qa_chain = create_stuff_documents_chain(self.llm, self.system_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)

        # Finalize with a runnable RAG chain that includes message history
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.session_manager.get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )

    def ask_question(self, question, session_id):
        result = self.conversational_rag_chain.invoke(
            {'input': question},
            config={'configurable': {'session_id': session_id}}
        )
        return result['answer']


# Usage example
if __name__ == "__main__":
    # Load documents and split them only once
    documents = DocumentLoader().get_documents()
    splits = TextSplitter().split(documents)

    # Initialize the VectorStoreHandler with the document splits
    vector_store_handler = VectorStoreHandler(splits)

    # Initialize ChatSessionManager and the main ConversationalRAGChain
    session_manager = ChatSessionManager()
    conversational_rag = ConversationalRAGChain(
        retriever=vector_store_handler.get_retriever(),
        session_manager=session_manager
    )

    # Run a test question through the chain
    answer = conversational_rag.ask_question("¿Qué es la alimentación?", session_id="abc134")
    print(answer)
