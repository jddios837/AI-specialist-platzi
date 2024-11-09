# pip install unstructured
# pip install python-magic
# pip install python-magic-bin
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db",
)


# Function to load PDFs, clean blank lines, and split into chunks
def load_clean_and_split_pdfs(folder):
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust as needed
        chunk_overlap=50  # Overlap to keep context between chunks
    )

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder, file)
            loader = PyPDFLoader(pdf_path)
            pdf_documents = loader.load()

            # Clean and split each document
            for document in pdf_documents:
                content = document.page_content
                # Remove blank lines
                cleaned_content = "\n".join(line for line in content.splitlines() if line.strip())

                # Split cleaned content into chunks
                chunks = splitter.split_text(cleaned_content)

                # Create a new document for each chunk
                for chunk in chunks:
                    chunk_doc = document.copy()  # Create a copy to retain metadata
                    chunk_doc.page_content = chunk
                    documents.append(chunk_doc)

    return documents

def add_vector_storage(documents):
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

def main():
    # print("Loading documents...")
    # documents = load_clean_and_split_pdfs("pdfs")
    #
    # print("Vectorizing documents...")
    # add_vector_storage(documents)

    results = vector_store.similarity_search(
        "Tuberías ocultas",
        k=2,
        # filter={"source": "tweet"}
    )

    for result in results:
        print(f"* {result.page_content} [{result.metadata}]")


if __name__ == "__main__":
    main()