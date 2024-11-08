# install pyPdf $pip install --upgrade --quiet pypdf
# install faiss-gpu $pip install faiss-gpu or pip install faiss-cpu
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

file_path = (
    "files/lorem-ipsum.pdf"
)
############### PDF LOADER ##############################
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

print(pages[0])

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("Lorem ipsum", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])


