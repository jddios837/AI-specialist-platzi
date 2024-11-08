# install pyPdf $pip install --upgrade --quiet pypdf
# install faiss-gpu $pip install faiss-gpu or pip install faiss-cpu
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()
api_key = os.getenv('OPENAI_API_LANGCHAIN_KEY')

############### CSV LOADER ##############################
file_csv_path = "files/customers-100.csv"

loader_csv = CSVLoader(file_path=file_csv_path)
data = loader_csv.load()

for record in data[:2]:
    print(record)

