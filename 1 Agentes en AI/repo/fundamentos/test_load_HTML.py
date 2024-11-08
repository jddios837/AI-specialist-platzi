# pip install bs4
# pip install -U lxml
from langchain_community.document_loaders import BSHTMLLoader

file_path = 'files/us-elections-2024.html'

loader = BSHTMLLoader(file_path)
data = loader.load()

print(data)