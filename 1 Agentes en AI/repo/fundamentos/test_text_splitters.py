# pip install -qU langchain-text-splitters

from langchain_text_splitters import RecursiveCharacterTextSplitter

with open('files/example.txt') as f:
    all_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False
)

text = text_splitter.create_documents([all_text])
for chunk in text:
    print(chunk)
# print(text[0])
# print(text[1])