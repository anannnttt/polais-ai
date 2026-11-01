from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import fitz

doc = fitz.open("NLTK.pdf")

page = doc[11]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500 , chunk_overlap = 50
)
chunks = text_splitter.create_documents([page.get_text()])
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
vector_store= Chroma.from_documents(
    documents = chunks,
    embedding= embeddings,
    persist_directory = "./chroma_db" 

)
print("embedding lodaded")




# # print(len(chunks))
# # print("next")
# print(chunks[2])
# for i, chunk in enumerate(chunks):
#     print(f"--- CHUNK {i} ---")
#     print(chunk)
#     print("\n" + "="*40 + "\n")