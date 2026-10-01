from src.helper import repo_ingestion,load_embedding, load_repo, text_splliter
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY

github_repo_url ="https://github.com/miteshupadhyay/Demo_Neural-Semantic-Matching-Protocol"

repo_ingestion(github_repo_url)

documents  = load_repo('/repo')
text_chunks = text_splliter(documents)
embeddings = load_embedding()

# Store vector in ChromaDB
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings,persist_directory='./db')
vectordb.persist()