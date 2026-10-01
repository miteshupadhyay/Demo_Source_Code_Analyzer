from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import os


# Clone any github repository
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)

# Loading Repository as Documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                           glob="**/*",
                                           suffixes=[".py"],
                                           parser=LanguageParser(language=Language.PYTHON, parser_threshold=500))

    documents = loader.load()

    return documents


# Creating text_chunks
def text_splliter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                          chunk_size = 2000,
                                                                          chunk_overlap=200)

    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks

# Load the embedding Model
def load_embedding():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings
