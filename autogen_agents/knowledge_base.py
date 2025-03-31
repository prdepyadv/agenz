import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class KnowledgeBase:
    def __init__(self, docs_path="knowledge_docs", persist_directory="kb_chroma_db"):
        self.docs_path = docs_path
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

        if not os.path.exists(self.persist_directory):
            self.build_index()
        else:
            self.load_index()

    def build_index(self):
        loader = DirectoryLoader(self.docs_path, loader_cls=TextLoader, silent_errors=True)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        self.db = Chroma.from_documents(
            texts, 
            self.embedding, 
            persist_directory=self.persist_directory
        )
        print("Knowledge base indexed successfully.")

    def load_index(self):
        self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
        print("Knowledge base loaded successfully.")

    def query(self, question, k=3):
        docs = self.db.similarity_search(question, k=k)
        return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    kb = KnowledgeBase()