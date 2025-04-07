import os
from dotenv import load_dotenv
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class KnowledgeBase:
    def __init__(self, docs_path="knowledge_docs", persist_directory="kb_chroma_db", rebuild=False):
        self.docs_path = docs_path
        self.persist_directory = persist_directory
        #self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.embedding = OllamaEmbeddings(model="nomic-embed-text")

        if not os.path.exists(self.persist_directory) or rebuild:
            self.build_index()
        else:
            self.load_index()

    def build_index(self):
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*",
            loader_cls=UnstructuredFileLoader,
            silent_errors=True,
            show_progress=True
        )
        documents = loader.load()

        print("Loaded documents:")
        for doc in documents:
            print(f"- {doc.metadata.get('source')}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
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
        if not docs:
            return ""

        # Group chunks by reference (filename)
        results_by_ref = {}
        for doc in docs:
            snippet = doc.page_content.strip()
            reference = doc.metadata.get("source", "No reference")
            if reference not in results_by_ref:
                results_by_ref[reference] = set()
            results_by_ref[reference].add(snippet)

        # Build final answer with snippets from each reference
        final_answers = []
        for ref, snippet_set in results_by_ref.items():
            # Combine them into a single text block, removing duplicates
            combined_text = " ".join(snippet_set)
            final_answers.append(f"{combined_text}\n[Reference: {ref}]")

        # Return neatly separated references
        return "\n\n".join(final_answers)

if __name__ == "__main__":
    kb = KnowledgeBase(docs_path="hr_docs", rebuild=True)
