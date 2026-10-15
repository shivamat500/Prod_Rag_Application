# load chunkc and query vector and pass it to llm
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class Retriever:
    def __init__(self, embeddings,index_path = "./faiss_index",top_k=3):
        self.embedding = embeddings
        self.index_path = index_path
        self.top_k = top_k
        self._retriever = self._load(embeddings)
        self.vector_store = self.load_vector_stores()


    def load(self,embeddings):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. Please ensure the index is built and available.")

        store = FAISS.load_local(
            self.index_path, 
            allow_dangerous_deserialization=True,
            embeddings=embeddings)
        return store.as_retriever(search_kwargs={"k": self.top_k})

    def retrieve_docs(self,query):
        if not query:
            return []
        return self._retriever.invoke(query)

    def retrieve(self,query):
        docs = self.retrieve_docs(query)
        return "\n\n".join([doc.page_content for doc in docs])


class WebSearchTools:

    def __init__(self,embedding= False,max_results =3):
        self._tool  =   None
        if enabled:
            try: 
                from langchain_community.tools import TavilySearchResults
                self.tool= TavilySearchResults(max_result = max_results)
            except Exception as e:
                print("f[retrieval] tavily unavailable")

    @property
    def available(self) :
        return self._tool is not None


    def search (self,query):
        if not self.available or not query:
            return ""
        try:
            results = self._tool.invoke(query)
        except Exception as e:
            print(f"retrieval web search has failed !!!")
        return "\n\n".join(
            f"[web] {r.get('title','')}:{r.get('content','')}" for r in results)
        )                