from functools import cached_property
#from pydantic import BaseSettings, SettingsConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict




class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive  = False,
        extra = "ignore")

    offline_mode: bool = False
    # google gemini
    gemini_api_key:str = None
    gemini_model:str  = 'gemini-2.0-flash'
    temperature:float= 0.3
    max_tokens:int = 1024 # allow use llm any much as want

    # embedding model 

    embedding_model:str = 'BAAI/bge-small-en-v1.5' # embedding model

    # vector stire --  FAISS
    faiss_index_path: str = "./faiss_index" # faiss index path
    top_k :int = 4 # top k results to return from vector store
    retrieval_min_score: float = 0.35 # minimum score for retrieval from vector store
    chunk_size :int = 500 # chunk size for text splitting
    chunk_overlap:int  = 50 # chunk overlap for text splitting

    # memory layer
    memory_dir:str =  "./memory_store" # memory directory path 
    memory_window:int = 10

    # tavily
    tavily_api_key : str | None = None

    # --- Optional observability (Langfuse) ---------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # guardrails
    use_guardrails_ai: bool = True

    @property
    def langfuse_enabled(self):
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property

    def tavily_enabled(self):
        return bool(self.tavily_api_key)

    # built the live objects for rest of your app

class Providers:
    def __init__(self, settings=None):
        self.settings = settings or settings()

    @cached_property
    def llm(self):        
        from langchain_google_genai import ChatGoogleGenerativeAI
        if self.settings.offline_mode:
            # A guard, not a code path anyone should hit: in offline mode the
            # agent never builds an LLM chain. Reaching here means a caller
            # bypassed that, and silently making a billed API call would be worse
            # than failing loudly.
            raise RuntimeError(
                "OFFLINE_MODE is enabled, so no LLM is available. "
                "Set OFFLINE_MODE=false to use Gemini."
            )

        if not self.settings.gemini_api_key:
            raise RuntimeError("[config][providers] Gemini API key is not set. Please set the 'GEMINI_API_KEY' environment variable.")
                  
        return ChatGoogleGenerativeAI(
            model=self.settings.gemini_model,  
            google_api_key=self.settings.gemini_api_key,
            temperature=self.settings.temperature,  
            max_output_tokens=self.settings.max_tokens)

    #@cached_property
    #def embeddings(self):        
    #    from langchain_community.embeddings  import FastEmbeddings
    #    return FastEmbeddings(model_name=self.settings.embedding_model)
    
    @cached_property
    def embeddings(self):
       from langchain_huggingface import HuggingFaceEmbeddings

       return HuggingFaceEmbeddings(model_name=self.settings.embedding_model)


    @cached_property
    def langfuse_handler(self):
        if not self.settings.langfuse_enabled:
            return None
        try:
            from langfuse.callback import CallbackHandler
            return CallHandler(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host
            )
        except Exception as e:
            print(f"[config][providers] Error initializing Langfuse handler: {e}")
            return None            

    def callbacks(self):
            return [self.langfuse_handler] if self.langfuse_handler else []