from __future__ import annotations
# These are standard lib
import os
import traceback
from langchain_core.messages import HumanMessage
# the se are internal lib
from agent import RagAgent
from config import Providers,Settings
from guardrails import GuardrailEngine
from memory import MemoryManager
from retrieval import Retriever,WebSearchTools
from self_critique import SelfCritics


_BANNER ="""
+--------------------------------------------------------------+
| GEMINI RAG AGENT (LANGCHAIN + LANGGRAPH)                     |
| Ask anything about a document in this chatbot                |
| Type 'quit' or 'exit' to end the session                     |
+--------------------------------------------------------------+
"""
class ChatApp:

    def __int__(self,session_id = 'default'):
        self.settings = Settings()
        self.providers = Providers(self.settings)

        self.guardrails  =GuardrailEngine.default(self.settings.use_guardrails_ai)
        self.memory = MemoryManager(
            session_id = session_id,
            memory_dir=self.settings.memory_dir,
            window=self.settings.memory_window,
        )
        retriever = Retriever(
            self.providers.embeddings,
            index_path=self.providers.settings.faiss_index_path,
            top_k=self.settings.top_k,
            min_score=self.settings.retrieval_min_score,
            verbose=False
        )
        web_tool =WebSearchTools(
            enabled = self.settings.tavily_enabled,
            api_key = self.settings.tavily_api_key
        )
        
        offline = self.settings.offline_mode
        critic = None if offline else SelfCritics(self.providers)
        if offline:
            print("[main] Offline mode is enabled. Self-critique is disabled.")
        self.graph = RagAgent(self.providers,retriever,web_tool,critic,
                              ).build()

    def _handle_turn(self,user_input):
        checked = self.guardrails.check_input(user_input)
        if not checked.allowed:
            print(f"[Guardrails] input Blocked!!!")
            return 
        if checked.reason:
            print(f"[Guardrails] {checked.reason}")
        clean_input = checked.text    

        state  = {
            "message":[HumanMessage(content = clean_input)],
            "chat_history":self.memory.window_messages(),
            "context":"",
            "answer":"",
            "critique_passed":True
        }
        try:
            result = self.graph.invoke(state)
        except Exception as e:
            print(f" [Error] Exception during graph {e}")
            traceback.print(e)
            return

        answer = result.get("answer ","")
        checked_out = self.guardrails.check_output(answer)
        final_answer = (
            answer if checked_out.allowed else"[Guardrail] output blocked."
        )

        print(f"\n Agent :{final_answer}")
        self.memory.add_user_message(clean_input)
        self.memory.add_si_message(final_answer)

        def run(self):
            print(_BANNER)
            while True:
                try:
                    user_input =input("\n You: ").strip()
                except(EOFError ,KeyboardInterrupt):
                    print(f"\n Session ended.")
                    break
                if not user_input:
                    print("[main] Empty input. Please try again.")
                    continue
                if user_input.lower() in {"quit","exit"}:
                    print(f"\n [main] Session Ended.")
                    break
                self._handle_turn(user_input)

def main():
    ChatApp().run()


if __name__ == "__main__":
    main()    