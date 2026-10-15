from __future__ import annotations
from typing import Annotated,TypedDict
from langchain_core.messages import AIMessage,BaseMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# langraph
from langgraph.graph import END,START,StateGraph
from langgraph.graph.message import add_messages

from config import Providers
from retrieval import Retriever,WebSearchTool
from self_critique import SelfCritic

class AgentState(TypedDict):
    messages :Annotated[list[BaseMessage] , add_messages]
    chat_history:list[BaseMessage]
    context:str
    answer:str
    critique_passed :bool

_SYSTEM = (
 "you are a helpful assistant. The context below is content extracted"
 "from documents that have already been uploaded and processed treated assertthe main"
 "document. Answer the question using only the provided context ifthe answer is present in the context"
 "stated clearly if it is genuinely not"
 "in the context,say i could not find in the provided documents never Say"
 "no document was provided and the context of the ocument is context:{context} ")    


NOTHING_FOUND = (
        "I could not find anything about that in the indexed documents. "
    "Try rephrasing, or ask about the Reagan biography or the T001 clinical "
    "trial report."
)

_OFFLINE_PREAMBLE = (

    "Answering directly from the indexed documents (no language model is "
    "running, so these are the matching passages verbatim):"
)

class RagAgent:
    def __init__(self,
                 
        providers:Providers,
        retriever:Retriever,
        web_tool:WebSearchTool,
        critic:SelfCritic | None = None,
        offline:bool= False,
        ):
        
        self.providers = providers
        self.retriever = retriever
        self.critics = critic
        self.web_tool = web_tool
        self.offline = offline
        if offline:
           self._gen_chain = None
           return 
       
        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            MessagesPlaceholder(variable_name = "chat_history"),
            ("human","{question}"),
        ]
        )
        self._gen_chain = prompt | providers.llm|StrOutputParser()

    # helper functions
    def _latest_human(state:AgentState):
        for msg in reversed (state.get("message",[])):
            if isinstance(msg,HumanMessage):
                 return msg.content
        return ""


    def retrieve_node(self,state):
        question = self._latest_human(state)
        return {'context':self.retrieve.retrieve(question) if question else ""}

    def  tool_node(self,state):
        question = self._latest_human(state)
        if not self.web_tool.available:
            print(f"[agent] no web search tool is configured - skipping the fallback")
            return {"context": ""}
        print("[agent] No local; content - falling back to search")
        web = self.web_tool.search(question)
        combined = (web+"\n\n" + state.get('cpntext','')).strip()
        return {"context":combined}

    @staticmethod
    def _extractive_answer(context: str) -> str:
        """Offline answer: hand back the retrieved passages themselves.

        Without an LLM there is nothing to synthesise prose from, so the honest
        thing is to show the source text and say that is what we are doing,
        rather than dress it up as a generated answer.
        """
        context = context.strip()
        if not context:
            return NOTHING_FOUND
        return f"{_OFFLINE_PREAMBLE}\n\n{context}"


    def generate_node(self,state:AgentState)-> dict:
        if self.offline:
            return {"answer":self._extractive_answer(state.get("context",""))}
        answer = self._gen_chain.invoke(
            {
                "context":state.get("context",""),
                "chat_history":state.get("chat_history",[]),
                "questions":self._latest_human(state),
            },
            config = { "callbacks":self.providers.callbacks()},
        )
        return {"answer":answer}

    def critique_node(self,state):
        final = self.critic.critique(
            self._latest_human(state),
            state.get("answer",""),
            state.get("context","")
            
        )
        return {'answer':final,"critique_passed":True}

    def output_node(self,state):
        return {"message":[AIMessage(content = state["answer"])]}

    def _route_after_retrieve(self,state):
        return "generate_node" if state.get("cpntext","").strip() else "tool_node"
    

    def built(self):
        g = StateGraph(AgentState)
        g.add_node(self.retrieve_node)
        g.add_node(self.tool_node)
        g.add_node(self.generate_node)
        g.add_node(self.critique_node)
        g.add_node(self.output_node)
        g.add_edge(START,"retrieve_node")
        g.add_conditional_edges(
            "retrieve_node",self._rote_after_retrieve,
            {"tool_node":"tool_node","generate_node":self.generate_node}



        )

        g.add_edge("tool_node","generate_node")

        if self.offline:
            g.add_edge("generate_node","output_node")
        else:
            g.add_node("critique_node", self.critique_node)
            g.add_edge("generate_node", "critique_node")
            g.add_edge("critique_node", "output_node")

        g.add_edge("output_node",END)
        return g.compile
