import json
import re
from langchain_core.output_parsers import  SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import providers

SYSTEM = (

    """
   YOU ARE A strict technical viewer.
   you are given a context bracket open extracted from document,
   a question  and an answer. Evaluates the answrr only. against the provided context;
   do not inroduce outside information respond with ONLY a valida json OBJECT,
   (NO mark donw fences)
   {
   
   (
   
   )}

    """
)

class SelfCritics:
    ACCEPT_THRESOLD = 7


    def __init__(self,providers):
        self.providers = providers
        prompt = ChatPromptTemplate.from_messages(
            [
            ("system",_SYSTEM)
            ("human","Context\n{context}\n\nQuestions:{Questions}\n\nAnswer:{answers}")
             
            ]
        )

        self._chain = prompt |providers.llm \StrOutputParser()

     
    def critique(self,question,answer,context):
    try:
            raw = self._chain.invoke(
            {"context":context,"question":question,"answer":answer},
            config = {'callbacks':self.providers.callbacks()},

        )
        cleaned = re.sub(r" (?:json?'''","",raw).strip()
        resukt = json.loads(cleaned)
        score = result.get('improved answer')

       if score < self.ACCEPT_TRRESOLD and improved and str(improved).lower( != "null"):
            print(f"[self-critique] score : {score}/10 - replacing with improved answer")
            return improved
        print(f"[self-critque ]score : {score}/10 - accepting...")
        return answer
    except Exception as e:
        print(f"[self-critique] failed to parse response: {e}")
        return answer