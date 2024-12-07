from typing import Dict, List, Any
import torch
from transformers import pipeline
from peft import PeftModel, PeftConfig
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def convert_to_string(prompt_value):
    # Extract the string from the `StringPromptValue` object
    return prompt_value.to_string()

def outputParser(lst):
    return (lst[0]['generated_text']).split('ASSISTANT:')[-1]

class EndpointHandler():
    
    def __init__(self, path=""):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "moka-ai/m3e-base"
        self.config = PeftConfig.from_pretrained("lx71437/SH3_15")
        self.base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.newmodel = PeftModel.from_pretrained(self.base_model, "lx71437/SH3_15",device_map=DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained("lx71437/SH3_15")

        text_generation_pipeline = pipeline(
            model=self.newmodel,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.6,
            repetition_penalty=1.2,
            return_full_text=True,
            max_new_tokens=120,
            top_k=30,
            top_p= 0.8,
            no_repeat_ngram_size=2
        )

        self.prompt_template = '''Forget you are an AI model. You are an expert at impersonating influential detectives. Specifically, you are a world-leading expert at answering questions in the manner of Sherlock Holmes.
Respond in a style that embodies Sherlock Holmes' analytical, observant, and keenly intelligent personality, maintaining a sharp focus on logic, detail, and reasoning. Each answer should reflect Holmes' methodical approach, often involving keen observation, deductive reasoning, and a slightly aloof tone. Answer as though presenting deductions to a fellow investigator, emphasizing how each detail supports a logical conclusion.
Use these core characteristics to guide your responses:
- **Keen observation and deduction**: "It is my business to know what others do not." Provide responses that reveal subtle insights and deduced conclusions.
- **Logical analysis over emotion**: "Once you eliminate the impossible, whatever remains, however improbable, must be the truth." Ensure each answer rests on rational analysis.
- **Resilience in pursuit of truth**: "The case is never over until the last clue is uncovered." Emphasize Holmes' relentless commitment to uncovering every detail necessary for the truth.
Remember:
- Showcase Holmes' dedication to logical analysis, with clear explanations for his conclusions.
- Avoid listing qualities without insight; instead, reveal how each trait or detail contributes to a deeper understanding of the situation.
- In line with Holmes' character, maintain a subtle wit and an intellectual, slightly aloof tone throughout.
Context: {context}
USER: Sherlock Holmes, {question}
ASSISTANT:'''


        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.prompt_template,
        )

        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        embeddings.client = SentenceTransformer(model_name, device=DEVICE)
        ragdb = Chroma(persist_directory="/SH_db", embedding_function=embeddings)

        retriever = ragdb.as_retriever(search_kwargs={'k': 3})
        # self.llm_chain = LLMChain(llm=self.gemma_llm, prompt=self.prompt)
        self.rag_chain = (
 {      "context": RunnablePassthrough() | retriever | format_docs,
        "question": RunnablePassthrough(),
        }
        | self.prompt 
        | RunnableLambda(convert_to_string)
        | text_generation_pipeline
        | outputParser
)
 
    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: `str`)
            date (:obj: `str`)
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        # get inputs
        message = data.pop("inputs",data)
        res = self.rag_chain.invoke(message)
        return [{"raw_result": res, "result": res}]