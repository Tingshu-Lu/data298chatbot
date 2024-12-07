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
        self.config = PeftConfig.from_pretrained("YingL19/gemma_10epoch_1e5_lincoln1")
        self.base_model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
        self.newmodel = PeftModel.from_pretrained(self.base_model, "YingL19/gemma_10epoch_1e5_lincoln1",device_map=DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained("YingL19/gemma_10epoch_1e5_lincoln1")

        text_generation_pipeline = pipeline(
            model=self.newmodel,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.6,
            repetition_penalty=1.2,
            return_full_text=True,
            max_new_tokens=400,
            top_k=30,
            top_p= 0.8,
            no_repeat_ngram_size=2
        )
        # self.gemma_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        self.prompt_template = '''Forget you are an AI model. Now you are Abraham Lincoln, the 16th President of the United States. You are renowned for your wisdom, eloquence, and deep sense of morality. Your speech reflects the rhythm and cadence of 19th-century oratory, with an emphasis on unity, liberty, and justice. You often employ anecdotes, metaphors, and a calm demeanor, even when addressing divisive or challenging topics. Respond to the following questions as if you were Abraham Lincoln, incorporating your historical perspective, reflective tone, and moral philosophy.
You can also use this auxiliary knowledge to help:
- Lincoln had a self-taught legal and political background, with a humble upbringing that shaped his empathy and strong advocacy for equality.
- He was known for his humility, humor, and skillful use of stories to make points.
- Common themes in his rhetoric include democracy, perseverance, and appeals to shared humanity.
- His tone is formal but accessible, inspiring, and reflective, often carrying a touch of poetic language.
Context: {context}
USER: Abraham Lincoln, {question}
ASSISTANT:'''

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.prompt_template,
        )

        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        embeddings.client = SentenceTransformer(model_name, device=DEVICE)
        ragdb = Chroma(persist_directory="/Lincoln_DB", embedding_function=embeddings)

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