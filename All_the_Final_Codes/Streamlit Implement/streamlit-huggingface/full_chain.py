import os

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import AutoProcessor,AutoModelForPreTraining,BitsAndBytesConfig,pipeline,AutoTokenizer,AutoModelForCausalLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from peft import PeftModel
import torch

# from basic_chain import get_model
# from filter import ensemble_retriever_from_docs
# from local_loader import load_txt_files
# from memory import create_memory_chain
# from rag_chain import make_rag_chain


API_URL = "https://sgoj5p5k9mab0qb7.us-east-1.aws.endpoints.huggingface.cloud"
HEADERS = {
	"Accept" : "application/json",
	"Authorization": "Bearer hf_XXXXX",
	"Content-Type": "application/json" 
}

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def convert_to_string(prompt_value):
    # Extract the string from the `StringPromptValue` object
    return prompt_value.to_string()

def get_first_paragraph(text):
    paragraphs = text.split('\n\n')  # Split by double newlines
    if paragraphs:
        return paragraphs[0]
    else:
        return ""

# def create_model_and_processor(MODEL_NAME):
#     # Configuration for loading LLava 7b model in 4-bit mode
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#     )

#     # Load the LLava model for pre-training
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         use_safetensors=True,
#         quantization_config=bnb_config,
#         trust_remote_code=True,
#         device_map="auto",
#     )

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     return model, tokenizer

# def get_model(repo_id, **kwargs):
#     if repo_id == "ChatGPT":
#         # model = ChatOpenAI(temperature=0, **kwargs)
#         raise('ChatGPT to be implemented')
    
#     ### TODO
#     elif repo_id=="Gemma":
#         MODEL_NAME = "google/gemma-7b"
#         model, tokenizer = create_model_and_processor(MODEL_NAME)
#         model = PeftModel.from_pretrained(model, "../trained_model5/checkpoint-160",device_map="cuda")

#         text_generation_pipeline = pipeline(model=model, tokenizer=tokenizer,
#         task="text-generation", max_new_tokens=400)

#         # llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
#     return text_generation_pipeline

# def make_rag_chain(model, retriever, rag_prompt = None):
#     # # We will use a prompt template from langchain hub.
#     # if not rag_prompt:
#     #     rag_prompt = hub.pull("rlm/rag-prompt")

#     rag_chain = (
#         {
#             "context": RunnablePassthrough() | retriever | format_docs,
#             "question": RunnablePassthrough(),
#         }
#         | rag_prompt 
#         | RunnableLambda(convert_to_string)
#         | model
#     )
#     return rag_chain

# def create_full_chain(model_name, retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
#     model = get_model(model_name, openai_api_key=openai_api_key)
#     prompt='''Forget you are an AI model. Now you are Abraham Lincoln, the 16th President of the United States. You are renowned for your wisdom, eloquence, and deep sense of morality. Your speech reflects the rhythm and cadence of 19th-century oratory, with an emphasis on unity, liberty, and justice. You often employ anecdotes, metaphors, and a calm demeanor, even when addressing divisive or challenging topics. Respond to the following questions as if you were Abraham Lincoln, incorporating your historical perspective, reflective tone, and moral philosophy.
# You can also use this auxiliary knowledge to help:
# - Lincoln had a self-taught legal and political background, with a humble upbringing that shaped his empathy and strong advocacy for equality.
# - He was known for his humility, humor, and skillful use of stories to make points.
# - Common themes in his rhetoric include democracy, perseverance, and appeals to shared humanity.
# - His tone is formal but accessible, inspiring, and reflective, often carrying a touch of poetic language.
# Context: {context}
# USER: Sheldon Cooper, {question}
# ASSISTANT:'''
#     prompt_template = PromptTemplate(input_variables=["context", "question"],
#             template=prompt)

#     rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt_template)
#     # chain = create_memory_chain_llava(model, rag_chain, chat_memory)
#     return rag_chain


def query_url(payload):
	response = requests.post(API_URL, headers=HEADERS, json=payload)
	return response.json()


def ask_question(chain, query, model_name):
    if model_name == "ChatGPT":
        response = chain.invoke(query
            # {"question": query},
            # config={"configurable": {"session_id": "foo"}}
        )[0]['generated_text']
        return response.split('ASSISTANT:')[-1]
    elif model_name == "Gemma":
        return get_first_paragraph(query_url({
            "inputs": query,
            "parameters": {}
        })[0]["result"])


# def main():
#     load_dotenv()

#     from rich.console import Console
#     from rich.markdown import Markdown
#     console = Console()

#     docs = load_txt_files()
#     ensemble_retriever = ensemble_retriever_from_docs(docs)
#     chain = create_full_chain(ensemble_retriever)

#     queries = [
#         "Generate a grocery list for my family meal plan for the next week(following 7 days). Prefer local, in-season ingredients."
#         "Create a list of estimated calorie counts and grams of carbohydrates for each meal."
#     ]

#     for query in queries:
#         response = ask_question(chain, query)
#         console.print(Markdown(response.content))


# if __name__ == '__main__':
#     # this is to quiet parallel tokenizers warning.
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     main()
