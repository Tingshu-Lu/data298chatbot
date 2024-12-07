import os

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import AutoProcessor,AutoModelForPreTraining,BitsAndBytesConfig,pipeline,AutoTokenizer
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from peft import PeftModel
import torch

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.history_aware_retriever import create_history_aware_retriever

# from basic_chain import get_model
# from filter import ensemble_retriever_from_docs
# from local_loader import load_txt_files
# from memory import create_memory_chain
# from rag_chain import make_rag_chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_question(input):
    if not input:
        return None
    elif isinstance(input,str):
        return input
    elif isinstance(input,dict) and 'input' in input:
        return input['input']
    # elif isinstance(input,BaseMessage):
    #     return input.content
    else:
        raise Exception("string or dict with 'input' key expected as RAG chain input.")
    
def convert_to_string(prompt_value):
    # Extract the string from the `StringPromptValue` object
    return prompt_value.to_string()

def outputPasrser(output):
    return output.split('ASSISTANT:')[-1]

def extract_response(output):
    return {"output": output.split('ASSISTANT:')[-1]}

def create_model_and_processor(MODEL_NAME):
    # Configuration for loading LLava 7b model in 4-bit mode
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the LLava model for pre-training
    model = AutoModelForPreTraining.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,  # Ensure use of safe tensor format
        quantization_config=bnb_config,  # Use the 4-bit quantization configuration
        trust_remote_code=True,  # Trust any custom code from the model repo
        device_map="cuda",  # Automatically map the model to available devices
    )

    # Load the tokenizer for LLava 7b
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,device_map="cuda")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    # tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token
    # tokenizer.padding_side = "right"  # Ensure padding is applied on the right side

    return model, processor, tokenizer

def get_model(repo_id, **kwargs):
    if repo_id == "ChatGPT":
        # model = ChatOpenAI(temperature=0, **kwargs)
        raise('ChatGPT to be implemented')
    elif repo_id=="Llava":
        MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
        model, processor, tokenizer = create_model_and_processor(MODEL_NAME)
        model = PeftModel.from_pretrained(model, "../trained_model5/checkpoint-160",device_map="cuda")

        text_generation_pipeline = pipeline(model=model, tokenizer=tokenizer,
        task="text-generation", max_new_tokens=400)

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm,processor

def make_rag_chain(model, retriever, rag_prompt):

    rag_chain = (
        {"context": retriever | format_docs,
         "input": RunnablePassthrough()            
        }
        | rag_prompt 
        | RunnableLambda(convert_to_string)
        | model
        | extract_response
    )

    # def get_session_history(session_id: str) -> BaseChatMessageHistory:
    #     return chat_memory

    # with_message_history = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    # )
    return rag_chain

def create_memory_chain(llm, base_chain, chat_memory):

    prompt='''Given the following chat history and the latest USER question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is.\
chat history:
{chat_history}
USER: {input}
ASSISTANT:'''
    contextualize_q_prompt = PromptTemplate(input_variables=["chat_history", "input"],
            template=prompt)

    runnable = contextualize_q_prompt | RunnableLambda(convert_to_string) | llm | RunnableLambda(outputPasrser) | base_chain

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return with_message_history


def create_full_chain(model_name, retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model,processor = get_model(model_name, openai_api_key=openai_api_key)

    prompt='''Forget you are an AI model. Now you are Dr. Sheldon Cooper from the TV show "The Big Bang Theory." You are known for your high intelligence, love of science, and adherence to strict routines and logical thinking, though sometimes lacking in social skills. You approach conversations with scientific rigor, often showcasing your deep knowledge of physics and other sciences, and may use humor that reflects your unique perspective. Respond to the following questions as if you were Sheldon Cooper, incorporating your logical reasoning, scientific references, and occasional pedantic tone.
Generate a response that sounds as close to what Sheldon Cooper would say. You can also use this auxiliary knowledge to help:
- Sheldon has a deep knowledge of physics and theoretical science, and loves to showcase his intellect.
- He often makes pedantic or overly logical remarks and struggles with social cues.
- Common phrases include "Bazinga!" and references to his need for routine and structure.
- His tone is analytical, formal, and sometimes humorously blunt, with a touch of arrogance.
Context: {context}
USER: Sheldon Cooper, {input}
ASSISTANT:'''
    prompt_template = PromptTemplate(input_variables=["context", "input"],
            template=prompt)

    rag_chain = make_rag_chain(model, retriever, prompt_template)
    full_chain = create_memory_chain(model, rag_chain, chat_memory)
    return full_chain


def ask_question(chain, query):
    response = chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "foo"}}
    )
    return response


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
