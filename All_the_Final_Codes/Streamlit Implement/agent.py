from functools import partial
import logging

from dotenv import load_dotenv 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForPreTraining,BitsAndBytesConfig,pipeline
from peft import PeftModel
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
load_dotenv()

_models = {}

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
        device_map="auto",  # Automatically map the model to available devices
    )

    # Load the processor for LLava 7b
    # processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Load the tokenizer for LLava 7b
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token
    tokenizer.padding_side = "right"  # Ensure padding is applied on the right side

    return model, tokenizer

# Example usage:


def load_ragchain(model_name: str):
    if model_name in _models:
        model, tokenizer = _models.get(model_name)
    elif model_name == "Llava":
        MODEL_NAME = "llava-hf/llava-1.5-7b-hf"  # Replace with the actual path or model name for LLava 7b
        model, tokenizer = create_model_and_processor(MODEL_NAME)
        model = PeftModel.from_pretrained(model, "trained_model4/checkpoint-60")
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

        text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.6,
        repetition_penalty=1.2,
        return_full_text=True,
        max_new_tokens=100,
        top_k=30,
        top_p= 0.8,
        no_repeat_ngram_size =2
    )
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        model_name = "moka-ai/m3e-base"

        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        embeddings.client = SentenceTransformer(
        embeddings.model_name, device='cuda')
        ragdb = Chroma(persist_directory="sheldon_DB", embedding_function=embeddings)
        retriever = ragdb.as_retriever(search_kwargs={'k': 1})
        prompt_template = """
    ### Instruction: You are an expert at impersonating animated characters. In particular, you are a world-leading expert at answering questions in the manner of Dr. Sheldon Cooper from the popular TV series *The Big Bang Theory*.
[INST]Generate a response that sounds as close to what Sheldon Cooper would say. You can also use this auxiliary knowledge to help:
    - Sheldon has a deep knowledge of physics and theoretical science, and loves to showcase his intellect.
    - He often makes pedantic or overly logical remarks and struggles with social cues.
    - Common phrases include "Bazinga!" and references to his need for routine and structure.
    - His tone is analytical, formal, and sometimes humorously blunt, with a touch of arrogance.

    {context}
    ### Question:
    {question}
    ### Response:[/INST]
    """

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)
    else:
        raise ValueError(f"Model {model_name} not supported.")   
    return rag_chain



def chatbot_response(rag_chain,message):
    op = rag_chain.invoke(message)
    response = str(op['text']).split(':[/INST]')[-1]
    return response


