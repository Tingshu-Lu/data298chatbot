import os

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import AutoProcessor,AutoModelForPreTraining,BitsAndBytesConfig,pipeline,AutoTokenizer,AutoModelForCausalLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from peft import PeftModel
import torch

API_URL_Mistral = "https://cvdmw6jwoq5gzmq6.us-east-1.aws.endpoints.huggingface.cloud"
HEADERS_Mistral = {
    "Accept": "application/json",
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

API_URL_Bloom = "https://ba9h7h3unh6m25wn.us-east-1.aws.endpoints.huggingface.cloud"
HEADERS_Bloom = {
    "Accept": "application/json",
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

API_URL_Gemma = "https://sgoj5p5k9mab0qb7.us-east-1.aws.endpoints.huggingface.cloud"
HEADERS_Gemma = {
    "Accept": "application/json",
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

API_URL_Llama = "https://psczb1gh76yzdsuq.us-east-1.aws.endpoints.huggingface.cloud"
HEADERS_Llama = {
    "Accept": "application/json",
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

API_URL_Llava = "https://qimb4tco992i5qk9.us-east-1.aws.endpoints.huggingface.cloud"
HEADERS_Llava = {
    "Accept": "application/json",
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def convert_to_string(prompt_value):
    return prompt_value.to_string()

def get_first_paragraph(text):
    paragraphs = text.split('\n\n')
    if paragraphs:
        return paragraphs[0]
    else:
        return ""

def query_url(payload, api_url, headers):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()


def ask_question(chain, query, model_name):
    if model_name == "Abraham Lincoln":
        return get_first_paragraph(query_url({
            "inputs": query,
            "parameters": {}
        }, API_URL_Gemma, HEADERS_Gemma)[0]["result"])
    elif model_name == "Richard Feynman":
        return get_first_paragraph(query_url({
            "inputs": query,
            "parameters": {}
        }, API_URL_Bloom, HEADERS_Bloom)[0]["result"])
    elif model_name == "Sherlock Holmes":
        return get_first_paragraph(query_url({
            "inputs": query,
            "parameters": {}
        }, API_URL_Mistral, HEADERS_Mistral)[0]["result"])
    elif model_name == "Malcolm X":
        return get_first_paragraph(query_url({
            "inputs": query,
            "parameters": {}
        }, API_URL_Llama, HEADERS_Llama)[0]["result"])
    elif model_name == "Sheldon Cooper":
        return get_first_paragraph(query_url({
            "inputs": query,
            "parameters": {}
        }, API_URL_Llava, HEADERS_Llava)[0]["result"])
