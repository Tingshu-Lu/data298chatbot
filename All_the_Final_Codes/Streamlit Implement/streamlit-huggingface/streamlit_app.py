import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings

# from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# from ensemble import ensemble_retriever_from_docs
from full_chain import ask_question
# from local_loader import load_txt_files

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("Chatbot with Personalities")


def show_ui(qa, model_name="ChatGPT", prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt, model_name)
                st.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


@st.cache_resource
def get_retriever(openai_api_key=None):
    # docs = load_txt_files()
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base",model_kwargs={'device':'cuda'})
    ragdb = Chroma(persist_directory="../Lincoln_DB", embedding_function=embeddings)
    retriever = ragdb.as_retriever(search_kwargs={'k': 3})

    return retriever

@st.cache_resource
def get_chain(model_name, openai_api_key=None):
    if model_name == "Gemma":
        return None
    ensemble_retriever = get_retriever(openai_api_key=openai_api_key)
    chain = create_full_chain(model_name, ensemble_retriever, openai_api_key=openai_api_key,
                              chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain


def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def run():
    ready = True

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    with st.sidebar:
        model_name = st.radio("Select a model", ["Gemma","ChatGPT"])
        # if not openai_api_key:
        #     openai_api_key = get_secret_or_input('OPENAI_API_KEY', "OpenAI API key",
        #                                          info_link="https://platform.openai.com/account/api-keys")
        # if not huggingfacehub_api_token:
        #     huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
        #                                                    info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

    if (not openai_api_key) and (model_name=='ChatGPT'):
        st.warning("Missing OPENAI_API_KEY")
        ready = False
    # if not huggingfacehub_api_token:
    #     st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
    #     ready = False

    if ready:
        # chain = get_chain(model_name,openai_api_key=openai_api_key)
        chain = "" # from API call
        # st.subheader("")
        show_ui(chain, model_name, "Hi, this is President Abraham Lincoln.")
    else:
        st.stop()


run()
