import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from full_chain import ask_question

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("Chatbot with Personalities")


def show_ui(qa, model_name="ChatGPT", prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt, model_name)
                st.markdown(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)


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
        previous_model = st.session_state.get("selected_model", None)
        model_name = st.radio("Select a model", ["Sheldon Cooper", "Richard Feynman", "Abraham Lincoln", "Sherlock Holmes", "Malcolm X"])
        if model_name != previous_model:
            st.session_state.messages = [{"role": "assistant", "content": f"Hi, this is {model_name}."}]
            st.session_state["selected_model"] = model_name

    if (not openai_api_key) and (model_name=='ChatGPT'):
        st.warning("Missing OPENAI_API_KEY")
        ready = False

    if ready:
        chain = ""
        show_ui(chain, model_name, "Hi.")
    else:
        st.stop()


run()