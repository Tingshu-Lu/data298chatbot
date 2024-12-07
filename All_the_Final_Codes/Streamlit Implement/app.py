import streamlit as st

import agent

st.title("Sheldon Bot")
model_name = st.radio("Select a model", ["Llava"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load the model
ragchain = agent.load_ragchain(model_name)

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"{model_name}: {agent.chatbot_response(ragchain,prompt)}"  # model(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

cleared = st.button("Clear chat history")
if cleared:
    st.session_state.messages = []




