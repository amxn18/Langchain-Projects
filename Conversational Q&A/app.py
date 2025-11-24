from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Conversational Chatbot")
st.title("Conversational Q&A Chatbot")
st.header("Hey, Let's Chat!")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=256
)
model = ChatHuggingFace(llm=llm)
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        SystemMessage(content="You are a helpful assistant.")
    ]

def generate_response(user_input):
    st.session_state['messages'].append(HumanMessage(content=user_input))
    response = model.invoke(st.session_state["messages"])
    st.session_state['messages'].append(AIMessage(content=response.content))
    return response.content

user_input = st.text_input("You: ", "")
if st.button("Send"):
    if user_input:
        response = generate_response(user_input)
        st.write(f"Bot: {response}")
    