from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)

st.title("Basic Q&A Chatbot")
st.header("Ask any question and get an answer!")

ipText = st.text_input("Enter your question here:")
submit = st.button("Ask the Question")

if submit and ipText:
    st.subheader("Fetching answer...")

    prompt = PromptTemplate(
        template="Answer the following question clearly: {question}",
        input_variables=["question"]
    )

    parser = StrOutputParser()

    chain = prompt | model | parser
    response = chain.invoke({"question": ipText})

    st.subheader("Answer:")
    st.write(response)
