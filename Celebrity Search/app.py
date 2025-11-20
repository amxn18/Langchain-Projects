from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import streamlit as st

load_dotenv()

prompt1 = PromptTemplate(
    template="Provide a brief biography and notable achievements of celebrity {celebrity_name}.",
    input_variables=["celebrity_name"]
)

prompt2 = PromptTemplate(
    template="What is the date of birth (DOB) of the celebrity named {celebrity_name}?",
    input_variables=["celebrity_name"]
)

prompt3 = PromptTemplate(
    template="List the most recent and notable headlines involving the celebrity {celebrity_name}.",
    input_variables=["celebrity_name"]
)

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.3,
    max_new_tokens=300,
)

parser = StrOutputParser()
model = ChatHuggingFace(llm=llm)

parallel_chain = RunnableParallel(
    biography = prompt1 | model | parser,
    dob       = prompt2  | model | parser,
    headlines = prompt3 | model | parser
)


st.title("Celebrity Search")

ipText = st.text_input("Enter celebrity name:")

if ipText:
    st.write(f"Searching information about **{ipText}**...")

    outputs = parallel_chain.invoke({"celebrity_name": ipText})

    st.subheader("Biography & Achievements")
    st.write(outputs["biography"])

    st.subheader("Date of Birth")
    st.write(outputs["dob"])

    st.subheader("Recent Headlines")
    st.write(outputs["headlines"])
