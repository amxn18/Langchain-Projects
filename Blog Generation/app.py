from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.4
)

parser = StrOutputParser()
model = ChatHuggingFace(llm=llm)

st.set_page_config(
    page_title="Blog Post Generator",
    layout="wide",
    initial_sidebar_state='collapsed'
)

st.title("Blog Post Generator")

inputText = st.text_area(
    "Enter a topic or title for the blog post:", 
    height=150,
    placeholder="e.g., The Future of Artificial Intelligence"
)
col1, col2 = st.columns([5, 5])
with col1:
    noOfWords = st.slider(
        "Select the approximate number of words for the blog post:",
        min_value=100,
        max_value=2000,
        value=500,
        step=50
    )
with col2:
    blogStyle = st.selectbox(
        "Select the style of the blog post:",
        options=["Informative", "Casual", "Professional", "Humorous", "Inspirational"]
    )

generateButton = st.button("Generate Blog Post", type="primary")

if generateButton:
    if inputText:
        prompt = PromptTemplate(
            input_variables=["topic", "word_count", "style"],
            template="Write a {word_count}-word {style} blog post on the topic: {topic}."
        )
        with st.spinner("Generating blog post..."):
            chain = prompt | model | parser
            response = chain.invoke({"topic": inputText, "word_count": noOfWords, "style": blogStyle.lower()})

        st.subheader("Generated Blog Post")
        st.markdown(response)
    else:
        st.warning("Please enter a topic or title for the blog post.")