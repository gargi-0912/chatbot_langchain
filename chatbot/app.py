import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# pip install langchain langchain-community langchain-groq streamlit python-dotenv

# Initialize LLM
llm = ChatGroq(
    temperature=0.5,
    groq_api_key=api_key,
    model_name="llama3-70b-8192"  # safer than mixtral which may be deprecated
)

# Prompt
template = """You are an AI assistant. Answer clearly.
Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.set_page_config(page_title="Groq Chatbot", layout="centered")
st.title("🧠 LangChain Chatbot (Groq API)")

user_input = st.text_input("Ask me anything:", "")

if user_input:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_input})
        st.success(response['text'])
