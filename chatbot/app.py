from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  #Default output parser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
#Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

#Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant. Please response to the queries'),
        ('user','Question:{question}')
    ]
)

# Streamlit framework

st.title('Langchain Demo With OPENAI API')
input_text=st.text_input('Search the topic you want')


#OpenAI LLM

llm=ChatOpenAI(model='gpt-3.5-turbo')
output_parser=StrOutputParser()   # Responsible for output
chain=prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
