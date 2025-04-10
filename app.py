import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader

# Load environment variables
load_dotenv()

# Streamlit interface setup
st.set_page_config(page_title="Phone Shop", page_icon="🤖")
st.title("🎯 Phone Shop")

# Sidebar API key setup
with st.sidebar:
    st.header("Settings")
    if "OPENAI_API_KEY" in os.environ:
        st.success("API key is set!")
    else:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.rerun()

# Custom prompt template 
template = """
You are a helpful chatbot answering questions based on the following context.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Give a helpful and detailed answer in normal text format.
"""
QA_PROMPT = PromptTemplate.from_template(template)

# Document processing function
@st.cache_resource
def load_documents():
    loader = CSVLoader(file_path="data/mobile_products.csv")  
    pages = loader.load()
    
    text_splitter = CharacterTextSplitter(
        separator="।",
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(pages)
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

# Chatbot initialization
@st.cache_resource
def init_chatbot(_db):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0.2 , model="gpt-4o-mini"),
        retriever=_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=True
    )
    return chain

# Main application
def main():
    if "OPENAI_API_KEY" not in os.environ:
        st.warning("Set API key in the sidebar")
        return
    
    # Load documents
    try:
        db = load_documents()
        chain = init_chatbot(db)
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Write your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            try:
                response = chain({"question": prompt})
                answer = response["answer"]
            except Exception as e:
                answer = f"Error: {str(e)}"
        
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()