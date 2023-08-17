import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import JinaChat
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def conversation_chain(vectorestore):
    llm = JinaChat()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorestore.as_retriever(),
        memory=memory
    )
    return conversation

def handle_input(question):
    response  = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)



def main():
    load_dotenv()
    st.set_page_config(page_title="PDFs chatbot", page_icon='📕')

    # When using session_state, variables must be declared before using
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    question = st.chat_input("Ask a question about your files")
    if question:
        handle_input(question)
  
    #Sidebar
    #Title and documents uploader
    with st.sidebar:
        st.title("🤖💬📕 PDFs chatbot")
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your documents 📁", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing..."):
                #Get text from pdfs
                raw_text = get_pdf_text(pdf_docs)

                #Get the text chunks 
                text_chunks = get_chunks(raw_text)

                #Create vector store
                vectorestore = get_vector_store(text_chunks)

                #Create conversation chain
                st.session_state.conversation = conversation_chain(vectorestore)
    



    #Create some space between the uploader and contact info
    with st.sidebar.container():
        st.text("")
        st.markdown("***")

    #Contact info
    with st.sidebar.container():
        st.subheader(
            "A Streamlit web app by [Hiram Cortes](https://www.linkedin.com/in/hdcortesd/)"
            )
        st.markdown("You can find the source code on my [GitHub](https://github.com/MaxPower14/pdfs-chatbot-streamlit)")


if __name__ == '__main__':
    main()