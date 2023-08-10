import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from streamlit_chat import message

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
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-base-en")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def conversation_chain(vectorestore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorestore.as_retriever(),
        memory=memory
    )
    return conversation

def handle_input(question):
    response  = st.session_state.conversation({'question': question})
    st.write(response)




def main():
    load_dotenv()
    st.set_page_config(page_title="PDFs chatbot", page_icon='📕')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    question = st.text_input("Ask a question about your files")
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
        st.markdown("You can find the source code on my [GitHub](https://github.com/MaxPower14)")


if __name__ == '__main__':
    main()