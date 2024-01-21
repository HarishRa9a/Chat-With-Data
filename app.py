# import libraries
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
os.environ['GOOGLE_API_KEY'] =  'Your API'

def create_conversational_chain(pdf):
    # convert document to raw text
    text=""
    pdf_reader= PdfReader(pdf)
    for page in pdf_reader.pages:
        text+= page.extract_text()

    # split the raw text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = GooglePalmEmbeddings()
    # create vectorstore
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    print(vector_store)
    
    # import llm model
    llm=GooglePalm()
    # memory to store previous conversation
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    # use llm and retriever to create conversation_cahin
    conv_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conv_chain

def handle_user(user_query):
    response = st.session_state.conversation({'question': user_query})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write(":boy:: ", message.content)
        else:
            st.write(":robot_face:: ", message.content)

def main():
    # create UI using streamlit
    st.set_page_config(page_title="CwD",page_icon=":books:")
    st.subheader("Upload your Document")
    pdf_docs = st.file_uploader("Upload your PDF", accept_multiple_files=True,type=['pdf'])
    if st.button("Upload"):
        if len(pdf_docs)==0:
            st.error("Please select a document")
        else:
            with st.spinner("Processing"):
                st.session_state.conversation = create_conversational_chain(pdf_docs[0])
                st.success("Done")
    if len(pdf_docs)!=0:
        st.write("Uploaded Document: ",pdf_docs[0].name)
    st.header("Chat with Data💬")
    user_query = st.text_input("Ask a Question")
    if user_query:
        if len(pdf_docs)==0:
            st.error("No document found")
        else:
            handle_user(user_query)

if __name__ == "__main__":
    main()
