import dotenv
from dotenv import load_dotenv
import streamlit as st

import PyPDF2
from PyPDF2 import PdfReader
import os

import langchain
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    print (os.getenv("OPENAI_API_KEY"))
    OPENAI_API_KEY = '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'
    
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask the 10K 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()

        st.write (text)
     # save the extracted text to a file
     # my_file = open("pdf_extract.txt", "w", encoding='utf-8') 
     # my_file.write (text)

     # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
      
  
if __name__ == '__main__':
    main()
