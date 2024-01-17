from dotenv import load_dotenv
import streamlit as st
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

def get_list_questions (filename):
  lines =[]
  with open (filename) as qFile :
    for i in qFile:
      lines.append (i.strip())
  return lines

def main():
    #load_dotenv()

    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

    print ('--------------app start ----------')
    st.set_page_config(page_title="Ask the 10K  PDF")
    st.header("Ask the 10K 💬")

    st.sidebar.header("List of Questions")

    list_of_button =[]  # list of check box in the sidebar

    list_of_questions = get_list_questions('10kq.txt')

    for i in range (len(list_of_questions)):
      # choice_text = f'Q:{list_of_questions[i]}'
      choice_text = list_of_questions[i]
      checkbox_f = st.sidebar.checkbox (choice_text)
      list_of_button.append (checkbox_f)


    submit_button_1 = st.sidebar.button("submit")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()

     # -----split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # ------create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)

      # show user input

      if submit_button_1:  # user click submit button
        for i in range (len(list_of_questions)):   # code detect the click of the list of checkboxes. 
          if list_of_button[i]:
            st.write (f"**{list_of_questions[i]}**")
            # st.write (list_of_questions[i])  
            docs = knowledge_base.similarity_search(list_of_questions[i])

            user_question = list_of_questions [i]
          
            llm = OpenAI(model_name = 'gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=user_question)
              print(cb)
             
            st.write(response)
      
  
if __name__ == '__main__':
    main()
