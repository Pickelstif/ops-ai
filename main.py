import streamlit
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request
import requests

def main():



    load_dotenv()
    streamlit.set_page_config(page_title="Ops-A(i)")
    streamlit.header("Ops-A(i)")
    #upload file
    with open("CX_FOP_OMA_230330.pdf", "rb") as pdf:
    #pdf = streamlit.file_uploader("Upload file:")

        #extract text
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages[250:500]:
                text += page.extract_text()
            #split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            #create embeddings which represent the knowledge base
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            #show user input
            user_question = streamlit.text_input("Ask a question.")

            if user_question:
                important_chunks = knowledge_base.similarity_search(user_question)
                # outputs chunks that are relevant
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=important_chunks, question=user_question)
                streamlit.write(response)




if __name__ == '__main__':
    main()