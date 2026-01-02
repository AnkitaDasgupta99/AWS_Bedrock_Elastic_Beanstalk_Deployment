import json
import os
import sys
import boto3
import streamlit as st

# Access secrets directly
access_key = st.secrets["AWS_ACCESS_KEY_ID"]
secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]

client = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

## We will be using Titan Embeddings Model To generate Embedding

# Robust BedrockEmbeddings import: don't crash at import time; record failure instead
try:
    from langchain_community.embeddings import BedrockEmbeddings
except Exception:
    try:
        from langchain_community.embeddings.bedrock import BedrockEmbeddings
    except Exception:
        BedrockEmbeddings = None
        _bedrock_embeddings_import_error = (
            "Could not import BedrockEmbeddings from langchain_community. "
            "Install/upgrade in your virtualenv: pip install -U langchain-community"
        )
    else:
        _bedrock_embeddings_import_error = None
else:
    _bedrock_embeddings_import_error = None

# Robust Bedrock LLM import
# Robust Bedrock LLM import for 2026
try:
    from langchain_aws import ChatBedrock as Bedrock
    _bedrock_llm_import_error = None
except Exception:
    try:
        from langchain_aws import BedrockLLM as Bedrock
        _bedrock_llm_import_error = None
    except Exception:
        Bedrock = None
        _bedrock_llm_import_error = (
            "Could not import Bedrock LLM from langchain_aws. "
            "Run: pip install -U langchain-aws"
        )

else:
    _bedrock_llm_import_error = None

## Data Ingestion

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain_community.vectorstores import FAISS

## LLm Models
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


## Bedrock Clients
def get_bedrock_client():
    try:
        return boto3.client(service_name="bedrock-runtime")
    except Exception as e:
        raise RuntimeError(f"Failed to create Bedrock boto3 client: {e}")

def get_bedrock_embeddings():
    if BedrockEmbeddings is None:
        raise ImportError(_bedrock_embeddings_import_error)
    client = get_bedrock_client()
    try:
        return BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=client)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize BedrockEmbeddings: {e}")

## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    embeddings = get_bedrock_embeddings()
    vectorstore_faiss = FAISS.from_documents(docs, embeddings)
    vectorstore_faiss.save_local("faiss_index")
#
# def get_claude_llm():
#     if Bedrock is None:
#         raise ImportError(_bedrock_llm_import_error)
#     client = get_bedrock_client()
#     # Updated to Claude 3.5 Sonnet for 2026
#     llm = Bedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", client=client, model_kwargs={'max_tokens': 512})
#     return llm

def get_llama2_llm():
    if Bedrock is None:
        raise ImportError(_bedrock_llm_import_error)
    client = get_bedrock_client()
    llm = Bedrock(
        model_id="us.meta.llama3-1-70b-instruct-v1:0", # Use the profile ID
        client=client,
        model_kwargs={'max_tokens': 512} # Updated parameter name
    )
    return llm



prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                try:
                    get_vector_store(docs)
                    st.success("Done")
                except Exception as e:
                    st.error(f"Failed to create/update vector store: {e}")

    # if st.button("Claude Output"):
    #     with st.spinner("Processing..."):
    #         try:
    #             embeddings = get_bedrock_embeddings()
    #             faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    #             llm = get_claude_llm()
    #             st.write(get_response_llm(llm, faiss_index, user_question))
    #             st.success("Done")
    #         except Exception as e:
    #             st.error(f"Error: {e}")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            try:
                embeddings = get_bedrock_embeddings()
                faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization= True)
                llm = get_llama2_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
