import streamlit as st
import warnings
import nltk
import json
import uuid
import os
from io import StringIO
from lxml import etree
import pandas as pd
from typing import Any
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

warnings.filterwarnings('ignore')

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

load_dotenv(find_dotenv())

st.title("Financial PDF Q/A Bot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    pdf_path = "temp.pdf"

    elements = partition_pdf(filename=pdf_path)

    element_dict = [element.to_dict() for element in elements]
    output = json.dumps(element_dict, indent=2)

    unique_element_type = set()

    for element in element_dict:
        unique_element_type.add(element["type"])

    unstructured_api_key = 'Your_unstructured_api_key'
    unstructured_api_url = 'https://api.unstructuredapp.io/general/v0/general'

    client = UnstructuredClient(
        api_key_auth=unstructured_api_key,
        server_url=unstructured_api_url
    )

    with open(pdf_path, "rb") as f:
        files = {
            "content": f.read(),
            "file_name": pdf_path
        }

    req = {
        "partition_parameters": {
            "files": files,
            "strategy": shared.Strategy.HI_RES,  
            "hi_res_model_name": "yolox",
            "skip_infer_table_types": [],
            "pdf_infer_table_structure": True
        }
    }

    try:
        resp = client.general.partition(request=req)  
        elements = resp.elements  
    except SDKError as e:
        st.error(e)

    unique_element_type = set()

    for element in elements:
        unique_element_type.add(element["type"])

    tables = [element for element in elements if element["type"] == "Table"]

    if tables:
        first_table_html = tables[0]["metadata"]["text_as_html"]

        parser = etree.XMLParser(remove_blank_text=True)
        file_obj = StringIO(first_table_html)
        tree = etree.parse(file_obj, parser)

        dfs = pd.read_html(first_table_html)
        df = dfs[0]

    texts = [el for el in elements if el["type"] != "Text"]
    extracted_text = ""

    for cat in elements:
        if cat["type"] == "Formula":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "FigureCaption":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "NarrativeText":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "ListItem":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "Title":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "Address":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "EmailAddress":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "Table":
            extracted_text += cat["metadata"]["text_as_html"] + "\n"
        if cat["type"] == "Header":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "Footer":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "CodeSnippet":
            extracted_text += cat["text"] + "\n"
        if cat["type"] == "UncategorizedText":
            extracted_text += cat["text"] + "\n"


    class Element(BaseModel):
        type: str
        page_content: Any

    categorized_elements = []

    for element in elements:
        if "Table" in element.get("type", ""):  
            categorized_elements.append(Element(type="table", page_content=element.get("metadata", {}).get("text_as_html", "")))
        elif "NarrativeText" in element.get("type", ""):
            categorized_elements.append(Element(type="text", page_content=element.get("text", "")))
        elif "ListItem" in element.get("type", ""):
            categorized_elements.append(Element(type="text", page_content=element.get("text", "")))
        elif "Title" in element.get("type", ""):
            categorized_elements.append(Element(type="text", page_content=element.get("text", "")))
        elif "Address" in element.get("type", ""):
            categorized_elements.append(Element(type="text", page_content=element.get("text", "")))
        elif "EmailAddress" in element.get("type", ""):
            categorized_elements.append(Element(type="text", page_content=element.get("text", "")))
        elif "Header" in element.get("type", ""):
            categorized_elements.append(Element(type="CodeSnippet", page_content=element.get("text", "")))
        elif "CodeSnippet" in element.get("type", ""):
            categorized_elements.append(Element(type="text", page_content=element.get("text", "")))
        elif "UncategorizedText" in element.get("type", ""):
            categorized_elements.append(Element(type="text", page_content=element.get("text", "")))

    
    table_elements = [element for element in categorized_elements if element.type == "table"]

    text_elements = [element for element in categorized_elements if element.type == "text"]

    all_table_html = ""

    for table in table_elements:
        all_table_html += table.page_content + "</br></br>"

    os.environ["OPENAI_API_KEY"] = "Your_openai_api_key"

    # LCEL
    summary_chain = (
        {"doc": lambda x: x}
        | ChatPromptTemplate.from_template("Summarize the following tables or text given below:\n\n{doc}")
        | ChatOpenAI(max_retries=3)
        | StrOutputParser()
    )

    # Table summaries
    tables_content = [i.page_content for i in table_elements]
    teable_summaries = summary_chain.batch(tables_content, {"max_concurrency": 5})

    # Text summaries
    text_content = [i.page_content for i in text_elements]
    text_summaries = summary_chain.batch(text_content, {"max_concurrency": 5})

    store = InMemoryStore()
    id_key = "doc_id"

    vectorstore = Chroma(
        collection_name="financials",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_data",
    )

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    doc_ids = [str(uuid.uuid4()) for _ in table_elements]
    summary_tables = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(teable_summaries)]

    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(doc_ids, table_elements)))

    doc_ids = [str(uuid.uuid4()) for _ in text_elements]
    summary_texts = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(text_summaries)]

    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, text_elements)))

    os.environ["GROQ_API_KEY"] = "Your_groq_api_key"

    # Initialize the LLM and chain
    llm = ChatGroq(
        temperature=0.7,
        max_tokens=1000,
        model_name="llama-3.3-70b-specdec",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    st.header("Chat with the Document")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        bot_response = chain.invoke(user_input)

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(bot_response)