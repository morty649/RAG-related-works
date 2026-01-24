Learnings And Projects related to

RETRIEVAL AUGMENTED GENERATION (RAG)
using LangChain and LangGraph


For installation of uv :

    curl -LsSf https://astral.sh/uv/install.sh | sh

In terminal:
uv init
uv venv
uv add -r requirements.txt

In Data Ingestion part :
LangChain wants us to convert the given data in particular format :
{
    page_content:"content",
    metadata(dict):""
}

TextLoader and DirectoryLoader 

CharacterTextSplitter - normal,based on given chunk size , no intelligence
RecursiveCharacterTextSplitter - hierarchy wise splitting ["\n\n","\n"] => first \n\n then \n ,intelligent splitting
TokenTextSplitter - based on tokens, it splits

Loading a PDF file
pypdf - simple but slow
pymupdf - speed

Pdf has some issues 
    - may contain images
    - formatting issues

    Creating smart chunks after cleaning those issues and  creating better metadata for chunks 

Parsing a docx
    - Docx2txtLoader => Simple,plain,dumb,fast
    - UnstructuredWordDocumentLoader => Parses the document into semantic elements
        Preserves structure:
        Titles
        Headings
        Paragraphs
        ists
        Tables
        Page breaks
        Adds metadata

Parsing a CSV file and Excel file is quite similar
    - CSVLoader 
    - UnstructuredCSVLoader => typically we create custom meta-data and page_content taken from csv columns

    EXCEL loading is done with pandas and also unstructured but unstructured doesn't seem to work so partition_xlsx did the job

Json parsing is can be done through langchain module or using custom way to get better readable documents

SQL databases can be loaded through 
    
    from langchain_community.utilities import SQLDatabase

or also creating a custom loader which makes it a readable document by LLM

Cosine Similarity :

    Cosine Similarity =     A.B / |A||B|

There is a chance to work with OPENAI embedding models but that is pricey
