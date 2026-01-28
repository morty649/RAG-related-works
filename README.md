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
So i considered to go with huggingface models which give same results 90 percent of time

Huggingface models as vector embeddings and google models as LLM work just fine

Vector Store => light weight , mini - version of vector database
Vector Database => full fledged and has many features 


FIASS works great with GPU availability and ChromaDB is production grade

Groq models as LLM works great for production grade because of better rate limits

    Conversational memory is the key for context length

Simple RAG
Streaming RAG - where StrOutputParser is not required
Conversational Memory RAG ***

There is another vectorstore known as InMemoryVectorStore 
    - stores in as dictionary , retrieves using cosine similarity

Datastax Astradb vectorstore connects with Vector database 

This datastax understanding will be done through google colab because of large set of dependancies

There is also Pinecone Vector Database that can be used for free

Chunks after splitting should be contextually rich,self-contained and logically seperated

Similar chunks are merged after splitting so that all the related stuff will be in one place

Initializing an embedding model got easier 

    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer('all-MiniLM-L6-v2')


That specific model is used because no api key required and opensource for embedding

Semantic Chunking - if similarity between chunks are above threshold append one to another to give meaningful context


------------------------------------------------------------------------------------------------------------------


Combining dense and Sparse Matrices to get better at Retrieval Augumented Generation

Sparse => TF-IDF
Dense  => ( Embeddings + Cosine Similarity )

    Score(hybrid) = alpha * Score(dense) + ( 1 - alpha ) * Score(sparse)

Better RAG model



Key Benefits of Hybrid Search

    1. Boosts Recall

        BM25 catches exact keyword matches

        Semantic search captures meaning even when wording differs

        Together, you reduce the chance of missing relevant documents

    2. Handles Synonyms & Rephrasing

        Semantic search can match queries like “create app” → “build LLM system”
        
        BM25 still catches exact terms like “LLM”, “app”

    3. Improves Retrieval Robustness

        Supports both:

        Users who search with precise keywords

        Users who use natural language questions

    4. Preserves Lexical Importance

        BM25 gives higher weight to rare or critical terms

        Essential for technical, legal, or medical domains
        (e.g., rare terms like “osteoporosis”)

    5. Bridges Document Diversity

        Works well across mixed data sources:

        Web pages

        PDFs

        Blogs

        Well-structured + loosely written text

        Hybrid retrieval adapts better than either method alone

    6. Easy to Tune with Weights

        You can control how much each method influences the final result:

            final_score = 0.7 * semantic_score + 0.3 * bm25_score


        This makes optimization simple and flexible.

    7. More Tolerant to Typos & Variants

        Semantic models handle:

        Misspellings

        Word variants

        BM25 alone may fail here, but hybrid search catches more cases.