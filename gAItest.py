# ==============================================================================
# 1. ALL LIBRARY IMPORTS
# ==============================================================================
import os
from dotenv import load_dotenv

# LangChain and Pinecone components
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Pinecone v3+ SDK
from pinecone import Pinecone, ServerlessSpec

# Google Cloud Vision for OCR
from google.cloud import vision
import io
import time

# --- NEW: Import LangChain for debugging ---
import langchain

# --- NEW: Enable LangChain's debug mode ---
# This will print every step the chain takes, including the final prompt to Gemini.
langchain.debug = True

# ==============================================================================
# 2. FETCH THE API KEYS
# ==============================================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([PINECONE_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Please set PINECONE_API_KEY and GEMINI_API_KEY in your .env file")

# ==============================================================================
# 3. INITIALIZE PINECONE
# ==============================================================================
print("Initializing Pinecone (v3+ SDK)...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "handwriting-quiz-grader"
embedding_dim = 768

if index_name not in pc.list_indexes().names():
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Using existing index: {index_name}")

# ==============================================================================
# 4. INITIALIZE EMBEDDINGS
# ==============================================================================
print("Initializing Google Generative AI embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GEMINI_API_KEY, model="models/embedding-001")

# ==============================================================================
# 5. READ IMAGE WITH GOOGLE CLOUD VISION API (OCR)
# ==============================================================================
image_path = "sample_img.png"
print(f"Reading handwriting from image with Google Cloud Vision: {image_path}")

try:
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    
    response = client.document_text_detection(image=image)
    quiz_text = response.full_text_annotation.text

    if response.error.message:
        raise Exception(f"Google Cloud Vision API Error: {response.error.message}")

except FileNotFoundError:
    raise FileNotFoundError(f"The image file '{image_path}' was not found.")

if not quiz_text.strip():
    raise ValueError("Google Cloud Vision did not extract any text.")

# --- NEW: Log the text extracted by Google Vision ---
print("\n" + "="*20 + " TEXT EXTRACTED BY GOOGLE VISION " + "="*20)
print(quiz_text)
print("="*75 + "\n")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.create_documents([quiz_text])
print(f"Image text successfully extracted and split into {len(documents)} documents.")

# ==============================================================================
# 6. CREATE AND POPULATE THE LANGCHAIN VECTOR STORE
# ==============================================================================
print("Connecting to Pinecone vector store and adding documents...")
vector_store = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)
# --- THIS IS THE CRITICAL FIX ---
# Wait for 10 seconds to give Pinecone time to index the new documents.
print("Waiting for 10 seconds for Pinecone to index the data...")
time.sleep(10)
# --- END OF FIX ---
print("Vector store is ready.")

# ==============================================================================
# 7. INITIALIZE THE GEMINI MODEL
# ==============================================================================
print("Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro',
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)

# ==============================================================================
# 8. INITIALIZING THE RETRIEVALQA CHAIN
# ==============================================================================
print("Initializing RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)
# ==============================================================================
# 9. RUNNING THE GRADING QUERY
# ==============================================================================

# --- NEW: A query that is highly specific to the content of the image ---
# The first paragraph is filled with keywords for the RETRIEVER to find the document.
# The second part contains instructions for the LLM after the document is found.
query = """
Review the provided document about "ensemble learning". The document answers the question "What is ensemble learning? When to use ensemble learning?" and discusses bagging, boosting, variance, and bias.

Using that specific document as the student's answer, please act as a teacher and grade it by following these steps:
1.  **Correct Answer**: First, provide a well-structured, correct answer to the original question.
2.  **Student's Answer Review**: Analyze the student's answer from the context. Point out correct statements and identify areas where information is missing or incorrect.
3.  **Grammar & Spelling**: Correct any grammatical or spelling mistakes found in the student's answer.
4.  **Score and Feedback**: Assign a score out of 10 and provide brief, overall feedback on the performance.

Return the result as a structured report.
"""

print("\nSending extracted text to the RAG system for grading...")
response = qa_chain.invoke(query)

# ==============================================================================
# 10. PRINT THE FINAL REPORT
# ==============================================================================
print("\n" + "="*25 + " GRADING REPORT " + "="*25)
print(response['result'])
print("="*72)