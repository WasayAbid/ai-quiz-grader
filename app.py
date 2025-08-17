# ==============================================================================
# 1. ALL LIBRARY IMPORTS
# ==============================================================================
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

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
import langchain

# ==============================================================================
# 2. INITIALIZE FLASK APP
# ==============================================================================
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==============================================================================
# 3. FETCH THE API KEYS
# ==============================================================================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not all([PINECONE_API_KEY, GEMINI_API_KEY]):
	raise ValueError("Please set PINECONE_API_KEY and GEMINI_API_KEY in your .env file")

# ==============================================================================
# 4. INITIALIZE PINECONE
# ==============================================================================
print("Initializing Pinecone...")
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
# 5. INITIALIZE EMBEDDINGS AND LLM
# ==============================================================================
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GEMINI_API_KEY, model="models/embedding-001")
llm = ChatGoogleGenerativeAI(
	model='gemini-1.5-pro',
	google_api_key=GEMINI_API_KEY,
	temperature=0.2
)

# ==============================================================================
# 6. FLASK ROUTES
# ==============================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')

@app.route('/grade', methods=['POST'])
def grade():
	if 'file' not in request.files:
		return jsonify({"status": "error", "message": "No file uploaded"}), 400
	file = request.files['file']
	if file.filename == '':
		return jsonify({"status": "error", "message": "No selected file"}), 400
	if file:
		filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
		file.save(filepath)
		try:
			# --- Real-time status updates ---
			status_updates = []
			# 1. Image fetched
			status_updates.append("✅ Image successfully fetched by backend.")
			# 2. OCR processing
			status_updates.append("🔍 Sending image to Google Cloud Vision OCR...")
			client = vision.ImageAnnotatorClient()
			with io.open(filepath, 'rb') as image_file:
				content = image_file.read()
			image = vision.Image(content=content)
			response = client.document_text_detection(image=image)
			quiz_text = response.full_text_annotation.text
			if not quiz_text.strip():
				return jsonify({"status": "error", "message": "No text detected in the image"}), 400
			status_updates.append("✅ OCR completed. Text extracted from image.")
			# 3. Vector store creation
			status_updates.append("📚 Creating vector store from extracted text...")
			text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
			documents = text_splitter.create_documents([quiz_text])
			vector_store = PineconeVectorStore.from_documents(
				documents=documents,
				embedding=embeddings,
				index_name=index_name
			)
			time.sleep(10)  # Wait for Pinecone
			status_updates.append("🔗 Vector store created and connected to Pinecone.")
			# 4. Grading query (plain text)
			status_updates.append("🤖 Sending extracted text to Gemini for grading...")
			
			# --- QUERY WITH MULTI-FACTOR GRADING ---
			query = """
			Review the provided 'Peer Feedback Workshop' document which contains handwritten answers.

			Your task: Act as an English instructor grading the quality of the handwritten peer feedback itself. For each handwritten answer provided in the document, you must analyze and grade it based on the following three criteria:

			1.  **Answer Quality & Insightfulness**: How helpful, specific, and insightful is the feedback? Does it provide clear, actionable advice?
			2.  **Grammar & Spelling**: Is the writing grammatically correct with no spelling errors?
			3.  **Writing Style & Clarity**: Is the feedback written in a clear, concise, and professional manner?

			Please structure your output as a detailed report as follows:

			**Part 1: Question-by-Question Analysis**
			For each of the 7 feedback questions in the document:
			-   First, restate the original question.
			-   Next, summarize the handwritten answer that was provided.
			-   Finally, provide a detailed **"Instructor's Analysis"** that explicitly evaluates the answer against the three criteria above (Answer Quality, Grammar, and Writing Style).

			**Part 2: Final Assessment**
			After analyzing all the answers:
			-   Provide an **"Overall Assessment"** of the reviewer's performance, summarizing their strengths and weaknesses across all three criteria.
			-   Give a **"Final Score out of 10"**, where this single score is a holistic reflection of the combined quality of their answers, their grammar, and their writing style.
			"""
			
			qa_chain = RetrievalQA.from_chain_type(
				llm=llm,
				chain_type="stuff",
				retriever=vector_store.as_retriever()
			)
			response = qa_chain.invoke(query)
			status_updates.append("🎓 Grading completed!")
			result = response['result']
			return jsonify({
				"status": "success",
				"status_updates": status_updates,
				"result": result
			})
		except Exception as e:
			return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500

# ==============================================================================
# 7. RUN THE APP
# ==============================================================================
if __name__ == '__main__':
	app.run(debug=True)