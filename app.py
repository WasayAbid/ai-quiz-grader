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
			query = """
Review the provided 'Peer Feedback Workshop' document. 
The document contains handwritten answers to 7 peer-review questions about an argument essay.

Here are the 7 peer-review questions in the document:
1. What (in your own words) is the essay’s thesis? 
2. Does the thesis assert a specific, explicit, debatable, and defendable claim? If not, which areas need work? 
3. Does each body paragraph have a well-developed topic sentence that supports the thesis or offers a counter-argument to the thesis? 
4. Does each paragraph have sufficient support? 
5. How many reasons does the writer offer to support his/her claim? In abbreviated form list the writer’s reasons. 
6. How clear is the essay’s structure? Should the reasons be rearranged to add emphasis to the writer’s argument? Is the logic coherent? 
7. What are some of the warrants of the essay? Do any of these assumptions need to be explicitly addressed by the author? 

Your task: Using the specific handwritten answers from the document as context, act as an instructor and analyze the feedback by following these steps:

For each of the 7 questions:
1. Restate the original peer-review question. 
2. Summarize the handwritten feedback provided. 
3. Analyze the quality and clarity of the feedback (e.g., is it constructive, specific, vague, insightful, etc.?). 
4. Suggest how the feedback itself could be improved to make it more useful to the essay writer. 

After analyzing all 7 questions separately:
- Provide an **Overall Assessment**: Highlight the key strengths and weaknesses of the peer feedback as a whole. 
- Give a **Score (1–10)** for the overall quality of the feedback. 
- Return the result as a **structured report**, with clear headings for each question and the overall assessment.
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