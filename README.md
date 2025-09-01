

# 🤖 AI Quiz Grader

An automated grading system for handwritten quizzes using Google Cloud Vision (OCR), Retrieval-Augmented Generation (RAG), and Gemini. This project allows educators to upload a quiz image and receive an AI-generated grade and feedback in seconds.

 
*You can use the Imgur link above or upload your own workflow image to the repository and link it here.*

---

## ✨ Key Features

-   **📸 Upload & Go:** Simply upload a quiz image to get a grade and feedback.
-   **✍️ Handwriting Recognition:** Extracts handwritten text using **Google Cloud Vision OCR**.
-   **🧠 RAG-Powered Grading:** Uses a **LangChain** and **Pinecone** pipeline for accurate, context-aware evaluation.
-   **📊 Comprehensive Feedback:** Grades on accuracy, grammar, and writing style with **Google Gemini**.

## 🛠️ Tech Stack

**Python** | **Flask** | **LangChain** | **Pinecone** | **Google Cloud Vision** | **Google Gemini**

## 🚀 Quick Start

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/WasayAbid/ai-quiz-grader.git
    cd ai-quiz-grader
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your credentials:**
    -   Enable the Google Cloud Vision API and create a **service account JSON key**.
    -   Create a `.env` file (use `.env.example` as a template) and add your `PINECONE_API_KEY`, `GOOGLE_API_KEY`, and the path to your service account JSON file.

4.  **Run the app:**
    ```bash
    python app.py
    ```
    Then open your browser to `http://127.0.0.1:5000`.
