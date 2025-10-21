# **Proof of Concept: RAG AI**

## Description
RAG AI Concept is a Proof of Concept (PoC) project that uses Retrieval-Augmented Generation (RAG) technology to implement an AI system capable of accessing and leveraging information from various documents uploaded by users. This system combines search (retrieval) and AI-driven text generation to provide answers based on relevant document content.

This project aims to demonstrate how RAG can be applied in real-world scenarios to optimize information retrieval and generate more accurate and informative answers from documents.

## Features
### 1. **Upload Documents**
Users can upload various types of documents that will be indexed and used in the information retrieval process.

### 2. **Load Index**
Once a document is uploaded, the system will build and load an index from the document. This index is used to speed up the retrieval of relevant information when needed.

### 3. **Save Session**
Users can save their conversation session for future reference or continued use.

### 4. **New Session**
Users can start a new session by selecting the document they want to use, initiate a search, or continue a previous session.

### 5. **Download Chat**
Users can download the conversation from the session as a text file for documentation or further reference.

## Tech Stack
- **Python**: The primary programming language used for developing the system.
- **Gemini**: An AI technology used for natural language processing and context-based search.
- **Streamlit**: A Python framework used to build interactive web applications.

## Installation
To get started, you need to install the required dependencies and configure some initial settings.

### 1. **Clone the Repository**
Clone this repository to your local machine:
```bash
git clone https://github.com/rizts/rag-ai-concept.git
cd rag-ai-concept
```
### 2. **Create a Virtual Environment (Optional)**
To avoid dependency issues, it's recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```
### 3. **Install Dependencies**
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
### 4. **Add API Key**
This project requires an API key for Gemini. You can add it by:
* Creating a `.env` file in the root directory of the project, refer to .env.example.
* Adding the following variable to the `.env` file:
```bash
GEMINI_API_KEY=your-api-key-here
```
Make sure to replace `your-api-key-here` with a valid API key from Gemini.

### 5. **Run the Server with Streamlit**
To run the web application with Streamlit, use the following command:
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501` in your browser.

### **Project Structure**
```bash
├── app.py
├── docs/
├── README.md
├── requirements.txt
└── semantic_search.py
```

### **Usage**
1. Upload Document: Click the button to upload a document you want to index and use for information retrieval.

2. Start a New Session: After uploading a document, select a new session to start asking AI questions based on the document's content, or when you want to start new session again.

3. Save Session: If you want to save your conversation session for future reference, select the "Save Session" option.

4. Download Chat: You can download the conversation that occurred during the session as a JSON file.

### **Support, Suggestions, Feedback, and Contributions**

We greatly appreciate your feedback! If you have questions, suggestions, or comments, feel free to open an issue on the GitHub repository or [email me](mailto:rizts.tech@gmail.com)

### **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

###

Happy coding! 🚀