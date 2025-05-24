# RAG-based Movie Chatbot

## Overview

A conversational AI assistant built with Streamlit and Google's Gemini API that answers questions about movies from the IMDB dataset.

**Key Features:**

- Answers questions about 1000+ movies from the IMDB dataset
- Supports complex queries like movie comparisons
- Maintains conversation history
- Clean, user-friendly Streamlit interface
- RAG (Retrieval-Augmented Generation) architecture for accurate responses

---

## Technologies Used

- **Python 3.10+**
- **Streamlit** - Web app framework
- **Google Gemini API** - LLM and embeddings
- **FAISS** - Vector similarity search
- **LangChain** - RAG pipeline construction
- **Pandas** - Data processing

---

## Prerequisites

1. Google API key with access to Gemini models
2. Python 3.10 or later
3. IMDB-Movie-Data.csv file in project root

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [RAG Pipeline](#rag-pipeline)
- [Chatbot](#chatbot)
- [Streamlit Deployment (Bonus)](#streamlit-deployment-bonus)
- [Contributing](#contributing)

---

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/rag-chatbot.git
   ```

2. **Navigate to the project directory:**

   ```sh
   cd rag-chatbot
   ```

3. **Create a virtual environment:**

   ```sh
   python -m venv venv
   ```

4. **Activate the virtual environment:**

   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

5. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

6. **Set up API keys:**
   - Create a [.env](http://_vscodecontentref_/0) file in the project root.
   - Add your API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```
   - Ensure [.env](http://_vscodecontentref_/1) is listed in [.gitignore](http://_vscodecontentref_/2) to keep it private.

> **Note:** Obtain your own API key from your chosen provider (e.g., Google Gemini) and add it to the [.env](http://_vscodecontentref_/3) file.

---

## Usage

1. **Run the chatbot:**

   ```sh
   python rag_app.py
   ```

2. **Interact with the chatbot:**
   - Enter your questions when prompted.
   - Type 'exit' to quit (if running in CLI).

**Sample Questions:**

- What is the plot of Inception?
- Compare Interstellar and Gravity.

Sample questions and responses are saved in [chatbot_responses.txt](http://_vscodecontentref_/4).

---

## Dataset

The chatbot uses [IMDB-Movie-Data.csv](http://_vscodecontentref_/5), located at the project root. This dataset contains movie details and serves as the knowledge base.

---

## RAG Pipeline

The RAG pipeline is built with LangChain’s `RetrievalQA`. It retrieves relevant data from the dataset and generates responses using Google Gemini (or your configured language model).

---

## Chatbot

The chatbot accepts user input, queries the dataset via the RAG pipeline, and provides contextual answers. It ensures accuracy by blending retrieval and generation.

---

## Streamlit Deployment (Bonus)

To run the chatbot as a web app:

1. **Install Streamlit:**

   ```sh
   pip install streamlit
   ```

2. **Launch the app:**

   ```sh
   streamlit run rag_app.py
   ```

3. **Access it:**
   - Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## Contributing

We welcome contributions! To get involved:

- Report issues via GitHub Issues.
- Submit pull requests with your improvements.

---
