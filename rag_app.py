import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import streamlit as st

# Load environment variables
load_dotenv()

class MovieChatbot:
    def __init__(self):
        self.conversation_log = st.session_state.get('conversation_log', [])
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            st.error("ERROR: Set GOOGLE_API_KEY in .env file")
            raise ValueError("ERROR: Set GOOGLE_API_KEY in .env file")
        
        self.data_file = "IMDB-Movie-Data.csv"
        self.qa_chain = self.rag_system()

    def load_data(self):
        try:
            df = pd.read_csv(self.data_file)
            
            df['Revenue (Millions)'] = df['Revenue (Millions)'].fillna('N/A')
            df['Actors'] = df['Actors'].str.replace('"', '')
            
            # Convert to LangChain documents
            documents = []
            for _, row in df.iterrows():
                doc = Document(
                    page_content=(
                        f"Title: {row['Title']}\n"
                        f"Year: {row['Year']}\n"
                        f"Director: {row['Director']}\n"
                        f"Actors: {row['Actors']}\n"
                        f"Rating: {row['Rating']}\n"
                        f"Runtime: {row['Runtime (Minutes)']} mins\n"
                        f"Genre: {row['Genre']}\n"
                        f"Box Office: ${row['Revenue (Millions)']}M\n"
                        f"Description: {row['Description']}"
                    ),
                    metadata={
                        'title': row['Title'],
                        'year': int(row['Year']),
                        'director': row['Director'],
                        'actors': row['Actors'],
                        'runtime': int(row['Runtime (Minutes)']),
                        'rating': float(row['Rating']),
                        'genre': row['Genre'],
                        'revenue': f"${row['Revenue (Millions)']}M" if row['Revenue (Millions)'] != 'N/A' else 'N/A',
                        'description': row['Description']
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def rag_system(self):
        """Set up RAG pipeline with enhanced prompt engineering"""
        data = self.load_data()
        if not data:
            raise ValueError("Failed to load movie data")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(data)
        
        # vector store
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.gemini_api_key
        )
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        #LLM configuration 
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.3
        )
        
        #prompt for enhhanced response generation
        prompt_template = """You are a movie expert assistant. Answer questions based strictly on the context below.
        
        Context: {context}

        Question: {question}

        Response Guidelines:
        
        1. Answer questions precisely based on the context provided.

        2. For comparison queries (multiple movies):
           "Comparison between [Title1] and [Title2]:
           
           [Title1] ([Year1])
           - Runtime: [X] minutes
           - Box Office: $[X]M
           - Rating: [Rating]/10
           
           [Title2] ([Year2])
           - Runtime: [X] minutes
           - Box Office: $[X]M
           - Rating: [Rating]/10
           
           Analysis: [Brief comparison analysis]"

        3. Always include year in responses
        4. For missing data: "Data not available for [specific info]"
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={
                "prompt": PROMPT,
                "document_variable_name": "context"
            },
            return_source_documents=True
        )

    def format_response(self, response):
        """Format the response for better readability"""
        if "Comparison between" in response:
            parts = response.split("Analysis:")
            comparison = parts[0].strip()
            analysis = parts[1].strip() if len(parts) > 1 else ""
            formatted = f"{comparison}\n\nAnalysis:\n{analysis}"
            return formatted.replace("\n", "\n    ")
        
        return response.replace("\n", "\n    ")

    def log_conversation(self, question, response):
        """Log the conversation to a file and session state"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_log_entry = (
            f"\n{'='*50}\n"
            f"{timestamp}\n"
            f"Q: {question}\n\n"
            f"A: {self.format_response(response)}\n"
            f"{'='*50}\n"
        )
        
        display_log_entry = f"Q: {question}\nA: {self.format_response(response)}\n\n"
        
        self.conversation_log.append(display_log_entry)
        st.session_state.conversation_log = self.conversation_log
        
        with open("chatbot_responses.txt", "a", encoding="utf-8") as f:
            f.write(file_log_entry)


def main():
    st.set_page_config(page_title="Movie Expert Bot", page_icon="🎥")
    st.title("🎥 Movie Expert Bot")
    st.write("Ask about your favourite movies! Type your question below.")

    #initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = MovieChatbot()
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {str(e)}")
            return

    #conversation history in sidebar
    with st.sidebar:
        st.subheader("Conversation History")
        if st.session_state.get('conversation_log'):
            for log_entry in st.session_state.conversation_log:
                st.markdown(log_entry)

    #input form
    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input("Your question:", placeholder="e.g., What is the plot of Guardians of the Galaxy?")
        submit_button = st.form_submit_button("Ask")

    if submit_button and question:
        try:
            result = st.session_state.chatbot.qa_chain.invoke({"query": question})
            response = result['result']
            formatted_response = st.session_state.chatbot.format_response(response)
            
            # Display response
            st.markdown(f"Response:\n\n{formatted_response}")
            
            # Log conversation
            st.session_state.chatbot.log_conversation(question, response)
            
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
            st.session_state.chatbot.log_conversation(question, f"Error: {str(e)}")

    if st.button("Save Conversation Log"):
        try:
            with open("chatbot_responses.txt", "w", encoding="utf-8") as f:
                f.writelines(st.session_state.get('conversation_log', []))
            st.success("Conversation saved successfully!")
        except Exception as e:
            st.error(f"Error saving log: {str(e)}")

if __name__ == "__main__":
    main()