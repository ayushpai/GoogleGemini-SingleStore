import os
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.singlestoredb import SingleStoreDB
from openai import OpenAI
import google.generativeai as gemini
from dotenv import load_dotenv
import os

# Set up API keys and database URL
# Load environment variables from .env file
load_dotenv()

# Access the API key
GEMINI_API_KEY = os.getenv('API_KEY')
gemini.configure(api_key=GEMINI_API_KEY)
os.environ["SINGLESTOREDB_URL"] = "<Insert SingleStore Database URL Here>"

# Load and process documents
loader = TextLoader("superbowl.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Generate embeddings and create a document search database
embeddings = OpenAIEmbeddings()
docsearch = SingleStoreDB.from_documents(docs, embeddings, table_name="tester1")


# Chat loop
while True:
    # Get user input
    user_query = input("\nYou: ")

    # Check for exit command
    if user_query.lower() in ['quit', 'exit']:
        print("Exiting chatbot.")
        break

    # Perform similarity search
    docs = docsearch.similarity_search(user_query)
    if docs:
        context = docs[0].page_content

        model = gemini.GenerativeModel('gemini-pro')

        response = model.generate_content(user_query + context)

        # Output the response
        print("AI: ", end="")
        for chunk in response:
            print(chunk.text)
            print("_" * 80)

    else:
        print("AI: Sorry, I couldn't find relevant information.")
