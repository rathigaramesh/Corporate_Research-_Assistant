import os
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  
from langchain_chroma import Chroma 

def create_vector_store(documents, persist_directory, collection_name):
    """Create and persist a vector store from documents using Ollama"""
    print(f"Creating vector store for {collection_name}...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(texts)} chunks")
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    # Note: In Chroma >= 0.4.x, persistence is automatic - no need to call .persist()
    print(f"Created vector store at {persist_directory} with {len(texts)} chunks")
    return vectorstore

def check_ollama_connection():
    """Check if Ollama is running and the model is available"""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        # Try to generate a small embedding to test the connection
        test_embedding = embeddings.embed_query("test connection")
        print("✅ Successfully connected to Ollama")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("Please make sure Ollama is running and the model 'nomic-embed-text' is available")
        return False

def ingest_hr_documents():
    """Ingest HR policies"""
    print("\n" + "="*50)
    print("Loading HR documents...")
    
    hr_doc_path = "data/company_handbook.txt"
    if not os.path.exists(hr_doc_path):
        print(f"⚠️  HR document not found at {hr_doc_path}")
        print("Creating a sample HR document...")
        os.makedirs("data", exist_ok=True)
        with open(hr_doc_path, "w") as f:
            f.write("""
            Company HR Policies
            ===================
            
            Vacation Policy:
            All full-time employees are entitled to 15 days of paid vacation per year.
            Vacation days accrue at a rate of 1.25 days per month.
            
            Sick Leave:
            Employees receive 10 days of paid sick leave per year.
            
            Remote Work Policy:
            Employees may work remotely up to 2 days per week with manager approval.
            
            Dress Code:
            Business casual attire is required Monday through Thursday.
            Casual attire is permitted on Fridays.
            """)
        print(f"Created sample HR document at {hr_doc_path}")
    
    try:
        loader = TextLoader(hr_doc_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} HR documents")
        
        vectorstore = create_vector_store(
            documents,
            "vector_dbs/hr_policies",
            "hr_policies"
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading HR documents: {e}")
        return None

def ingest_tech_documents():
    """Ingest technical documentation"""
    print("\n" + "="*50)
    print("Loading technical documents...")
    
    tech_dir = "data/technical_guides/"
    os.makedirs(tech_dir, exist_ok=True)
    
    # Create a sample technical document if none exists
    sample_tech_file = os.path.join(tech_dir, "api_guide.txt")
    if not os.path.exists(sample_tech_file):
        print("⚠️  No technical documents found. Creating a sample...")
        with open(sample_tech_file, "w") as f:
            f.write("""
            API Documentation
            =================
            
            Authentication:
            All API requests require an API key in the header.
            Example: Authorization: Bearer YOUR_API_KEY
            
            Rate Limits:
            API requests are limited to 1000 requests per hour per API key.
            
            Error Codes:
            400 - Bad Request
            401 - Unauthorized
            404 - Resource Not Found
            500 - Internal Server Error
            """)
        print(f"Created sample technical document at {sample_tech_file}")
    
    try:
        loader = DirectoryLoader(tech_dir, glob="**/*.txt")
        documents = loader.load()
        print(f"Loaded {len(documents)} technical documents")
        
        vectorstore = create_vector_store(
            documents,
            "vector_dbs/tech_docs", 
            "tech_docs"
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading technical documents: {e}")
        return None

def main():
    """Main ingestion function"""
    print("Starting document ingestion process...")
    start_time = time.time()
    
    # Create directories if they don't exist
    os.makedirs("vector_dbs/hr_policies", exist_ok=True)
    os.makedirs("vector_dbs/tech_docs", exist_ok=True)
    os.makedirs("data/technical_guides", exist_ok=True)
    
    # Check if Ollama is available
    if not check_ollama_connection():
        return
    
    # Ingest documents
    hr_db = ingest_hr_documents()
    tech_db = ingest_tech_documents()
    
    # Calculate and display timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*50)
    if hr_db or tech_db:
        print(f"✅ Document ingestion complete! Time taken: {elapsed_time:.2f} seconds")
        print("\nSummary:")
        print(f"- HR Policies: {'✅ Loaded' if hr_db else '❌ Failed'}")
        print(f"- Technical Docs: {'✅ Loaded' if tech_db else '❌ Failed'}")
        print("\nYou can now run your Corporate Research Assistant!")
    else:
        print("❌ Document ingestion failed!")
    
    print("="*50)

if __name__ == "__main__":
    main()