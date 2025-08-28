from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


class OllamaAgent:
    def __init__(self):
        print("Initializing Ollama Agent...")
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model="llama3.2:1b",
            base_url="http://localhost:11434",
            temperature=0.1
        )
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        #  vector databases check
        self._ensure_vector_dbs()
        
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
        print("Ollama Agent initialized successfully!")
    
    def _ensure_vector_dbs(self):
        """Ensure vector databases exist, create them if not"""
        os.makedirs("vector_dbs/hr_policies", exist_ok=True)
        os.makedirs("vector_dbs/tech_docs", exist_ok=True)
        os.makedirs("data/technical_guides", exist_ok=True)
        
        if not os.path.exists("vector_dbs/hr_policies/chroma.sqlite3"):
            print("Creating HR vector database...")
            self.ingest_hr_documents()
        
        if not os.path.exists("vector_dbs/tech_docs/chroma.sqlite3"):
            print("Creating Tech vector database...")
            self.ingest_tech_documents()
    
    def create_vector_store(self, documents, persist_directory, collection_name):
        """Create and persist a vector store from documents using Ollama"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        texts = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        vectorstore.persist()
        print(f"Created vector store at {persist_directory} with {len(texts)} chunks")
        return vectorstore
    
    def ingest_hr_documents(self):
        """Ingest HR policies"""
        print("Loading HR documents...")
        try:
            loader = TextLoader("data/company_handbook.txt")
            documents = loader.load()
        except FileNotFoundError:
            print("Warning: data/company_handbook.txt not found. Creating empty HR database.")
            vectorstore = Chroma(
                persist_directory="vector_dbs/hr_policies",
                embedding_function=self.embeddings,
                collection_name="hr_policies"
            )
            vectorstore.persist()
            return vectorstore
        
        return self.create_vector_store(
            documents,
            "vector_dbs/hr_policies",
            "hr_policies"
        )
    
    def ingest_tech_documents(self):
        """Ingest technical documentation"""
        print("Loading technical documents...")
        try:
            loader = DirectoryLoader("data/technical_guides/", glob="**/*.txt")
            documents = loader.load()
        except FileNotFoundError:
            print("Warning: data/technical_guides/ directory not found. Creating empty tech database.")
            vectorstore = Chroma(
                persist_directory="vector_dbs/tech_docs",
                embedding_function=self.embeddings,
                collection_name="tech_docs"
            )
            vectorstore.persist()
            return vectorstore
        
        return self.create_vector_store(
            documents,
            "vector_dbs/tech_docs", 
            "tech_docs"
        )
    
    def _hr_policy_search(self, query: str) -> str:
        """Search HR policies and return formatted string"""
        try:
            hr_vectorstore = Chroma(
                persist_directory="vector_dbs/hr_policies",
                embedding_function=self.embeddings,
                collection_name="hr_policies"
            )
            docs = hr_vectorstore.similarity_search(query, k=3)
            return self._format_documents(docs)
        except Exception as e:
            return f"Error searching HR policies: {str(e)}"
    
    def _tech_docs_search(self, query: str) -> str:
        """Search technical docs and return formatted string"""
        try:
            tech_vectorstore = Chroma(
                persist_directory="vector_dbs/tech_docs",
                embedding_function=self.embeddings,
                collection_name="tech_docs"
            )
            docs = tech_vectorstore.similarity_search(query, k=3)
            return self._format_documents(docs)
        except Exception as e:
            return f"Error searching technical docs: {str(e)}"
    
    def _format_documents(self, docs: list[Document]) -> str:
        """Format documents into a readable string"""
        if not docs:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            if len(content) > 500:
                content = content[:500] + "..."
            formatted.append(f"Document {i}:\n{content}\n")
        
        return "\n".join(formatted)
    
    def _setup_tools(self):
        """Initialize all retrieval tools"""
        print("Setting up tools...")
        
        web_search_tool = DuckDuckGoSearchRun()
        
        tools = [
            Tool(
                name="HR_Policy_Search",
                func=self._hr_policy_search,
                description="Use for HR questions: PTO, benefits, company policies, hiring, onboarding. Input: clear question."
            ),
            Tool(
                name="Technical_Docs_Search", 
                func=self._tech_docs_search,
                description="Use for technical questions: APIs, deployment, coding standards, troubleshooting. Input: clear question."
            ),
            Tool(
                name="Web_Search",
                func=web_search_tool.run,
                description="Use for current events, news, market trends, competitor info. Input: search query."
            )
        ]
        
        return tools
    
    def _create_agent(self):
        """Create the conversational agent"""
        print("Creating agent...")

        # OPENAI_FUNCTIONS for structured tool calls
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            early_stopping_method="generate"
        )
        return agent
    
    def query_agent(self, question):
        """Query the agent with a question"""
        try:
            print(f"Processing question: {question}")
            response = self.agent.invoke({"input": question})
            # Handle multiple possible return keys
            return response.get("output") or response.get("output_text") or str(response)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return error_msg


# Create global instance
agent_instance = OllamaAgent()
