import os
from pathlib import Path
from typing import List, Tuple
import gradio as gr
import chromadb
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentQASystem:
    """A universal document Q&A system using RAG architecture."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        n_results: int = 3,
        collection_name: str = "document_qa"
    ):
        """
        Initialize the Q&A system.
        
        Args:
            api_key: OpenAI API key
            model_name: OpenAI model to use
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            n_results: Number of relevant chunks to retrieve
            collection_name: Name for the ChromaDB collection
        """
        os.environ["OPENAI_API_KEY"] = api_key
        set_llm_cache(InMemoryCache())
        
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.n_results = n_results
        self.collection_name = collection_name
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        
        # Initialize ChromaDB
        self.client = chromadb.Client()
        self.collection = None
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful document assistant. Use the following context from the document to answer the question.
If the answer is not found in the context, say "I could not find this information in the provided document."

Context:
{context}

Question: {question}

Answer clearly and concisely:"""
        )
    
    def load_document(self, file_path: str) -> None:
        """
        Load and process a document into the vector store.
        
        Args:
            file_path: Path to the document file (txt, md, etc.)
        """
        print(f"Loading document from {file_path}...")
        
        # Read document
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        print(f"Document split into {len(chunks)} chunks.")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        
        # Clear existing data if any
        if self.collection.count() > 0:
            print("Clearing existing collection...")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
        
        # Add chunks to vector store
        print("Adding chunks to vector store...")
        doc_name = Path(file_path).name
        self.collection.add(
            documents=chunks,
            metadatas=[
                {"source": doc_name, "chunk_index": i} 
                for i in range(len(chunks))
            ],
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        print(f"Successfully loaded {len(chunks)} chunks.")
    
    def load_multiple_documents(self, file_paths: List[str]) -> None:
        """
        Load multiple documents into the vector store.
        
        Args:
            file_paths: List of paths to document files
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []
        chunk_counter = 0
        
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            chunks = self.text_splitter.split_text(text)
            doc_name = Path(file_path).name
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": doc_name,
                    "chunk_index": i
                })
                all_ids.append(f"chunk_{chunk_counter}")
                chunk_counter += 1
        
        # Create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        
        if self.collection.count() > 0:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
        
        # Add all chunks
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"Successfully loaded {len(all_chunks)} chunks from {len(file_paths)} documents.")
    
    def retrieve_and_answer(self, question: str) -> Tuple[str, List[str]]:
        """
        Retrieve relevant context and generate an answer.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (answer, retrieved_documents)
        """
        if not self.collection:
            return "Please load a document first.", []
        
        # Retrieve relevant chunks
        results = self.collection.query(
            query_texts=[question],
            n_results=self.n_results
        )
        
        retrieved_docs = results["documents"][0]
        context = "\n\n---\n\n".join(retrieved_docs)
        
        # Generate answer
        formatted_prompt = self.prompt_template.format(
            context=context, 
            question=question
        )
        response = self.llm.invoke(formatted_prompt)
        
        return response.content, retrieved_docs
    
    def launch_ui(self, share: bool = False) -> None:
        """
        Launch Gradio interface.
        
        Args:
            share: Whether to create a public link
        """
        def ask_question(question, history):
            if not question.strip():
                return "Please enter a question about the document."
            
            if not self.collection:
                return "No document loaded. Please load a document first."
            
            answer, sources = self.retrieve_and_answer(question)
            
            if sources:
                answer += "\n\n--- Retrieved Sources ---"
                for i, doc in enumerate(sources, 1):
                    snippet = doc[:200].replace("\n", " ")
                    answer += f"\n[{i}] {snippet}..."
            
            return answer
        
        demo = gr.ChatInterface(
            fn=ask_question,
            title="Document Q&A Assistant",
            description="Ask any question about your uploaded document(s).",
            examples=[
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What are the important details mentioned?",
            ],
        )
        
        demo.launch(share=share)


def main():
    """Example usage of the DocumentQASystem."""
    # Load API key
    with open("apikey.txt", "r") as file:
        api_key = file.read().strip()
    
    # Initialize system
    qa_system = DocumentQASystem(
        api_key=api_key,
        model_name="gpt-4o-mini",
        chunk_size=1000,
        chunk_overlap=200,
        n_results=3
    )
    
    # Load document(s)
    # Single document:
    qa_system.load_document("hw1-4-insurance_contract.txt")
    
    # Or load multiple documents:
    # qa_system.load_multiple_documents([
    #     "document1.txt",
    #     "document2.txt",
    #     "document3.txt"
    # ])
    
    # Launch UI
    qa_system.launch_ui(share=False)


if __name__ == "__main__":
    main()