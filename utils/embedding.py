import os
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AdvancedEmbeddingManager:
    """
    Enhanced embedding manager with chunking, MMR, and semantic search capabilities.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Initialize Langchain embeddings for advanced features
        self.langchain_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def chunk_text(self, metadata):
        """
        Split text into smaller chunks for better retrieval.
        """
        try:
            chunked_metadata = []
            
            for entry in metadata:
                text = entry.get("text", "").strip()
                if not text:
                    continue
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        chunked_entry = entry.copy()
                        chunked_entry["text"] = chunk.strip()
                        chunked_entry["chunk_id"] = i
                        chunked_entry["total_chunks"] = len(chunks)
                        chunked_metadata.append(chunked_entry)
            
            logging.info(f"Created {len(chunked_metadata)} chunks from {len(metadata)} documents")
            return chunked_metadata
            
        except Exception as e:
            logging.error(f"Error chunking text: {e}")
            return metadata
    
    def generate_embeddings_and_store(self, metadata, faiss_folder):
        """
        Generate embeddings with chunking and store in FAISS with enhanced features.
        """
        try:
            with st.status("Generating Embeddings...", expanded=True) as status:
                st.write("Chunking text for better retrieval...")
                
                # Ensure FAISS folder exists
                os.makedirs(faiss_folder, exist_ok=True)
                
                # Chunk text for better retrieval
                chunked_metadata = self.chunk_text(metadata)
                
                if not chunked_metadata:
                    st.error("No text found to generate embeddings")
                    return False
                
                st.write(f"Generating embeddings for {len(chunked_metadata)} chunks...")
                
                # Extract texts for embedding
                texts = [entry["text"] for entry in chunked_metadata]
                
                # Generate embeddings
                embeddings = []
                progress_bar = st.progress(0)
                
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                    embeddings.extend(batch_embeddings)
                    
                    # Update progress
                    progress = min((i + batch_size) / len(texts), 1.0)
                    progress_bar.progress(progress)
                
                st.write("Creating FAISS index...")
                
                # Convert to numpy array
                embeddings_np = np.array(embeddings, dtype=np.float32)
                
                # Create FAISS index with better similarity search
                dimension = embeddings_np.shape[1]
                
                # Use IndexHNSWFlat for better performance with similarity search
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 100
                
                # Add embeddings to index
                index.add(embeddings_np)
                
                # Save FAISS index
                index_path = os.path.join(faiss_folder, "index.bin")
                faiss.write_index(index, index_path)
                
                # Save metadata
                metadata_path = os.path.join(faiss_folder, "metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(chunked_metadata, f, indent=4, ensure_ascii=False)
                
                # Create Langchain FAISS store for MMR
                self._create_langchain_store(texts, chunked_metadata, faiss_folder)
                
                status.update(label="Embeddings generation completed!", state="complete")
                
            st.success(f"Successfully created embeddings for {len(chunked_metadata)} text chunks")
            return True
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            logging.error(f"Embedding generation error: {e}")
            return False
    
    def _create_langchain_store(self, texts, metadata, faiss_folder):
        """
        Create Langchain FAISS store for advanced retrieval features.
        """
        try:
            # Create documents with metadata
            from langchain.schema import Document
            
            documents = []
            for i, (text, meta) in enumerate(zip(texts, metadata)):
                doc = Document(
                    page_content=text,
                    metadata={
                        "page": meta.get("page", 1),
                        "source_file": meta.get("source_file", "unknown"),
                        "file_type": meta.get("file_type", "unknown"),
                        "chunk_id": meta.get("chunk_id", 0)
                    }
                )
                documents.append(doc)
            
            # Create FAISS store
            vectorstore = LangchainFAISS.from_documents(
                documents, 
                self.langchain_embeddings
            )
            
            # Save langchain store
            langchain_path = os.path.join(faiss_folder, "langchain_store")
            vectorstore.save_local(langchain_path)
            
            logging.info("Langchain FAISS store created successfully")
            
        except Exception as e:
            logging.error(f"Error creating Langchain store: {e}")
    
    def retrieve_relevant_chunks(self, query, faiss_folder, top_k=5, use_mmr=True):
        """
        Enhanced retrieval with MMR and semantic search.
        """
        try:
            # Try Langchain store first for MMR
            if use_mmr:
                try:
                    langchain_path = os.path.join(faiss_folder, "langchain_store")
                    if os.path.exists(langchain_path):
                        vectorstore = LangchainFAISS.load_local(
                            langchain_path, 
                            self.langchain_embeddings,
                            allow_dangerous_deserialization=True
                        )
                        
                        # Use MMR for diverse results
                        docs = vectorstore.max_marginal_relevance_search(
                            query, 
                            k=top_k,
                            fetch_k=top_k * 2,  # Fetch more candidates
                            lambda_mult=0.7  # Balance between relevance and diversity
                        )
                        
                        retrieved_data = []
                        for doc in docs:
                            retrieved_data.append({
                                "page": doc.metadata.get("page", 1),
                                "text": doc.page_content,
                                "source_file": doc.metadata.get("source_file", "unknown"),
                                "chunk_id": doc.metadata.get("chunk_id", 0)
                            })
                        
                        return retrieved_data
                        
                except Exception as e:
                    logging.warning(f"MMR retrieval failed, falling back to similarity: {e}")
            
            # Fallback to regular FAISS similarity search
            return self._similarity_search(query, faiss_folder, top_k)
            
        except Exception as e:
            logging.error(f"Error in retrieval: {e}")
            return [{"text": "Error retrieving relevant information.", "page": None}]
    
    def _similarity_search(self, query, faiss_folder, top_k):
        """
        Regular similarity search using FAISS.
        """
        try:
            # Load FAISS index
            index_path = os.path.join(faiss_folder, "index.bin")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found: {index_path}")
            
            index = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = os.path.join(faiss_folder, "metadata.json")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Encode query
            query_embedding = self.model.encode([query.lower()]).astype(np.float32)
            
            # Search
            distances, indices = index.search(query_embedding, top_k)
            
            # Retrieve results
            retrieved_data = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(metadata):
                    entry = metadata[idx].copy()
                    entry["similarity_score"] = float(1 / (1 + distance))  # Convert distance to similarity
                    retrieved_data.append(entry)
            
            return retrieved_data
            
        except Exception as e:
            logging.error(f"Similarity search error: {e}")
            return [{"text": "Error in similarity search.", "page": None}]
    
    def get_embedding_stats(self, faiss_folder):
        """
        Get statistics about the embedding store.
        """
        try:
            stats = {}
            
            # Index stats
            index_path = os.path.join(faiss_folder, "index.bin")
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
                stats["total_vectors"] = index.ntotal
                stats["vector_dimension"] = index.d
            
            # Metadata stats
            metadata_path = os.path.join(faiss_folder, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                stats["total_chunks"] = len(metadata)
                stats["total_pages"] = len(set(entry.get("page", 1) for entry in metadata))
                stats["file_types"] = list(set(entry.get("file_type", "unknown") for entry in metadata))
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting embedding stats: {e}")
            return {}