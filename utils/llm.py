import os
import json
import requests
import logging
import streamlit as st
from typing import Dict, List, Any, Optional
from utils.prompt import AdvancedPromptManager
from utils.embedding import AdvancedEmbeddingManager
import markdown
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LLMResponseManager:
    """
    Advanced LLM response manager with proper markdown formatting,
    error handling, and multiple API support.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ""
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.prompt_manager = AdvancedPromptManager()
        self.embedding_manager = AdvancedEmbeddingManager()
        
        # Model configurations
        self.model_configs = {
            "llama-3.3-70b-versatile": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
                "description": "Llama 3.3 70B - Best for complex reasoning"
            },
            "llama-3.1-8b-instant": {
                "max_tokens": 2048,
                "temperature": 0.6,
                "top_p": 0.9,
                "description": "Llama 3.1 8B - Fast and efficient"
            },
            "mixtral-8x7b-32768": {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.95,
                "description": "Mixtral 8x7B - Good for long contexts"
            }
        }
        
        self.default_model = "llama-3.3-70b-versatile"
    
    def format_response_markdown(self, response_text: str) -> str:
        """
        Enhanced markdown formatting for LLM responses.
        """
        try:
            # Clean up the response
            formatted_text = response_text.strip()
            
            # Ensure proper spacing for headers
            formatted_text = formatted_text.replace('\n#', '\n\n#')
            formatted_text = formatted_text.replace('\n##', '\n\n##')
            formatted_text = formatted_text.replace('\n###', '\n\n###')
            
            # Ensure proper spacing for lists
            formatted_text = formatted_text.replace('\n-', '\n\n-')
            formatted_text = formatted_text.replace('\n*', '\n\n*')
            formatted_text = formatted_text.replace('\n1.', '\n\n1.')
            
            # Fix double spacing issues
            while '\n\n\n' in formatted_text:
                formatted_text = formatted_text.replace('\n\n\n', '\n\n')
            
            # Add emphasis to key phrases
            key_phrases = [
                ("No relevant information available", "**No relevant information available**"),
                ("Information not found", "**Information not found**"),
                ("Page ", "**Page "),
                ("Total:", "**Total:**"),
                ("Amount:", "**Amount:**"),
                ("Date:", "**Date:**"),
                ("Name:", "**Name:**")
            ]
            
            for original, formatted in key_phrases:
                if original in formatted_text and not formatted in formatted_text:
                    formatted_text = formatted_text.replace(original, formatted)
            
            return formatted_text
            
        except Exception as e:
            logging.error(f"Error formatting markdown: {e}")
            return response_text
    
    def create_response_summary(self, response: str, query: str) -> Dict[str, Any]:
        """
        Create a structured summary of the response.
        """
        try:
            summary = {
                "has_answer": "no relevant information" not in response.lower(),
                "confidence": "high" if len(response) > 100 else "medium",
                "response_length": len(response),
                "contains_page_ref": "page" in response.lower(),
                "query_type": self._classify_query(query)
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error creating response summary: {e}")
            return {}
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the type of query for better response handling.
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "who", "where", "when", "which"]):
            return "factual"
        elif any(word in query_lower for word in ["how", "why"]):
            return "explanatory" 
        elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
            return "summary"
        elif any(word in query_lower for word in ["list", "enumerate", "show all"]):
            return "listing"
        elif "?" in query:
            return "question"
        else:
            return "general"
    
    def query_model(self, 
                   query: str, 
                   faiss_folder: str, 
                   metadata_file: str,
                   model_name: str = None,
                   prompt_type: str = "base",
                   use_mmr: bool = True,
                   top_k: int = 5,
                   chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced query processing with advanced prompting and response formatting.
        """
        try:
            # Use default model if not specified
            if not model_name:
                model_name = self.default_model
            
            # Retrieve relevant chunks with advanced techniques
            with st.spinner("🔍 Searching relevant information..."):
                retrieved_data = self.embedding_manager.retrieve_relevant_chunks(
                    query, faiss_folder, top_k=top_k, use_mmr=use_mmr
                )
            
            if not retrieved_data or not retrieved_data[0].get("text"):
                return {
                    "response": "**No relevant information found in the document.**\n\nPlease try rephrasing your question or check if the document was processed correctly.",
                    "page": None,
                    "confidence": "low",
                    "sources": [],
                    "processing_time": 0
                }
            
            # Construct context with source attribution
            context_parts = []
            sources = []
            
            for i, data in enumerate(retrieved_data):
                page_info = f"Page {data.get('page', 'Unknown')}"
                source_file = data.get('source_file', 'Unknown')
                text = data.get('text', '').strip()
                
                if text:
                    context_parts.append(f"**Source {i+1} - {page_info}:**\n{text}")
                    sources.append({
                        "page": data.get('page'),
                        "file": source_file,
                        "similarity": data.get('similarity_score', 0)
                    })
            
            context = "\n\n".join(context_parts)
            
            # Determine document type for role-specific prompts
            document_type = self._detect_document_type(context)
            
            # Create appropriate prompt
            with st.spinner("🧠 Generating response..."):
                start_time = time.time()
                
                prompt = self.prompt_manager.get_prompt(
                    context=context,
                    query=query,
                    prompt_type=prompt_type,
                    document_type=document_type,
                    chat_history=chat_history,
                    use_cot=True
                )
                
                # Validate prompt length
                prompt = self.prompt_manager.validate_prompt_length(prompt)
                
                # Make API call
                response = self._make_api_call(prompt, model_name)
                
                processing_time = time.time() - start_time
            
            if response.get("success"):
                # Format the response
                formatted_response = self.format_response_markdown(response["content"])
                
                # Create response summary
                summary = self.create_response_summary(formatted_response, query)
                
                # Get primary source page
                primary_page = retrieved_data[0].get("page") if retrieved_data else None
                
                return {
                    "response": formatted_response,
                    "page": primary_page,
                    "sources": sources,
                    "confidence": summary.get("confidence", "medium"),
                    "query_type": summary.get("query_type", "general"),
                    "processing_time": round(processing_time, 2),
                    "model_used": model_name,
                    "success": True
                }
            else:
                return {
                    "response": f"**Error processing your query:**\n\n{response.get('error', 'Unknown error occurred')}",
                    "page": None,
                    "sources": sources,
                    "confidence": "low",
                    "processing_time": round(time.time() - start_time, 2),
                    "success": False
                }
                
        except Exception as e:
            logging.error(f"Error in query processing: {e}")
            return {
                "response": f"**An error occurred while processing your query:**\n\n{str(e)}\n\nPlease try again or contact support if the issue persists.",
                "page": None,
                "sources": [],
                "confidence": "low",
                "processing_time": 0,
                "success": False
            }
    
    def _detect_document_type(self, context: str) -> str:
        """
        Detect document type from context for role-specific prompting.
        """
        context_lower = context.lower()
        
        # Financial documents
        if any(term in context_lower for term in ["invoice", "total", "amount", "payment", "bill", "$"]):
            return "invoice"
        
        # Legal documents
        elif any(term in context_lower for term in ["agreement", "contract", "terms", "conditions", "party"]):
            return "contract"
        
        # Reports
        elif any(term in context_lower for term in ["report", "analysis", "summary", "findings", "conclusion"]):
            return "report"
        
        # Letters/Correspondence
        elif any(term in context_lower for term in ["dear", "sincerely", "regards", "letter", "correspondence"]):
            return "letter"
        
        # Forms
        elif any(term in context_lower for term in ["form", "application", "field", "checkbox", "signature"]):
            return "form"
        
        else:
            return "general"
    
    def _make_api_call(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """
        Make API call to the LLM service with proper error handling.
        """
        try:
            config = self.model_configs.get(model_name, self.model_configs[self.default_model])
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_name,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "top_p": config["top_p"],
                "stream": False
            }
            
            # Make the request with timeout
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if not content:
                    return {"success": False, "error": "Empty response from model"}
                
                return {
                    "success": True,
                    "content": content,
                    "model": model_name,
                    "usage": response_data.get("usage", {})
                }
            else:
                error_msg = f"API call failed with status {response.status_code}: {response.text}"
                logging.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out. Please try again."}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection error. Please check your internet connection."}
        except Exception as e:
            error_msg = f"Unexpected error during API call: {str(e)}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Get list of available models with descriptions.
        """
        return {name: config["description"] for name, config in self.model_configs.items()}
    
    def stream_response(self, prompt: str, model_name: str = None):
        """
        Stream response for real-time display (if supported by API).
        """
        # Note: Groq API streaming implementation would go here
        # For now, return regular response
        return self._make_api_call(prompt, model_name or self.default_model)
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity to suggest optimal processing parameters.
        """
        try:
            complexity_score = 0
            suggestions = {}
            
            # Length-based complexity
            if len(query) > 100:
                complexity_score += 1
                
            # Question word complexity
            complex_words = ["analyze", "compare", "explain", "detail", "comprehensive", "thoroughly"]
            if any(word in query.lower() for word in complex_words):
                complexity_score += 2
                suggestions["prompt_type"] = "role_specific"
                
            # Multiple questions
            if query.count("?") > 1:
                complexity_score += 1
                suggestions["top_k"] = 7
                
            # Technical terms
            technical_terms = ["algorithm", "analysis", "methodology", "framework", "structure"]
            if any(term in query.lower() for term in technical_terms):
                complexity_score += 1
                suggestions["use_mmr"] = True
                
            return {
                "complexity_score": complexity_score,
                "complexity_level": "high" if complexity_score >= 3 else "medium" if complexity_score >= 1 else "low",
                "suggestions": suggestions
            }
            
        except Exception as e:
            logging.error(f"Error analyzing query complexity: {e}")
            return {"complexity_score": 1, "complexity_level": "medium", "suggestions": {}}
