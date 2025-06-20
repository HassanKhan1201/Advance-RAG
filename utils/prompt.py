import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AdvancedPromptManager:
    """
    Advanced prompt management system with multiple prompting techniques:
    - Chain of Thought (CoT)
    - Zero-shot prompting
    - Few-shot prompting
    - Role-based prompting
    - Retrieval-Augmented Generation (RAG) specific prompts
    """
    
    def __init__(self):
        self.base_system_role = """You are an advanced AI document analysis assistant with expertise in information extraction and comprehension. Your primary role is to help users understand and extract information from their uploaded documents (PDF, images, Word files) using optical character recognition (OCR) data."""
        
        self.few_shot_examples = [
            {
                "query": "What is the total amount mentioned in the document?",
                "context": "Invoice #12345\nDate: 2024-01-15\nSubtotal: $450.00\nTax: $45.00\nTotal: $495.00",
                "response": "According to the document, the total amount is $495.00. This includes a subtotal of $450.00 plus $45.00 in tax."
            },
            {
                "query": "Who is the recipient of this letter?",
                "context": "Dear Mr. John Smith,\n\nWe are pleased to inform you about your application status...",
                "response": "The recipient of this letter is Mr. John Smith, as indicated in the salutation 'Dear Mr. John Smith'."
            },
            {
                "query": "What is the main subject of this document?",
                "context": "ANNUAL PERFORMANCE REVIEW\nEmployee: Sarah Johnson\nReview Period: January 2024 - December 2024\nOverall Rating: Excellent",
                "response": "The main subject of this document is an annual performance review for employee Sarah Johnson, covering the period from January 2024 to December 2024, with an overall rating of Excellent."
            }
        ]
    
    def create_base_prompt(self, context: str, query: str, use_cot: bool = True) -> str:
        """
        Create base RAG prompt with Chain of Thought reasoning.
        """
        
        cot_instruction = """
        
**REASONING APPROACH (Chain of Thought):**
Before providing your final answer, think through this step-by-step:
1. **Document Analysis**: What type of document is this and what information does it contain?
2. **Query Understanding**: What specific information is the user asking for?
3. **Information Location**: Where in the document can I find relevant information?
4. **Information Extraction**: What exact details answer the user's question?
5. **Verification**: Does this information directly and completely answer the query?
6. **Response Formulation**: How can I present this information clearly and accurately?
        """ if use_cot else ""
        
        prompt = f"""
{self.base_system_role}

**CORE PRINCIPLES:**
- **DOCUMENT-ONLY RESPONSES**: Only provide information explicitly found in the provided document content
- **NO EXTERNAL KNOWLEDGE**: Do not add information from your general knowledge base
- **ACCURACY FIRST**: If information is unclear or missing, state this explicitly
- **CITE SOURCES**: Reference specific pages or sections when possible
- **COMPREHENSIVE ANALYSIS**: Thoroughly examine all provided content before responding

{cot_instruction}

**RESPONSE GUIDELINES:**
1. **Primary Response**: Answer the user's question using ONLY the document content
2. **Source Attribution**: Mention the page number or section where information was found
3. **Confidence Level**: Indicate if information is explicit, implied, or uncertain
4. **Missing Information**: Clearly state if requested information is not found in the document
5. **Formatting**: Use clear, professional language with proper markdown formatting

**DOCUMENT CONTENT:**
```
{context}
```

**USER QUERY:**
"{query}"

**INSTRUCTIONS:**
Analyze the document content carefully and provide a comprehensive answer to the user's query. Remember to:
- Only use information from the provided document
- Be specific about where information was found (page numbers, sections)
- If the information is not in the document, clearly state "This information is not available in the provided document"
- Use proper markdown formatting for better readability
- Provide a structured, professional response

**YOUR RESPONSE:**
"""
        return prompt
    
    def create_few_shot_prompt(self, context: str, query: str) -> str:
        """
        Create few-shot prompt with examples.
        """
        
        examples_text = "\n\n".join([
            f"**Example {i+1}:**\n**Query:** {ex['query']}\n**Document Context:** {ex['context']}\n**Response:** {ex['response']}"
            for i, ex in enumerate(self.few_shot_examples)
        ])
        
        prompt = f"""
{self.base_system_role}

**LEARNING FROM EXAMPLES:**
Here are examples of how to properly analyze documents and respond to queries:

{examples_text}

**KEY PATTERNS TO FOLLOW:**
- Extract specific information mentioned in the document
- Provide context and location when possible
- Be precise and factual
- Acknowledge when information is not available

**CURRENT TASK:**

**DOCUMENT CONTENT:**
```
{context}
```

**USER QUERY:**
"{query}"

**YOUR RESPONSE (following the example patterns):**
"""
        return prompt
    
    def create_zero_shot_prompt(self, context: str, query: str) -> str:
        """
        Create zero-shot prompt for direct inference.
        """
        
        prompt = f"""
{self.base_system_role}

**TASK:** Document Information Extraction and Analysis

**INSTRUCTIONS:**
You will be provided with extracted text from a document (PDF, image, or Word file) and a specific query. Your task is to:

1. Analyze the document content thoroughly
2. Extract information that directly answers the user's query
3. Provide accurate, document-based responses only
4. Format your response professionally using markdown

**IMPORTANT CONSTRAINTS:**
- ONLY use information present in the provided document
- Do NOT add external knowledge or assumptions
- If information is not in the document, explicitly state this
- Include page references when available
- Maintain professional tone and clarity

**DOCUMENT CONTENT:**
```
{context}
```

**USER QUERY:**
"{query}"

**ANALYSIS AND RESPONSE:**
"""
        return prompt
    
    def create_role_specific_prompt(self, context: str, query: str, document_type: str = "general") -> str:
        """
        Create role-specific prompts based on document type.
        """
        
        role_mappings = {
            "invoice": "financial document analyst specializing in billing and payment information",
            "contract": "legal document analyst specializing in agreements and contractual terms",
            "report": "business analyst specializing in reports and data interpretation",
            "letter": "correspondence analyst specializing in formal communications",
            "form": "form processing specialist focusing on structured data extraction",
            "general": "document analysis specialist with expertise across various document types"
        }
        
        specific_role = role_mappings.get(document_type.lower(), role_mappings["general"])
        
        prompt = f"""
You are a professional {specific_role}. Your expertise includes understanding document structure, extracting key information, and providing accurate analysis based solely on document content.

**YOUR SPECIALIZED SKILLS:**
- Deep understanding of {document_type} document formats and conventions
- Ability to identify and extract critical information accurately
- Experience in analyzing document context and relationships
- Proficiency in presenting findings in clear, structured formats

**ANALYSIS FRAMEWORK:**
1. **Document Structure Recognition**: Identify the type and format of the provided content
2. **Information Hierarchy**: Understand the importance and relationship of different data points
3. **Precision Extraction**: Extract exactly what is asked without interpretation or addition
4. **Professional Presentation**: Format findings in a clear, professional manner

**DOCUMENT CONTENT TO ANALYZE:**
```
{context}
```

**SPECIFIC QUERY:**
"{query}"

**YOUR EXPERT ANALYSIS:**
Provide a thorough, professional analysis focusing on the specific query while leveraging your expertise in {document_type} documents.
"""
        return prompt
    
    def create_conversational_prompt(self, context: str, query: str, chat_history: List[Dict] = None) -> str:
        """
        Create conversational prompt that considers chat history.
        """
        
        history_context = ""
        if chat_history:
            history_context = "\n**PREVIOUS CONVERSATION:**\n"
            for i, exchange in enumerate(chat_history[-3:]):  # Last 3 exchanges
                history_context += f"Q{i+1}: {exchange.get('query', '')}\nA{i+1}: {exchange.get('response', '')}\n\n"
        
        prompt = f"""
{self.base_system_role}

You are engaged in a conversational analysis of a document. The user may ask follow-up questions or request clarification based on previous responses.

{history_context}

**CONVERSATIONAL GUIDELINES:**
- Reference previous responses when relevant
- Maintain consistency with earlier answers
- Clarify or expand on previous responses if asked
- Build upon the conversational context while staying document-focused

**CURRENT DOCUMENT CONTENT:**
```
{context}
```

**CURRENT QUERY:**
"{query}"

**CONTEXTUAL RESPONSE:**
"""
        return prompt
    
    def create_structured_extraction_prompt(self, context: str, query: str, output_format: str = "markdown") -> str:
        """
        Create prompt for structured information extraction.
        """
        
        format_instructions = {
            "markdown": "Use markdown formatting with headers, lists, and emphasis",
            "json": "Structure your response as a JSON object with relevant fields",
            "table": "Present information in table format when appropriate",
            "list": "Use bullet points or numbered lists for structured information"
        }
        
        format_instruction = format_instructions.get(output_format, format_instructions["markdown"])
        
        prompt = f"""
{self.base_system_role}

**STRUCTURED EXTRACTION TASK:**
Extract and organize information from the document in a structured format that best answers the user's query.

**OUTPUT FORMAT REQUIREMENTS:**
{format_instruction}

**EXTRACTION PRINCIPLES:**
- Organize information logically and hierarchically
- Use appropriate headings and subheadings
- Highlight key information clearly
- Maintain document accuracy while improving readability
- Include source references (page numbers) where applicable

**DOCUMENT CONTENT:**
```
{context}
```

**EXTRACTION QUERY:**
"{query}"

**STRUCTURED RESPONSE:**
"""
        return prompt
    
    def create_summary_prompt(self, context: str, query: str = None) -> str:
        """
        Create prompt for document summarization.
        """
        
        query_focus = f"\n**SPECIFIC FOCUS:** {query}" if query else ""
        
        prompt = f"""
{self.base_system_role}

**SUMMARIZATION TASK:**
Create a comprehensive yet concise summary of the document content. Focus on key information, main points, and essential details.

**SUMMARIZATION GUIDELINES:**
1. **Identify Key Information**: Extract the most important points from the document
2. **Maintain Accuracy**: Ensure all summarized information is factually correct
3. **Logical Organization**: Structure the summary in a logical, easy-to-follow manner
4. **Appropriate Length**: Provide sufficient detail while being concise
5. **Source Attribution**: Include page references for major points{query_focus}

**DOCUMENT CONTENT:**
```
{context}
```

**COMPREHENSIVE SUMMARY:**
Provide a well-structured summary that captures the essential information from the document.
"""
        return prompt
    
    def get_prompt(self, 
                   context: str, 
                   query: str, 
                   prompt_type: str = "base",
                   document_type: str = "general",
                   chat_history: List[Dict] = None,
                   output_format: str = "markdown",
                   use_cot: bool = True) -> str:
        """
        Main method to get appropriate prompt based on requirements.
        
        Args:
            context: Document content
            query: User query
            prompt_type: Type of prompt (base, few_shot, zero_shot, role_specific, conversational, structured, summary)
            document_type: Type of document for role-specific prompts
            chat_history: Previous conversation for context
            output_format: Desired output format
            use_cot: Whether to use Chain of Thought reasoning
        """
        
        try:
            if prompt_type == "few_shot":
                return self.create_few_shot_prompt(context, query)
            elif prompt_type == "zero_shot":
                return self.create_zero_shot_prompt(context, query)
            elif prompt_type == "role_specific":
                return self.create_role_specific_prompt(context, query, document_type)
            elif prompt_type == "conversational":
                return self.create_conversational_prompt(context, query, chat_history)
            elif prompt_type == "structured":
                return self.create_structured_extraction_prompt(context, query, output_format)
            elif prompt_type == "summary":
                return self.create_summary_prompt(context, query)
            else:  # Default to base prompt
                return self.create_base_prompt(context, query, use_cot)
                
        except Exception as e:
            logging.error(f"Error creating prompt: {e}")
            # Fallback to simple prompt
            return f"""
Based on the following document content, please answer the user's query accurately:

Document Content:
{context}

User Query: {query}

Please provide a detailed response based only on the information in the document.
"""
    
    def validate_prompt_length(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Validate and truncate prompt if necessary to fit token limits.
        """
        
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = len(prompt) // 4
        
        if estimated_tokens > max_tokens:
            # Truncate context while preserving structure
            target_length = max_tokens * 4 * 0.7  # Use 70% of available space for context
            
            if "DOCUMENT CONTENT:" in prompt and "USER QUERY:" in prompt:
                parts = prompt.split("DOCUMENT CONTENT:")
                if len(parts) >= 2:
                    pre_context = parts[0]
                    post_context_parts = parts[1].split("USER QUERY:")
                    
                    if len(post_context_parts) >= 2:
                        context_section = post_context_parts[0]
                        query_section = "USER QUERY:" + post_context_parts[1]
                        
                        # Truncate context section
                        available_length = target_length - len(pre_context) - len(query_section)
                        if available_length > 0:
                            truncated_context = context_section[:int(available_length)]
                            truncated_context += "\n\n[... Content truncated for length ...]"
                            
                            prompt = pre_context + "DOCUMENT CONTENT:" + truncated_context + "\n\n" + query_section
            
            logging.warning(f"Prompt truncated due to length constraints")
        
        return prompt