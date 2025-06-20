# 🤖 Advanced RAG Document Analyzer

A sophisticated Retrieval-Augmented Generation (RAG) system built with Streamlit that enables intelligent document analysis and querying. Upload PDF, image, or Word documents and ask questions using advanced AI techniques.

## 🌟 Features

### Document Processing
- **Multi-format Support**: PDF, Images (PNG, JPG, JPEG, BMP, TIFF, GIF), Word documents (.docx, .doc)
- **Advanced OCR**: Enhanced text extraction with image preprocessing
- **Intelligent Chunking**: Smart text segmentation for optimal retrieval

### Advanced RAG Pipeline
- **Vector Database**: FAISS with HNSW indexing for fast similarity search
- **MMR (Maximal Marginal Relevance)**: Diverse and relevant result retrieval
- **Semantic Search**: Advanced embedding-based document understanding
- **LangChain Integration**: Professional-grade document processing

### AI-Powered Querying
- **Multiple Prompting Techniques**:
  - Chain of Thought (CoT) reasoning
  - Few-shot learning with examples
  - Zero-shot direct inference
  - Role-specific prompting
  - Conversational context awareness
- **Advanced LLM Support**: Multiple Groq models (Llama 3.3, Mixtral)
- **Query Complexity Analysis**: Automatic optimization based on question complexity

### Professional UI
- **Modern Interface**: Clean, responsive Streamlit design
- **Real-time Chat**: Interactive document querying with chat history
- **Analytics Dashboard**: Performance metrics and insights
- **Export Capabilities**: Download chat history and analytics reports

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR installed on your system
- Groq API key (provided in code, or set your own)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd advanced-rag-document-analyzer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR**:

**Windows**:
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH

**macOS**:
```bash
brew install tesseract
```

**Ubuntu/Debian**:
```bash
sudo apt-get install tesseract-ocr
```

4. **Run the application**:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
advanced-rag-document-analyzer/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
└── utils/                         # Core modules
    ├── processing_ocr.py          # Document processing & OCR
    ├── embedding.py               # Vector embeddings & FAISS
    ├── prompt.py                  # Advanced prompt management
    └── llm.py                     # LLM response handling
```

### Core Components

#### 🔧 `utils/processing_ocr.py`
- **DocumentProcessor**: Multi-format document processing
- **Image Preprocessing**: Contrast enhancement and sharpening
- **OCR Configuration**: Optimized Tesseract settings
- **Metadata Management**: Structured document information storage

#### 🧠 `utils/embedding.py`
- **AdvancedEmbeddingManager**: Intelligent text chunking and embedding
- **FAISS Integration**: High-performance vector search
- **MMR Support**: Diverse result retrieval
- **LangChain FAISS**: Advanced retrieval features

#### 💭 `utils/prompt.py`
- **AdvancedPromptManager**: Multiple prompting strategies
- **Chain of Thought**: Step-by-step reasoning
- **Few-shot Learning**: Example-based prompting
- **Role-specific Prompts**: Document type awareness
- **Conversational Context**: Chat history integration

#### 🤖 `utils/llm.py`
- **LLMResponseManager**: Professional response handling
- **Multiple Models**: Llama 3.3, Mixtral support
- **Markdown Formatting**: Clean response presentation
- **Error Handling**: Robust API interaction
- **Performance Analytics**: Response time and confidence tracking

## 🎯 Usage Guide

### 1. Document Upload
1. Navigate to the "📄 Document Upload" tab
2. Upload your document (PDF, image, or Word file)
3. Click "🚀 Process Document" to extract text and create embeddings
4. Wait for processing to complete (status indicators will guide you)

### 2. Intelligent Querying
1. Switch to the "💬 Chat Interface" tab
2. Use quick suggestions or type your own questions
3. The system automatically optimizes processing based on query complexity
4. View responses with source attribution and confidence scores

### 3. Analytics & Insights
1. Visit the "📈 Analytics" tab to view:
   - Chat statistics and metrics
   - Response time trends
   - Confidence distribution
   - Most referenced document pages
2. Export detailed reports for further analysis

## ⚙️ Configuration

### Model Settings
- **Default Model**: Llama 3.3 70B Versatile
- **Available Models**: 
  - `llama-3.3-70b-versatile`: Best for complex reasoning
  - `llama-3.1-8b-instant`: Fast and efficient
  - `mixtral-8x7b-32768`: Good for long contexts

### Advanced Settings
- **Prompt Strategy**: Choose from base, few-shot, zero-shot, role-specific, or conversational
- **MMR**: Enable/disable Maximal Marginal Relevance for diverse results
- **Retrieval Count**: Adjust number of relevant chunks (3-10)

### API Configuration
To use your own Groq API key:
1. Get an API key from [Groq](https://console.groq.com/)
2. Replace the API key in `utils/llm.py`:
```python
self.api_key = "your-api-key-here"
```

## 🔧 Technical Details

### Document Processing Pipeline
1. **File Detection**: Automatic format recognition
2. **Content Extraction**: 
   - PDF → Images → OCR
   - Images → Direct OCR with preprocessing
   - Word → Text extraction from paragraphs and tables
3. **Text Chunking**: Recursive character splitting with overlap
4. **Embedding Generation**: SentenceTransformers with HuggingFace models
5. **Vector Storage**: FAISS with HNSW indexing

### Retrieval Strategy
1. **Query Analysis**: Complexity assessment and parameter optimization
2. **Embedding Search**: Semantic similarity in vector space
3. **MMR Application**: Diversity-aware result selection
4. **Context Assembly**: Source attribution and metadata inclusion

### Response Generation
1. **Prompt Construction**: Dynamic prompt based on query type and context
2. **LLM Processing**: API call with error handling and retries
3. **Response Formatting**: Markdown enhancement and structure
4. **Metadata Extraction**: Confidence scoring and source tracking

## 📊 Performance Optimization

### Embedding Efficiency
- **Batch Processing**: 32-document batches for embedding generation
- **Caching**: Streamlit resource caching for model initialization
- **HNSW Indexing**: Fast approximate nearest neighbor search

### Query Optimization
- **Complexity Analysis**: Automatic parameter tuning based on query difficulty
- **Context Limitation**: Smart truncation to fit model token limits
- **MMR Configuration**: Balanced relevance vs. diversity (λ = 0.7)

## 🐛 Troubleshooting

### Common Issues

**Tesseract not found**:
```bash
# Install Tesseract and ensure it's in PATH
which tesseract  # Should return path
```

**PIL/Pillow import errors**:
```bash
pip uninstall Pillow
pip install --no-cache-dir Pillow
```

**FAISS installation issues**:
```bash
pip install faiss-cpu  # Use CPU version for compatibility
```

**Memory issues with large documents**:
- Reduce chunk size in `embedding.py`
- Lower the `top_k` retrieval parameter
- Use smaller embedding models

### Performance Tips
1. **Optimize Images**: Compress large image files before upload
2. **Batch Questions**: Process multiple related queries together
3. **Clear History**: Regularly clear chat history to improve performance
4. **Model Selection**: Use faster models for simple queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the excellent web framework
- **LangChain**: For advanced document processing capabilities
- **FAISS**: For high-performance vector search
- **Groq**: For fast LLM inference
- **HuggingFace**: For pre-trained embedding models
- **Tesseract**: For robust OCR capabilities

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Provide error logs and system information

---

**Built with ❤️ using Python, Streamlit, and modern AI techniques**