# Financial Data RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system specifically designed for financial document analysis. This project combines advanced PDF processing, table extraction, and hybrid search capabilities to provide accurate answers to financial queries.

## 🌟 Features

### Step 1: Basic RAG Pipeline
- **PDF Text Extraction**: Extract and clean text from financial documents
- **Semantic Chunking**: Intelligent text segmentation for optimal retrieval
- **Vector Embeddings**: Using SentenceTransformers for semantic similarity
- **Vector Search**: FAISS-powered similarity search for relevant context
- **Answer Generation**: FLAN-T5 model for generating contextual responses

### Step 2: Structured Data Integration
- **Multi-Method Table Extraction**: PDFPlumber, Camelot, and Tabula integration
- **Financial Table Detection**: Automatic identification of financial data tables
- **Hybrid Retrieval**: Combines vector search with structured data queries
- **Numerical Analysis**: Advanced handling of financial comparisons and calculations
- **Temporal Comparisons**: Quarter-over-quarter and year-over-year analysis

## 🚀 Key Capabilities

- **Complex Financial Queries**: Handle questions about revenue, expenses, profits, and growth
- **Comparative Analysis**: Compare metrics across different time periods
- **Hybrid Search**: Leverage both unstructured text and structured table data
- **Clean API**: Simple methods for getting just answers without verbose output
- **Multi-Format Support**: Extract data from various table formats in PDFs

## 📋 Requirements

### Python Dependencies
```
torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
pandas>=1.5.0
numpy>=1.21.0
PyPDF2>=3.0.0
pdfplumber>=0.7.0
tabula-py>=2.5.0
camelot-py[cv]>=0.10.0
nltk>=3.7
scikit-learn>=1.1.0
tqdm>=4.64.0
```

### System Requirements
- Python 3.8+
- Java (required for Tabula)
- OpenCV (for Camelot table extraction)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/codenim34/Financial-Data-RAG-.git
   cd Financial-Data-RAG-
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch transformers sentence-transformers
   pip install faiss-cpu pandas numpy
   pip install PyPDF2 pdfplumber tabula-py
   pip install "camelot-py[cv]" nltk scikit-learn tqdm
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   ```

## 📁 Project Structure

```
Financial-Data-RAG-/
├── data/
│   └── Meta's Q1 2024 Financial Report.pdf
├── notebooks/
│   ├── step_1.ipynb                 # Basic RAG pipeline
│   └── step_2.ipynb                 # Enhanced RAG with structured data
├── src/                             # Source code modules (future)
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## 💻 Usage

### Quick Start

1. **Load the Enhanced RAG Pipeline** (from Step 2 notebook):
   ```python
   # After running all cells in step_2.ipynb
   
   # Simple question answering
   answer = enhanced_rag.ask("What is Meta's revenue in Q1 2024?")
   print(answer)
   
   # Comparison queries
   comparison = enhanced_rag.compare(
       "How did revenue change from Q1 2023 to Q1 2024?",
       metric_type="revenue",
       period1="Q1 2024", 
       period2="Q1 2023"
   )
   print(comparison)
   ```

### API Methods

#### Simple Methods (Clean Output)
- `enhanced_rag.ask(question)` - Returns just the answer text
- `enhanced_rag.compare(question, metric_type, period1, period2)` - Returns comparison answer

#### Detailed Methods (Full Results)
- `enhanced_rag.query(question, verbose=True)` - Returns full search results and metadata
- `enhanced_rag.comparison_query(question, metric_type, period1, period2, verbose=True)` - Detailed comparison results

### Example Queries

```python
# Revenue analysis
revenue_answer = enhanced_rag.ask("What was Meta's total revenue in Q1 2024?")

# Expense breakdown
expenses_answer = enhanced_rag.ask("Summarize Meta's operating expenses in Q1 2024")

# Growth comparison
growth_comparison = enhanced_rag.compare(
    "How much did Meta's net income grow from Q1 2023 to Q1 2024?",
    metric_type="net income",
    period1="Q1 2024",
    period2="Q1 2023"
)

# Research & Development costs
rd_answer = enhanced_rag.ask("How much did Meta spend on R&D in Q1 2024?")
```

## 🔧 Technical Architecture

### Step 1: Basic RAG Pipeline
1. **Text Extraction**: PyPDF2 for raw text extraction
2. **Text Processing**: Regex-based cleaning and NLTK tokenization
3. **Chunking**: Semantic chunking with 4 sentences per chunk
4. **Embeddings**: all-MiniLM-L6-v2 model (384-dimensional vectors)
5. **Vector Store**: FAISS IndexFlatIP for efficient similarity search
6. **Generation**: FLAN-T5-small for answer generation

### Step 2: Enhanced Pipeline
1. **Table Extraction**: Multi-method approach (PDFPlumber + Camelot + Tabula)
2. **Financial Detection**: Keyword-based financial table identification
3. **Structured Storage**: Organized financial metrics database
4. **Hybrid Search**: Weighted combination of vector and structured search
5. **Enhanced Generation**: Dual-context prompts with structured data

### Search Weights
- Vector Search: 40%
- Structured Data Search: 40%
- Financial Metrics: 20%

## 📊 Performance Metrics

### Table Extraction Results
- **Extraction Methods**: 3 (PDFPlumber, Camelot, Tabula)
- **Financial Tables Found**: 7 unique tables
- **Financial Metrics Extracted**: 100+ individual metrics
- **Metric Categories**: 8 types (Revenue, Net Income, Operating Expenses, etc.)

### Query Capabilities
- ✅ Text-based financial queries
- ✅ Structured data comparisons
- ✅ Temporal analysis (Q1 2024 vs Q1 2023)
- ✅ Numerical calculations and growth rates
- ✅ Multi-metric summaries

## 🧪 Testing

The system has been tested with various query types:

1. **Revenue Queries**: "What is Meta's revenue in Q1 2024?"
2. **Comparison Queries**: "How did net income change from Q1 2023 to Q1 2024?"
3. **Expense Analysis**: "Summarize Meta's operating expenses"
4. **Growth Analysis**: "What was Meta's revenue growth rate?"

## 🔮 Future Enhancements

- [ ] Support for multiple document types (10-K, 10-Q, annual reports)
- [ ] Advanced financial calculations (ratios, margins, trends)
- [ ] Multi-company comparisons
- [ ] Interactive web interface
- [ ] API endpoint deployment
- [ ] Real-time financial data integration
- [ ] Chart and graph generation from financial data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **codenim34** - *Initial work* - [GitHub Profile](https://github.com/codenim34)

## 🙏 Acknowledgments

- OpenAI for the Transformer architecture concepts
- Hugging Face for the pre-trained models
- Facebook AI Research for FAISS
- The open-source community for the various PDF processing libraries

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/codenim34/Financial-Data-RAG-/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

## 📈 Project Status

- ✅ Step 1: Basic RAG Pipeline - **Complete**
- ✅ Step 2: Structured Data Integration - **Complete**
- 🔄 Step 3: Multi-Document Support - **Planned**
- 🔄 Step 4: Web Interface - **Planned**
- 🔄 Step 5: API Deployment - **Planned**

---

*This README provides a comprehensive overview of the Financial Data RAG Pipeline. For detailed implementation, refer to the Jupyter notebooks in the `notebooks/` directory.*