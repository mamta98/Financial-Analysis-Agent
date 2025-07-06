# Financial-Analysis-Agent

A Python script that provides a financial analysis agent capable of answering queries about Google, Microsoft, and Nvidia using their annual reports stored in a vector database.

## Features

- **Hybrid Search**: Uses OpenAI embeddings and keyword search for intelligent document retrieval
- **Vector Storage**: ChromaDB for efficient document storage and retrieval
- **Function Calling**: GPT-4o with structured function calling for precise information extraction
- **Hybrid PDF Parsing**: Combines OCR text and GPT-4o vision summaries for comprehensive text extraction
- **Structured Output**: Returns answers with reasoning and source citations

## Prerequisites

- Python 3.10
- OpenAI API key
- 10-K fillings as PDF documents

## Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Basic Usage

```python
from financial_assistant import FinancialAgent

# Initialize the agent
agent = FinancialAgent()

# Query the agent
response = agent.query("Which company had the highest operating margin in 2023?")
print(response)
```

### Command Line Usage

```bash
python financial_assistant.py
```

### Extract the rich text from the PDF documents
```
run text_extraction.ipynb notebook 
```


### Setting Up Vector Store

If you need to set up the vector store with new data:

```python
agent = FinancialAgent()
agent.setup_vector_store("path/to/parsed_pdf_docs.json")
```

## Project Structure

```
project/
├── data/                          # Original PDF annual reports for Google, Microsoft, and Nvidia (2022-2024)
├── data_processed/                # Processed and parsed document data in JSON format
├── downloads/                     # Downloaded HTML and PDF versions of annual reports
├── vector_store/                  # ChromaDB vector database for document embeddings and retrieval
├── agent.ipynb                    # Jupyter notebook for interactive agent testing and development
├── demo.ipynb                     # Demonstration notebook showcasing agent capabilities
├── documents_download.ipynb       # Notebook for downloading annual reports from SEC
├── text_extraction.ipynb          # Notebook for extracting and processing text from PDF documents
├── financial_assistant.py         # Main Python script containing the FinancialAgent class
├── prompts.py                     # Collection of prompts used by the agent for different query types
├── requirements.txt               # Python dependencies and package versions
├── test_queries_responses.json    # Sample queries and their structured responses for testing
├── assignment-v2_small.pdf        # Project assignment document with requirements
├── Financial_Analysis_Agent_Design_Doc.pdf  # Design documentation for the financial analysis agent
├── demo_recording.mov             # Video demonstration of the agent in action
└── README.md                      # This documentation file
```

## Supported Companies and Years

- **Companies**: google, microsoft, nvidia
- **Years**: 2022, 2023, 2024

## Output Format

The agent returns structured responses in JSON format:

```json
{
  "answer": "Direct answer to the query",
  "reasoning": "Explanation of how the answer was derived",
  "sub_queries": ["List of subqueries used for retrieval"],
  "sources": [
    {
      "company": "google",
      "year": "2023",
      "page": 42,
      "excerpt": "Relevant excerpt from the document"
    }
  ],
}
```

## Test Queries and Responses

All responses to test queries are available in the `test_queries_responses.json` file, which contains sample queries and their corresponding structured responses for reference and testing purposes.

### Sample Responses

**Query 1**: "What was NVIDIA's total revenue in fiscal year 2024?"
```json
{
  "query": "What was NVIDIA's total revenue in fiscal year 2024?",
  "answer": "NVIDIA's total revenue for fiscal year 2024 was $60,922 million.",
  "reasoning": "The retrieved financial documents explicitly state that NVIDIA's total revenue for the fiscal year ending January 28, 2024, was $60,922 million. This figure is consistent across multiple sections of the report.",
  "sub_queries": ["nvidia total revenue 2024"],
  "sources": [
    {
      "company": "nvidia",
      "year": "2024",
      "page": 91,
      "excerpt": "Revenue by geographic areas was as follows: Total revenue Jan 28, 2024: $60,922 million."
    }
  ]
}
```

**Query 2**: "How much did Microsoft's cloud revenue grow from 2022 to 2023?"
```json
{
  "query": "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
  "answer": "Microsoft's cloud revenue grew from $91.4 billion in 2022 to $111.6 billion in 2023, representing a growth of $20.2 billion or approximately 22.1%.",
  "reasoning": "The retrieved documents indicate that Microsoft's cloud revenue, which includes Azure, Office 365 Commercial, LinkedIn, Dynamics 365, and other cloud properties, increased from $91.4 billion in 2022 to $111.6 billion in 2023. This growth reflects the company's strong performance in its cloud services segment.",
  "sub_queries": [
    "microsoft cloud revenue growth or total cloud revenue 2022",
    "microsoft cloud revenue growth or total cloud revenue 2023"
  ],
  "sources": [
    {
      "company": "microsoft",
      "year": "2023",
      "page": 101,
      "excerpt": "Microsoft Cloud revenue, which includes Azure and other cloud services, Office 365 Commercial, the commercial portion of LinkedIn, Dynamics 365, and other commercial cloud properties, was $111.6 billion, $91.4 billion, and $69.1 billion in fiscal years 2023, 2022, and 2021, respectively."
    }
  ]
}
```

**Query 3**: "Compare the R&D spending as a percentage of revenue across all three companies in 2023"
```json
{
  "query": "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
  "answer": "In 2023, R&D spending as a percentage of revenue was 15% for Google, 13% for Microsoft, and 27.2% for Nvidia.",
  "reasoning": "The retrieved data shows that Google allocated 15% of its revenue to R&D, Microsoft allocated 13%, and Nvidia allocated 27.2%. These figures are directly stated in the respective financial reports for 2023.",
  "sub_queries": [
    "google R&D spending as a percentage of revenue 2023",
    "microsoft R&D spending as a percentage of revenue 2023",
    "nvidia R&D spending as a percentage of revenue 2023"
  ],
  "sources": [
    {
      "company": "google",
      "year": "2023",
      "page": 41,
      "excerpt": "Research and development expenses as a percentage of revenues: 2023 - 15%."
    },
    {
      "company": "microsoft",
      "year": "2023",
      "page": 49,
      "excerpt": "In 2023, research and development expenses amounted to $27,195 million, representing 13% of revenue."
    },
    {
      "company": "nvidia",
      "year": "2023",
      "page": 50,
      "excerpt": "In fiscal year 2023, these expenses amounted to $7,339 million, representing 27.2% of revenue."
    }
  ]
}
```
