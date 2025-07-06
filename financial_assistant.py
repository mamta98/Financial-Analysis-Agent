import os
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import chromadb
from prompts import FINANCIAL_ANALYST_PROMPT


class FinancialAgent:
    """Financial analysis agent for querying company annual reports."""
    
    def __init__(self, model_name: str = "gpt-4o", embed_model: str = "text-embedding-3-small"):
        """
        Initialize the financial agent.
        
        Args:
            model_name: OpenAI model name to use for chat completions
            embed_model: Embedding model to use
        """
        # Get API key from environment variable
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.embed_model = embed_model
        self.chroma_client = chromadb.PersistentClient(path='vector_store')
        self.collection = self.chroma_client.get_collection("hybrid_financial_chunks")
        
        # Define the retrieval function schema
        self.retrieval_function = {
            "name": "retrieve_financial_info",
            "description": "Retrieve relevant documents from annual reports given the company name, year, and subquery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The company name",
                        "enum": ["google", "microsoft", "nvidia"] 
                    },
                    "year": {
                        "type": "string",
                        "description": "The year of the report",
                        "enum": ["2022", "2023", "2024"]  
                    },
                    "subquery": {"type": "string"}
                },
                "required": ["company_name", "year", "subquery"]
            }
        }
        
        # System prompt for the agent
        self.system_prompt = FINANCIAL_ANALYST_PROMPT

    def hybrid_text(self, page: Dict[str, Any]) -> str:
        """
        Combine OCR text and GPT-4o vision summary for hybrid search.
        
        Args:
            page: Page data containing raw_text and img_desc
            
        Returns:
            Combined text for embedding
        """
        return f"PAGE OCR TEXT:\n{page['raw_text']}\n\nGPT-4o VISION SUMMARY:\n{page['img_desc']}"

    def setup_vector_store(self, json_file_path: str = "data_processed/parsed_pdf_docs.json"):
        """
        Set up the vector store with parsed PDF documents.
        
        Args:
            json_file_path: Path to the parsed PDF documents JSON file
        """
        print("Loading parsed PDF documents...")
        with open(json_file_path) as f:
            pages = json.load(f)
        
        print("Preparing hybrid texts...")
        texts = [self.hybrid_text(p) for p in pages]
        
        print("Preparing metadata...")
        metadatas = [
            {
                "company": p["company"],
                "year": p["year"],
                "pdf_name": p["pdf_name"],
                "page_number": p["page_number"]
            }
            for p in pages
        ]
        
        print("Generating embeddings...")
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(model=self.embed_model, input=batch)
            embeddings.extend([e.embedding for e in response.data])
            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        print("Indexing in ChromaDB...")
        coll = self.chroma_client.get_or_create_collection("hybrid_financial_chunks")
        
        for text, meta, emb in zip(texts, metadatas, embeddings):
            unique_id = str(uuid.uuid4())
            coll.add(
                ids=[unique_id],
                embeddings=[emb],
                metadatas=[meta],
                documents=[text]
            )
        
        print("Vector store setup complete!")

    def retrieve_financial_info(self, subquery: str, company_name: str, year: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant financial information from the vector store.
        
        Args:
            subquery: Search query
            company_name: Company name to filter by
            year: Year to filter by
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with metadata
        """
        subquery = subquery.lower()
        company = company_name.lower()
        
        # Get embedding for the query
        embed_resp = self.client.embeddings.create(model=self.embed_model, input=[subquery])
        query_emb = embed_resp.data[0].embedding
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_emb],  # semantic
            query_texts=[subquery],        # keyword
            n_results=top_k,
            where={
                "$and": [
                    {"company": company_name},
                    {"year": year}
                ]
            }
        )
        
        # Format results
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        outs = []
        for text, meta in zip(docs, metas):
            outs.append({
                "text": text,
                "page_number": meta.get("page_number"),
                "pdf_name": meta.get("pdf_name"),
                "company": meta.get("company"),
                "year": meta.get("year")
            })
        return outs

    def get_llm_response(self, messages: List[Dict[str, str]]) -> Any:
        """
        Get response from the LLM with function calling.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            OpenAI response object
        """
        tools = [self.retrieval_function]
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            functions=tools,
            function_call="auto",
            max_tokens=8000,
            temperature=0
        )

    def get_tool_calls(self, response: Any) -> List[Tuple]:
        """
        Extract tool calls from the LLM response.
        
        Args:
            response: OpenAI response object
            
        Returns:
            List of tool calls
        """
        tool_calls = []
        msg = response.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = [(call, 'tool') for call in msg.tool_calls]
        elif hasattr(msg, "function_call") and msg.function_call:
            tool_calls = [(msg.function_call, 'function')]
        return tool_calls

    def get_tool_response(self, tool_call: Tuple) -> Tuple[Dict[str, Any], str]:
        """
        Execute a tool call and return the response.
        
        Args:
            tool_call: Tuple of (call, role)
            
        Returns:
            Tuple of (function_message, sub_query_text)
        """
        call, role = tool_call
        func_args = json.loads(call.arguments)
        func_name = call.name
        sub_query_text = f"{func_args['company_name']} {func_args['subquery']} {func_args['year']}"
        print(f'Running tool `{func_name}` with `{func_args}`\nSubquery: {sub_query_text}')
        
        tool_response = {}
        if func_name == 'retrieve_financial_info':
            tool_response = self.retrieve_financial_info(**func_args)

        tool_call_id = getattr(call, "id", None)
        function_message = {
            "role": role,           
            "name": func_name,
            "tool_call_id": tool_call_id,
            "content": json.dumps(tool_response),
        }
        return function_message, sub_query_text

    def generate_answer(self, query: str) -> Tuple[Any, List[str]]:
        """
        Generate an answer to a query using the agent.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (llm_message, sub_queries)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        sub_queries = []
        
        while True:
            response = self.get_llm_response(messages)
            llm_message = response.choices[0].message  
            messages.append(llm_message)
            tool_calls = self.get_tool_calls(response)
            
            if tool_calls:
                for tool_call in tool_calls:
                    function_message, sub_query_text = self.get_tool_response(tool_call)
                    sub_queries.append(sub_query_text)
                    messages.append(function_message)
                continue
            else:
                return llm_message, sub_queries

    def parse_llm_response_to_dict(self, llm_response: Any) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response to dictionary.
        
        Args:
            llm_response: LLM response object
            
        Returns:
            Parsed dictionary or None if parsing fails
        """
        content = llm_response.content
        content = content.strip()
        if content.startswith("```json"):
            content = content[len("```json"):].strip("` \n")
        elif content.startswith("```"):
            content = content.strip("` \n")
        try:
            return json.loads(content)
        except Exception as e:
            print(f"Failed to parse JSON: {e}\nContent was:\n{content}")
            return None

    def get_output_json(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get structured output for a query.
        
        Args:
            query: User query
            
        Returns:
            Structured response dictionary or None
        """
        llm_message, sub_queries = self.generate_answer(query)
        llm_res_dict = self.parse_llm_response_to_dict(llm_message)
        if llm_res_dict:
            llm_res_dict['sub_queries'] = sub_queries
            llm_res_dict['query'] = query
            return {k:llm_res_dict[k] for k in ['query', 'answer', 'reasoning', 'sub_queries', 'sources']}
            # return llm_res_dict
        else:
            return None

    def query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Main method to query the financial agent.
        
        Args:
            query: User query about financial information
            
        Returns:
            Structured response with answer, reasoning, and sources
        """
        return self.get_output_json(query)


def main():
    """Main function to demonstrate usage."""
    # Initialize agent
    agent = FinancialAgent()
    
    # Example query
    query = "Which company had the highest operating margin in 2023?"
    print(f"Query: {query}")
    print("-" * 50)
    
    response = agent.query(query)
    if response:
        print(json.dumps(response, indent=2))
    else:
        print("Failed to get response")


if __name__ == "__main__":
    main() 