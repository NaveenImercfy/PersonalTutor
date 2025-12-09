"""
RAG Corpus Management Tools for Vertex AI using ADK function tools pattern.

RAG Corpus Management:
1. Create a new RAG corpus
2. Update an existing RAG corpus
3. List all RAG corpora
4. Get details of a specific RAG corpus
5. Delete a RAG corpus

RAG File Management (within a corpus):
6. Upload RAG files
7. List RAG files
8. Get RAG file details
9. Delete RAG files
10. Query RAG files
"""

import vertexai
import re
from vertexai.preview import rag
from google.adk.tools import FunctionTool
from typing import Dict, Optional, Any, List
from rag.config import (
    PROJECT_ID,
    LOCATION,
    RAG_DEFAULT_EMBEDDING_MODEL,
    RAG_DEFAULT_TOP_K,
    RAG_DEFAULT_SEARCH_TOP_K,
    RAG_DEFAULT_VECTOR_DISTANCE_THRESHOLD,
    RAG_DEFAULT_PAGE_SIZE
)
from rag.metadata_schema import (
    validate_metadata,
    get_metadata_schema,
    create_metadata_filter
)

# Initialize Vertex AI API
vertexai.init(project=PROJECT_ID, location=LOCATION)


def create_rag_corpus(
    display_name: str,
    description: Optional[str] = None,
    embedding_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a new RAG corpus in Vertex AI.
    
    Args:
        display_name: A human-readable name for the corpus
        description: Optional description for the corpus
        embedding_model: The embedding model to use (default: text-embedding-004)
    
    Returns:
        A dictionary containing the created corpus details including:
        - status: "success" or "error"
        - corpus_name: The full resource name of the created corpus
        - corpus_id: The ID portion of the corpus name
        - display_name: The human-readable name provided
        - error_message: Present only if an error occurred
    """
    if embedding_model is None:
        embedding_model = RAG_DEFAULT_EMBEDDING_MODEL
    try:
        # Configure embedding model
        embedding_model_config = rag.EmbeddingModelConfig(
            publisher_model=f"publishers/google/models/{embedding_model}"
        )
        
        # Create the corpus
        corpus = rag.create_corpus(
            display_name=display_name,
            description=description or f"RAG corpus: {display_name}",
            embedding_model_config=embedding_model_config,
        )
        
        # Extract corpus ID from the full name
        corpus_id = corpus.name.split('/')[-1]
        
        return {
            "status": "success",
            "corpus_name": corpus.name,
            "corpus_id": corpus_id,
            "display_name": corpus.display_name,
            "message": f"Successfully created RAG corpus '{display_name}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "message": f"Failed to create RAG corpus: {str(e)}"
        }


def update_rag_corpus(
    corpus_id: str,
    display_name: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Updates an existing RAG corpus with new display name and/or description.
    
    Args:
        corpus_id: The ID of the corpus to update
        display_name: New display name for the corpus (optional)
        description: New description for the corpus (optional)
    
    Returns:
        A dictionary containing the update result:
        - status: "success" or "error"
        - corpus_name: The full resource name of the updated corpus
        - corpus_id: The ID of the corpus
        - display_name: The updated display name
        - description: The updated description
        - error_message: Present only if an error occurred
    """
    try:
        # Construct full corpus name
        corpus_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"
        
        # Get the existing corpus
        corpus = rag.get_corpus(name=corpus_name)
        
        # Update fields if provided
        if display_name:
            corpus.display_name = display_name
        if description:
            corpus.description = description
        
        # Apply updates
        updated_corpus = rag.update_corpus(
            corpus=corpus,
            update_mask=["display_name", "description"]
        )
        
        return {
            "status": "success",
            "corpus_name": updated_corpus.name,
            "corpus_id": corpus_id,
            "display_name": updated_corpus.display_name,
            "description": updated_corpus.description,
            "message": f"Successfully updated RAG corpus '{corpus_id}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "error_message": str(e),
            "message": f"Failed to update RAG corpus: {str(e)}"
        }


def list_rag_corpora() -> Dict[str, Any]:
    """
    Lists all RAG corpora in the current project and location.
    
    Returns:
        A dictionary containing the list of corpora:
        - status: "success" or "error"
        - corpora: List of corpus objects with id, name, and display_name
        - count: Number of corpora found
        - error_message: Present only if an error occurred
    """
    try:
        corpora = rag.list_corpora()
        
        corpus_list = []
        for corpus in corpora:
            corpus_id = corpus.name.split('/')[-1]
            
            # Get corpus status
            status = None
            if hasattr(corpus, "corpus_status") and hasattr(corpus.corpus_status, "state"):
                status = corpus.corpus_status.state
            elif hasattr(corpus, "corpusStatus") and hasattr(corpus.corpusStatus, "state"):
                status = corpus.corpusStatus.state
            
            # Make an explicit API call to count files
            files_count = 0
            try:
                # List all files to get the count
                files_response = rag.list_files(corpus_name=corpus.name)
                
                if hasattr(files_response, "rag_files"):
                    files_count = len(files_response.rag_files)
            except Exception:
                # If counting files fails, continue with zero count
                pass
            
            corpus_list.append({
                "id": corpus_id,
                "name": corpus.name,
                "display_name": corpus.display_name,
                "description": corpus.description if hasattr(corpus, "description") else None,
                "create_time": str(corpus.create_time) if hasattr(corpus, "create_time") else None,
                "files_count": files_count,
                "status": status
            })
        
        return {
            "status": "success",
            "corpora": corpus_list,
            "count": len(corpus_list),
            "message": f"Found {len(corpus_list)} RAG corpora"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "message": f"Failed to list RAG corpora: {str(e)}"
        }


def get_rag_corpus(corpus_id: str) -> Dict[str, Any]:
    """
    Retrieves details of a specific RAG corpus.
    
    Args:
        corpus_id: The ID of the corpus to retrieve
    
    Returns:
        A dictionary containing the corpus details:
        - status: "success" or "error"
        - corpus: Detailed information about the corpus
        - files_count: Number of files in the corpus
        - error_message: Present only if an error occurred
    """
    try:
        # Construct full corpus name
        corpus_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"
        
        # Get the corpus
        corpus = rag.get_corpus(name=corpus_name)
        
        # Get corpus status
        status = None
        if hasattr(corpus, "corpus_status") and hasattr(corpus.corpus_status, "state"):
            status = corpus.corpus_status.state
        elif hasattr(corpus, "corpusStatus") and hasattr(corpus.corpusStatus, "state"):
            status = corpus.corpusStatus.state
        
        # Make an explicit API call to count files
        files_count = 0
        try:
            # List all files to get the count
            files_response = rag.list_files(corpus_name=corpus_name)
            
            if hasattr(files_response, "rag_files"):
                files_count = len(files_response.rag_files)
        except Exception as file_error:
            # If counting files fails, log but continue with zero count
            print(f"Warning: Could not count files: {str(file_error)}")
        
        # Extract basic information
        corpus_details = {
            "id": corpus_id,
            "name": corpus.name,
            "display_name": corpus.display_name,
            "description": corpus.description if hasattr(corpus, "description") else None,
            "create_time": str(corpus.create_time) if hasattr(corpus, "create_time") else None,
            "update_time": str(corpus.update_time) if hasattr(corpus, "update_time") else None,
            "files_count": files_count,
            "state": status
        }
        
        # Include raw API response data for transparency
        raw_data = {}
        if hasattr(corpus, "to_dict"):
            raw_data = corpus.to_dict()
        elif hasattr(corpus, "__dict__"):
            raw_data = {k: v for k, v in corpus.__dict__.items() if not k.startswith('_')}
        
        if raw_data:
            corpus_details["raw_api_data"] = raw_data
        
        return {
            "status": "success",
            "corpus": corpus_details,
            "files_count": files_count,
            "message": f"Successfully retrieved RAG corpus '{corpus_id}' with {files_count} files"
        }
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "error_message": str(e),
            "message": f"Failed to retrieve RAG corpus: {str(e)}"
        }


def delete_rag_corpus(corpus_id: str) -> Dict[str, Any]:
    """
    Deletes a RAG corpus.
    
    Args:
        corpus_id: The ID of the corpus to delete
    
    Returns:
        A dictionary containing the deletion result:
        - status: "success" or "error"
        - corpus_id: The ID of the deleted corpus
        - error_message: Present only if an error occurred
    """
    try:
        # Construct full corpus name
        corpus_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"
        
        # Delete the corpus
        rag.delete_corpus(name=corpus_name)
        
        return {
            "status": "success",
            "corpus_id": corpus_id,
            "message": f"Successfully deleted RAG corpus '{corpus_id}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "error_message": str(e),
            "message": f"Failed to delete RAG corpus: {str(e)}"
        }


# Function for importing documents into a RAG corpus
def import_document_to_corpus(
    corpus_id: str,
    gcs_uri: str,
    rag_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Dict[str, Any]:
    """
    Imports a document from Google Cloud Storage into a RAG corpus with optional metadata.
    
    Args:
        corpus_id: The ID of the corpus to import the document into
        gcs_uri: GCS path of the document to import (gs://bucket-name/file-name)
        rag_metadata: Optional metadata dictionary for the document. Must include:
                     - board (required): Education board (e.g., "CBSE")
                     - grade (required): Grade level as string (e.g., "10")
                     - subject (required): Subject name (e.g., "Mathematics")
                     Optional fields: term, chapter, chapter_number, publisher, edition,
                     language, content_type, difficulty
        chunk_size: Optional chunk size for document splitting (default: 512)
        chunk_overlap: Optional chunk overlap for document splitting (default: 100)
    
    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - corpus_id: The ID of the corpus
        - message: Status message
        - metadata_validation: Validation result if metadata was provided
    """
    try:
        # Construct full corpus name
        corpus_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"
        
        # Validate metadata if provided
        metadata_validation = None
        normalized_metadata = None
        
        if rag_metadata:
            validation_result = validate_metadata(rag_metadata, strict=False)
            metadata_validation = {
                "valid": validation_result["valid"],
                "errors": validation_result["errors"],
                "warnings": validation_result["warnings"]
            }
            
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "corpus_id": corpus_id,
                    "error_message": f"Metadata validation failed: {', '.join(validation_result['errors'])}",
                    "metadata_validation": metadata_validation,
                    "message": f"Failed to import document: Invalid metadata. Errors: {', '.join(validation_result['errors'])}"
                }
            
            normalized_metadata = validation_result["normalized"]
            
            # Show warnings if any
            if validation_result["warnings"]:
                print(f"Metadata warnings: {', '.join(validation_result['warnings'])}")
        
        # Prepare import parameters
        import_params = {
            "corpus_name": corpus_name,
            "paths": [gcs_uri]
        }
        
        # Add chunking parameters if provided
        if chunk_size is not None:
            import_params["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            import_params["chunk_overlap"] = chunk_overlap
        
        # Add metadata if provided
        if normalized_metadata:
            import_params["rag_metadata"] = normalized_metadata
        
        # Import document
        result = rag.import_files(**import_params)
        
        # Return success result
        response = {
            "status": "success",
            "corpus_id": corpus_id,
            "message": f"Successfully imported document {gcs_uri} to corpus '{corpus_id}'"
        }
        
        if metadata_validation:
            response["metadata_validation"] = metadata_validation
            response["metadata"] = normalized_metadata
        
        return response
        
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "error_message": str(e),
            "message": f"Failed to import document: {str(e)}"
        }

# RAG File Management Functions

def list_rag_files(
    corpus_id: str,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lists all RAG files in a corpus.
    
    Args:
        corpus_id: The ID of the corpus to list files from
        page_size: Maximum number of files to return (default: 50)
        page_token: Token for pagination
    
    Returns:
        A dictionary containing the list of files:
        - status: "success" or "error"
        - corpus_id: The ID of the corpus
        - files: List of file objects
        - count: Number of files found
        - next_page_token: Token for the next page (if any)
        - error_message: Present only if an error occurred
    """
    if page_size is None:
        page_size = RAG_DEFAULT_PAGE_SIZE
    try:
        # Construct full corpus name
        corpus_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"
        
        # List files
        response = rag.list_files(
            corpus_name=corpus_name,
            page_size=page_size, 
            page_token=page_token
        )
        
        # Process files
        files = []
        for file in response.rag_files:
            file_id = file.name.split("/")[-1]
            files.append({
                "id": file_id,
                "name": file.name,
                "display_name": file.display_name if hasattr(file, "display_name") else None,
                "description": file.description if hasattr(file, "description") else None,
                "source_uri": file.source_uri if hasattr(file, "source_uri") else None,
                "create_time": str(file.create_time) if hasattr(file, "create_time") else None,
                "update_time": str(file.update_time) if hasattr(file, "update_time") else None
            })
        
        return {
            "status": "success",
            "corpus_id": corpus_id,
            "files": files,
            "count": len(files),
            "next_page_token": response.next_page_token if hasattr(response, "next_page_token") else None,
            "message": f"Found {len(files)} file(s) in corpus '{corpus_id}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "error_message": str(e),
            "message": f"Failed to list files: {str(e)}"
        }

def get_rag_file(
    corpus_id: str,
    file_id: str
) -> Dict[str, Any]:
    """
    Gets details of a specific RAG file in a corpus.
    
    Args:
        corpus_id: The ID of the corpus
        file_id: The ID of the file to get
    
    Returns:
        A dictionary containing the file details:
        - status: "success" or "error"
        - corpus_id: The ID of the corpus
        - file: Detailed information about the file
        - error_message: Present only if an error occurred
    """
    try:
        # Construct full file name
        file_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}/ragFiles/{file_id}"
        
        # Get the file
        file = rag.get_file(name=file_name)
        
        # Extract file details
        file_details = {
            "id": file_id,
            "name": file.name,
            "display_name": file.display_name if hasattr(file, "display_name") else None,
            "description": file.description if hasattr(file, "description") else None,
            "source_uri": file.source_uri if hasattr(file, "source_uri") else None,
            "create_time": str(file.create_time) if hasattr(file, "create_time") else None,
            "update_time": str(file.update_time) if hasattr(file, "update_time") else None
        }
        
        # Include raw API response data for transparency
        raw_data = {}
        if hasattr(file, "to_dict"):
            raw_data = file.to_dict()
        elif hasattr(file, "__dict__"):
            raw_data = {k: v for k, v in file.__dict__.items() if not k.startswith('_')}
        
        if raw_data:
            file_details["raw_api_data"] = raw_data
        
        return {
            "status": "success",
            "corpus_id": corpus_id,
            "file": file_details,
            "message": f"Successfully retrieved file '{file_id}' from corpus '{corpus_id}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "file_id": file_id,
            "error_message": str(e),
            "message": f"Failed to retrieve file: {str(e)}"
        }

def delete_rag_file(
    corpus_id: str,
    file_id: str
) -> Dict[str, Any]:
    """
    Deletes a RAG file from a corpus.
    
    Args:
        corpus_id: The ID of the corpus
        file_id: The ID of the file to delete
    
    Returns:
        A dictionary containing the deletion result:
        - status: "success" or "error"
        - corpus_id: The ID of the corpus
        - file_id: The ID of the deleted file
        - error_message: Present only if an error occurred
    """
    try:
        # Construct full file name
        file_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}/ragFiles/{file_id}"
        
        # Delete the file
        rag.delete_file(name=file_name)
        
        return {
            "status": "success",
            "corpus_id": corpus_id,
            "file_id": file_id,
            "message": f"Successfully deleted file '{file_id}' from corpus '{corpus_id}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "file_id": file_id,
            "error_message": str(e),
            "message": f"Failed to delete file: {str(e)}"
        }

# Function for simple direct corpus querying
def query_rag_corpus(
    corpus_id: str,
    query_text: str,
    top_k: Optional[int] = None,
    vector_distance_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Directly queries a RAG corpus using the Vertex AI RAG API with optional metadata filtering.
    
    Args:
        corpus_id: The ID of the corpus to query
        query_text: The search query text
        top_k: Maximum number of results to return (default: 10)
        vector_distance_threshold: Threshold for vector similarity (default: 0.5)
        metadata_filter: Optional dictionary to filter results by metadata fields
                       (e.g., {"board": "CBSE", "grade": "10", "subject": "Mathematics"})
        
    Returns:
        A dictionary containing the query results
    """
    if top_k is None:
        top_k = RAG_DEFAULT_TOP_K
    if vector_distance_threshold is None:
        vector_distance_threshold = RAG_DEFAULT_VECTOR_DISTANCE_THRESHOLD
    try:
        # Construct full corpus resource path
        corpus_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"
        
        # Create the resource config
        rag_resource = rag.RagResource(rag_corpus=corpus_path)
        
        # Normalize metadata filter if provided
        normalized_filter = None
        if metadata_filter:
            normalized_filter = create_metadata_filter(metadata_filter)
        
        # Build filter conditions for server-side filtering
        # Note: Vertex AI RAG API may have limited metadata filter support
        # We'll apply client-side filtering as a reliable fallback
        filter_obj = None
        if vector_distance_threshold is not None:
            try:
                # Create filter with vector distance threshold
                filter_obj = rag.utils.resources.Filter(
                    vector_distance_threshold=vector_distance_threshold
                )
            except Exception:
                # If Filter creation fails, continue without filter
                filter_obj = None
        
        # Configure retrieval parameters
        if filter_obj:
            retrieval_config = rag.RagRetrievalConfig(
                top_k=top_k * 2 if normalized_filter else top_k,  # Fetch more results if we need to filter client-side
                filter=filter_obj
            )
        else:
            retrieval_config = rag.RagRetrievalConfig(
                top_k=top_k * 2 if normalized_filter else top_k  # Fetch more results if we need to filter client-side
            )
        
        # Execute the query directly using the API
        response = rag.retrieval_query(
            rag_resources=[rag_resource],
            text=query_text,
            rag_retrieval_config=retrieval_config
        )
        
        # Process the results
        results = []
        if hasattr(response, "contexts"):
            # Handle different response structures
            contexts = response.contexts
            if hasattr(contexts, "contexts"):
                contexts = contexts.contexts
            
            # Extract text and metadata from each context
            for context in contexts:
                result = {
                    "text": context.text if hasattr(context, "text") else "",
                    "source_uri": context.source_uri if hasattr(context, "source_uri") else None,
                    "relevance_score": context.relevance_score if hasattr(context, "relevance_score") else None
                }
                
                # Debug: Check all available attributes on context object
                context_attrs = [attr for attr in dir(context) if not attr.startswith('_')]
                
                # Extract page number if available - check multiple possible locations
                page_number = None
                
                # Check direct attributes
                if hasattr(context, "page_number"):
                    page_number = context.page_number
                elif hasattr(context, "page"):
                    page_number = context.page
                elif hasattr(context, "page_num"):
                    page_number = context.page_num
                elif hasattr(context, "pageNumber"):
                    page_number = context.pageNumber
                
                # Check if context has a dict representation
                if page_number is None and hasattr(context, "__dict__"):
                    context_dict = context.__dict__
                    page_number = (context_dict.get("page_number") or 
                                  context_dict.get("page") or 
                                  context_dict.get("page_num") or
                                  context_dict.get("pageNumber"))
                
                # Extract metadata if available
                context_metadata = None
                if hasattr(context, "metadata") and context.metadata:
                    context_metadata = context.metadata
                elif hasattr(context, "rag_metadata") and context.rag_metadata:
                    context_metadata = context.rag_metadata
                
                # Store metadata in result
                if context_metadata:
                    # Handle different metadata formats (dict, protobuf, etc.)
                    if isinstance(context_metadata, dict):
                        result["metadata"] = context_metadata
                        # Check if page number is in metadata dict
                        if page_number is None:
                            page_number = context_metadata.get("page_number") or context_metadata.get("page") or context_metadata.get("page_num")
                    elif hasattr(context_metadata, "__dict__"):
                        result["metadata"] = {k: v for k, v in context_metadata.__dict__.items() if not k.startswith('_')}
                        # Check if page number is in metadata attributes
                        if page_number is None:
                            page_number = getattr(context_metadata, "page_number", None) or getattr(context_metadata, "page", None) or getattr(context_metadata, "page_num", None)
                    else:
                        # Try to convert to dict
                        try:
                            result["metadata"] = dict(context_metadata)
                            if page_number is None:
                                page_number = result["metadata"].get("page_number") or result["metadata"].get("page") or result["metadata"].get("page_num")
                        except:
                            result["metadata"] = {}
                
                # Store page number if found
                if page_number is not None:
                    result["page_number"] = page_number
                
                # Try to extract page number from source_uri if it contains page info
                if page_number is None and result.get("source_uri"):
                    # Some URIs might contain page information in the format: gs://bucket/file.pdf#page=123
                    source_uri = result["source_uri"]
                    if "#page=" in source_uri:
                        try:
                            page_number = int(source_uri.split("#page=")[1].split("&")[0])
                            result["page_number"] = page_number
                        except:
                            pass
                
                # Try to extract page number from text content if embedded
                # Look for patterns like "Page 45", "p. 45", "pg 45", etc.
                if page_number is None and result.get("text"):
                    text = result["text"]
                    # Look for common page number patterns
                    page_patterns = [
                        r'\bpage\s+(\d+)\b',
                        r'\bp\.\s*(\d+)\b',
                        r'\bpg\.?\s*(\d+)\b',
                        r'\bpage\s*:\s*(\d+)\b',
                        r'\(page\s+(\d+)\)',
                        r'\[page\s+(\d+)\]',
                    ]
                    for pattern in page_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            try:
                                page_number = int(match.group(1))
                                result["page_number"] = page_number
                                break
                            except:
                                continue
                
                # Store debug info about available fields (for troubleshooting)
                # Only store if page number wasn't found (to help debug)
                if page_number is None and 'context_attrs' in locals():
                    result["debug_info"] = {
                        "context_attributes": context_attrs[:20],  # Limit to first 20
                        "has_page_number": False
                    }
                
                # Apply client-side metadata filtering if metadata filter is provided
                if normalized_filter:
                    # If result has no metadata, skip it when filtering is requested
                    if "metadata" not in result or not result["metadata"]:
                        continue
                    
                    # Check if all filter criteria match
                    matches = True
                    for key, value in normalized_filter.items():
                        # Check if metadata field exists and matches
                        if key not in result["metadata"]:
                            matches = False
                            break
                        
                        # Compare values (handle string normalization)
                        metadata_value = result["metadata"][key]
                        if isinstance(metadata_value, str) and isinstance(value, str):
                            # Normalize both values for comparison
                            # For board field, normalize similarly to create_metadata_filter
                            if key == "board":
                                # Normalize metadata_value
                                norm_meta = str(metadata_value).strip().upper()
                                norm_meta = norm_meta.replace("-", "_").replace(".", "_")
                                # Handle camelCase by inserting underscores before capitals
                                norm_meta = re.sub(r'(?<!^)(?<!_)([A-Z])', r'_\1', norm_meta)
                                norm_meta = re.sub(r'_+', '_', norm_meta.replace(" ", "_")).strip('_')
                                # Normalize filter value (already normalized, but ensure consistency)
                                norm_filter = str(value).strip().upper()
                                norm_filter = norm_filter.replace("-", "_").replace(".", "_")
                                norm_filter = re.sub(r'_+', '_', norm_filter.replace(" ", "_")).strip('_')
                                # Compare normalized values
                                if norm_meta != norm_filter:
                                    matches = False
                                    break
                            else:
                                # Case-insensitive string comparison for other fields
                                if metadata_value.strip().lower() != value.strip().lower():
                                    matches = False
                                    break
                        elif metadata_value != value:
                            matches = False
                            break
                    
                    # Skip this result if it doesn't match the filter
                    if not matches:
                        continue
                
                # Add result if it passed all filters
                results.append(result)
                
                # Limit results to top_k if we fetched extra for client-side filtering
                if normalized_filter and len(results) >= top_k:
                    break
        
        # Add note about multiple results from same file
        note = ""
        if len(results) > 1:
            # Count how many results come from the same file
            file_counts = {}
            for r in results:
                source = r.get("source_uri", "unknown")
                file_counts[source] = file_counts.get(source, 0) + 1
            
            multiple_from_same_file = [f for f, count in file_counts.items() if count > 1]
            if multiple_from_same_file:
                note = f" Note: Multiple chunks retrieved from the same document(s) - this is normal as documents are split into chunks for better search."
        
        return {
            "status": "success",
            "corpus_id": corpus_id,
            "results": results,
            "count": len(results),
            "query": query_text,
            "metadata_filter": metadata_filter if metadata_filter else None,
            "message": f"Found {len(results)} results for query: '{query_text}'" + 
                      (f" with metadata filter: {metadata_filter}" if metadata_filter else "") + note,
            "note": note if note else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "error_message": str(e),
            "message": f"Failed to query corpus: {str(e)}"
        }

# Function to search across all corpora
def search_all_corpora(
    query_text: str,
    top_k_per_corpus: Optional[int] = None,
    vector_distance_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Searches across ALL available corpora for the given query text with optional metadata filtering.
    When a user wants to search for information without specifying a corpus,
    this is the default tool to use.
    
    Args:
        query_text: The search query text
        top_k_per_corpus: Maximum number of results to return per corpus (default: 5)
        vector_distance_threshold: Threshold for vector similarity (default: 0.5)
        metadata_filter: Optional dictionary to filter results by metadata fields
                       (e.g., {"board": "CBSE", "grade": "10", "subject": "Mathematics"})
        
    Returns:
        A dictionary containing the combined search results with citations
    """
    if top_k_per_corpus is None:
        top_k_per_corpus = RAG_DEFAULT_SEARCH_TOP_K
    if vector_distance_threshold is None:
        vector_distance_threshold = RAG_DEFAULT_VECTOR_DISTANCE_THRESHOLD
    try:
        # First, list all available corpora
        corpora_response = list_rag_corpora()
        
        if corpora_response["status"] != "success":
            return {
                "status": "error",
                "error_message": f"Failed to list corpora: {corpora_response.get('error_message', '')}",
                "message": "Failed to search all corpora - could not retrieve corpus list"
            }
        
        all_corpora = corpora_response.get("corpora", [])
        
        if not all_corpora:
            return {
                "status": "warning",
                "message": "No corpora found to search in"
            }
        
        # Search in each corpus
        all_results = []
        corpus_results_map = {}  # Map of corpus name to its results
        searched_corpora = []
        
        for corpus in all_corpora:
            corpus_id = corpus["id"]
            corpus_name = corpus.get("display_name", corpus_id)
            
            # Query this corpus with metadata filter if provided
            corpus_results = query_rag_corpus(
                corpus_id=corpus_id,
                query_text=query_text,
                top_k=top_k_per_corpus,
                vector_distance_threshold=vector_distance_threshold,
                metadata_filter=metadata_filter
            )
            
            # Add corpus info to the results
            if corpus_results["status"] == "success":
                results = corpus_results.get("results", [])
                corpus_specific_results = []
                
                for result in results:
                    # Add citation and source information
                    result["corpus_id"] = corpus_id
                    result["corpus_name"] = corpus_name
                    result["citation"] = f"[Source: {corpus_name} ({corpus_id})]"
                    
                    # Add source file information if available
                    if "source_uri" in result and result["source_uri"]:
                        source_path = result["source_uri"]
                        file_name = source_path.split("/")[-1] if "/" in source_path else source_path
                        # Remove any fragment identifiers from file name
                        if "#" in file_name:
                            file_name = file_name.split("#")[0]
                        result["citation"] += f" File: {file_name}"
                    
                    # Add page number if available
                    if "page_number" in result and result["page_number"] is not None:
                        result["citation"] += f" Page: {result['page_number']}"
                    
                    corpus_specific_results.append(result)
                    all_results.append(result)
                
                # Save results for this corpus
                if corpus_specific_results:
                    corpus_results_map[corpus_name] = {
                        "corpus_id": corpus_id,
                        "corpus_name": corpus_name,
                        "results": corpus_specific_results,
                        "count": len(corpus_specific_results)
                    }
                    searched_corpora.append(corpus_name)
        
        # Sort all results by relevance score (if available)
        all_results.sort(
            key=lambda x: x.get("relevance_score", 0) if x.get("relevance_score") is not None else 0,
            reverse=True
        )
        
        # Format citations summary
        citations_summary = []
        for corpus_name in searched_corpora:
            corpus_data = corpus_results_map[corpus_name]
            citations_summary.append(
                f"{corpus_name} ({corpus_data['corpus_id']}): {corpus_data['count']} results"
            )
        
        return {
            "status": "success",
            "results": all_results,
            "corpus_results": corpus_results_map,
            "searched_corpora": searched_corpora,
            "citations_summary": citations_summary,
            "count": len(all_results),
            "query": query_text,
            "metadata_filter": metadata_filter if metadata_filter else None,
            "message": f"Found {len(all_results)} results for query '{query_text}' across {len(searched_corpora)} corpora" +
                      (f" with metadata filter: {metadata_filter}" if metadata_filter else ""),
            "citation_note": "Each result includes a citation indicating its source corpus and file."
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "message": f"Failed to search all corpora: {str(e)}"
        }


def search_corpus_by_name(
    corpus_name: str,
    query_text: str,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Finds a corpus by its display name and performs a search query on it with optional metadata filtering.

    Args:
        corpus_name: The display name of the RAG corpus to search.
        query_text: The question to ask the corpus.
        metadata_filter: Optional dictionary to filter results by metadata fields
                       (e.g., {"board": "CBSE", "grade": "10", "subject": "Mathematics"})

    Returns:
        A dictionary containing the search results and citation summary.
    """
    try:
        corpora_response = list_rag_corpora()
        if corpora_response["status"] != "success":
            return {
                "status": "error",
                "error_message": f"Failed to list corpora to find '{corpus_name}'.",
                "message": f"Could not find corpus '{corpus_name}' because listing corpora failed."
            }

        target_corpus = None
        for corpus in corpora_response.get("corpora", []):
            if corpus.get("display_name") and corpus.get("display_name").strip().lower() == corpus_name.strip().lower():
                target_corpus = corpus
                break

        if not target_corpus:
            return {
                "status": "error",
                "message": f"❌ Error: Corpus with name '{corpus_name}' not found."
            }

        corpus_id = target_corpus["id"]
        return query_rag_corpus(
            corpus_id=corpus_id, 
            query_text=query_text,
            metadata_filter=metadata_filter
        )
    except Exception as e:
        return {"status": "error", "error_message": str(e), "message": f"An unexpected error occurred while searching by name: {e}"}

# Helper function to get metadata schema information
def get_metadata_schema_info() -> Dict[str, Any]:
    """
    Returns information about the metadata schema for RAG corpus file imports.
    
    Returns:
        A dictionary containing schema information including required/optional fields,
        field types, allowed values, and examples.
    """
    return {
        "status": "success",
        "schema": get_metadata_schema(),
        "message": "Metadata schema information retrieved successfully"
    }

# Helper function to inspect metadata in a corpus (for debugging)
def inspect_corpus_metadata(
    corpus_id: str,
    query_text: str = "",
    sample_size: int = 20
) -> Dict[str, Any]:
    """
    Inspects metadata values in a corpus by querying without filters.
    Useful for debugging metadata filtering issues.
    
    Args:
        corpus_id: The ID of the corpus to inspect
        query_text: Optional query text (use empty string to get random samples)
        sample_size: Number of results to inspect (default: 20)
    
    Returns:
        A dictionary containing sample metadata values found in the corpus
    """
    try:
        # Query without metadata filter to see what's actually stored
        if not query_text:
            query_text = "science"  # Generic query to get some results
        
        result = query_rag_corpus(
            corpus_id=corpus_id,
            query_text=query_text,
            top_k=sample_size,
            metadata_filter=None  # No filter - get all results
        )
        
        if result["status"] != "success":
            return {
                "status": "error",
                "error_message": result.get("error_message", "Failed to query corpus"),
                "message": "Failed to inspect metadata"
            }
        
        # Collect unique metadata values
        metadata_samples = {}
        metadata_counts = {}
        
        for res in result.get("results", []):
            if "metadata" in res and res["metadata"]:
                meta = res["metadata"]
                # Collect all metadata fields and their values
                for key, value in meta.items():
                    if key not in metadata_samples:
                        metadata_samples[key] = set()
                        metadata_counts[key] = {}
                    
                    value_str = str(value)
                    metadata_samples[key].add(value_str)
                    metadata_counts[key][value_str] = metadata_counts[key].get(value_str, 0) + 1
        
        # Convert sets to sorted lists for better readability
        metadata_summary = {}
        for key, values in metadata_samples.items():
            # Sort by frequency (most common first)
            sorted_values = sorted(
                values,
                key=lambda v: metadata_counts[key].get(v, 0),
                reverse=True
            )
            metadata_summary[key] = {
                "unique_values": sorted_values[:50],  # Limit to top 50
                "total_unique": len(values),
                "value_counts": {v: metadata_counts[key][v] for v in sorted_values[:10]}  # Top 10 counts
            }
        
        return {
            "status": "success",
            "corpus_id": corpus_id,
            "query_text": query_text,
            "results_inspected": len(result.get("results", [])),
            "metadata_fields_found": list(metadata_summary.keys()),
            "metadata_samples": metadata_summary,
            "message": f"Inspected {len(result.get('results', []))} results from corpus '{corpus_id}'"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "error_message": str(e),
            "message": f"Failed to inspect metadata: {str(e)}"
        }

# Create FunctionTools from the functions for the RAG corpus management tools
create_corpus_tool = FunctionTool(create_rag_corpus)
update_corpus_tool = FunctionTool(update_rag_corpus)
list_corpora_tool = FunctionTool(list_rag_corpora)
get_corpus_tool = FunctionTool(get_rag_corpus)
delete_corpus_tool = FunctionTool(delete_rag_corpus)
import_document_tool = FunctionTool(import_document_to_corpus)
get_metadata_schema_tool = FunctionTool(get_metadata_schema_info)

# Create FunctionTools from the functions for the RAG file management tools
list_files_tool = FunctionTool(list_rag_files)
get_file_tool = FunctionTool(get_rag_file)
delete_file_tool = FunctionTool(delete_rag_file)

# Create FunctionTools from the functions for the RAG query tools
query_rag_corpus_tool = FunctionTool(query_rag_corpus)
search_all_corpora_tool = FunctionTool(search_all_corpora) 
search_corpus_by_name_tool = FunctionTool(search_corpus_by_name)
inspect_metadata_tool = FunctionTool(inspect_corpus_metadata)