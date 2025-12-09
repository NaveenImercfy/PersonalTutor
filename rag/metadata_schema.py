"""
Metadata Schema for RAG Corpus File Imports

This module defines and validates metadata schemas for educational content
uploaded to RAG corpora. It enforces consistent metadata formatting to enable
efficient filtering and searching.
"""

from typing import Dict, Any, Optional, List
import re


# Required metadata fields
REQUIRED_METADATA_FIELDS = {
    "board": str,      # Education board (e.g., "CBSE", "ICSE")
    "grade": str,      # Grade level as string (e.g., "10")
    "subject": str     # Subject name (e.g., "Mathematics")
}

# Optional metadata fields with their types
OPTIONAL_METADATA_FIELDS = {
    "term": str,                    # Optional: omit for annual subjects
    "chapter": str,                 # Chapter name
    "chapter_number": str,          # Chapter number
    "publisher": str,               # Publisher name
    "edition": str,                 # Edition year/version
    "language": str,                # Language (e.g., "English")
    "content_type": str,            # theory|exercises|solutions|examples
    "difficulty": str              # basic|medium|advanced
}

# Allowed values for enum-like fields
ALLOWED_CONTENT_TYPES = ["theory", "exercises", "solutions", "examples"]
ALLOWED_DIFFICULTY_LEVELS = ["basic", "medium", "advanced"]

# Common education boards (for validation suggestions)
COMMON_BOARDS = ["CBSE", "ICSE", "State", "IGCSE", "IB", "State_Board"]


def validate_metadata(metadata: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """
    Validates metadata against the schema requirements.
    
    Args:
        metadata: Dictionary containing metadata fields
        strict: If True, enforces stricter validation (e.g., enum values)
    
    Returns:
        Dictionary with validation result:
        - valid: bool - Whether metadata is valid
        - errors: List[str] - List of validation errors
        - warnings: List[str] - List of validation warnings
        - normalized: Dict[str, Any] - Normalized metadata (if valid)
    """
    errors = []
    warnings = []
    normalized = {}
    
    # Check required fields
    for field, field_type in REQUIRED_METADATA_FIELDS.items():
        if field not in metadata:
            errors.append(f"Missing required field: '{field}'")
        else:
            value = metadata[field]
            
            # Type validation
            if not isinstance(value, field_type):
                errors.append(f"Field '{field}' must be of type {field_type.__name__}, got {type(value).__name__}")
            else:
                # Normalize and validate specific fields
                if field == "board":
                    # Normalize board name: uppercase, replace spaces with underscores
                    normalized_value = str(value).strip().upper().replace(" ", "_")
                    if normalized_value != value:
                        warnings.append(f"Board name normalized: '{value}' -> '{normalized_value}'")
                    normalized[field] = normalized_value
                elif field == "grade":
                    # Ensure grade is a string
                    normalized_value = str(value).strip()
                    if normalized_value != str(value):
                        warnings.append(f"Grade normalized to string: '{value}' -> '{normalized_value}'")
                    normalized[field] = normalized_value
                elif field == "subject":
                    # Normalize subject: title case, replace spaces with underscores for multi-word
                    normalized_value = str(value).strip()
                    # Keep original format but ensure no leading/trailing spaces
                    normalized[field] = normalized_value
                else:
                    normalized[field] = value
    
    # Check optional fields
    for field, field_type in OPTIONAL_METADATA_FIELDS.items():
        if field in metadata:
            value = metadata[field]
            
            # Type validation
            if not isinstance(value, field_type):
                errors.append(f"Optional field '{field}' must be of type {field_type.__name__}, got {type(value).__name__}")
            else:
                # Validate enum-like fields
                if field == "content_type" and strict:
                    if value not in ALLOWED_CONTENT_TYPES:
                        errors.append(f"content_type must be one of {ALLOWED_CONTENT_TYPES}, got '{value}'")
                
                if field == "difficulty" and strict:
                    if value not in ALLOWED_DIFFICULTY_LEVELS:
                        errors.append(f"difficulty must be one of {ALLOWED_DIFFICULTY_LEVELS}, got '{value}'")
                
                # Normalize string fields
                if field_type == str:
                    normalized_value = str(value).strip()
                    if normalized_value != value:
                        warnings.append(f"Field '{field}' normalized: '{value}' -> '{normalized_value}'")
                    normalized[field] = normalized_value
                else:
                    normalized[field] = value
    
    # Check for unknown fields
    all_allowed_fields = set(REQUIRED_METADATA_FIELDS.keys()) | set(OPTIONAL_METADATA_FIELDS.keys())
    unknown_fields = set(metadata.keys()) - all_allowed_fields
    if unknown_fields:
        warnings.append(f"Unknown fields found (will be ignored): {', '.join(unknown_fields)}")
    
    # Validate key naming conventions
    for key in metadata.keys():
        if key != key.lower():
            warnings.append(f"Key '{key}' should be lowercase (use '{key.lower()}')")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized": normalized if len(errors) == 0 else None
    }


def get_metadata_schema() -> Dict[str, Any]:
    """
    Returns the complete metadata schema definition.
    
    Returns:
        Dictionary containing schema information
    """
    return {
        "required_fields": list(REQUIRED_METADATA_FIELDS.keys()),
        "optional_fields": list(OPTIONAL_METADATA_FIELDS.keys()),
        "field_types": {
            **{k: v.__name__ for k, v in REQUIRED_METADATA_FIELDS.items()},
            **{k: v.__name__ for k, v in OPTIONAL_METADATA_FIELDS.items()}
        },
        "allowed_content_types": ALLOWED_CONTENT_TYPES,
        "allowed_difficulty_levels": ALLOWED_DIFFICULTY_LEVELS,
        "common_boards": COMMON_BOARDS,
        "examples": {
            "minimum": {
                "board": "CBSE",
                "grade": "10",
                "subject": "Mathematics"
            },
            "complete": {
                "board": "CBSE",
                "grade": "10",
                "subject": "Mathematics",
                "term": "1",
                "chapter": "Algebra",
                "chapter_number": "3",
                "publisher": "NCERT",
                "edition": "2024",
                "language": "English",
                "content_type": "theory",
                "difficulty": "medium"
            }
        }
    }


def create_metadata_filter(metadata_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a filter dictionary for metadata-based search.
    
    Args:
        metadata_criteria: Dictionary with metadata fields to filter by
    
    Returns:
        Dictionary suitable for use in RAG query filters
    """
    filter_dict = {}
    
    for field, value in metadata_criteria.items():
        if field in REQUIRED_METADATA_FIELDS or field in OPTIONAL_METADATA_FIELDS:
            # Normalize the value for filtering
            if field == "board":
                # More flexible normalization for board names
                # Handle variations like "TamilNaduStateBoard", "Tamil Nadu State Board", etc.
                normalized = str(value).strip()
                # Replace common separators with underscores
                normalized = normalized.replace("-", "_").replace(".", "_")
                # Handle camelCase by inserting underscores before capitals (but keep consecutive capitals together)
                import re
                # Insert underscore before capital letters (but not if previous char is also capital)
                normalized = re.sub(r'(?<!^)(?<!_)([A-Z])', r'_\1', normalized)
                # Convert to uppercase and replace spaces with underscores
                normalized = normalized.upper().replace(" ", "_")
                # Clean up multiple underscores
                normalized = re.sub(r'_+', '_', normalized).strip('_')
                filter_dict[field] = normalized
            elif field == "grade":
                filter_dict[field] = str(value).strip()
            else:
                # For other fields, normalize but keep original case for subject names
                normalized = str(value).strip()
                filter_dict[field] = normalized
    
    return filter_dict

