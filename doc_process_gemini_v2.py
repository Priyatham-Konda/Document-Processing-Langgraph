# Lovable Multi-Agent Document Processing System with LangGraph
# Vibe: Clean, modular, and a joy to read.

# --- 1. Imports: The essentials for our delightful workflow ---
import uuid
import random
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional, TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END
from tools import (
    get_file_metadata,
    convert_segment_to_image,
    ask_gemini_about_image,
    ask_gemini_text_only,
    get_extraction_prompt,
    set_extraction_prompt,
)
from salesforce_integration import SalesforceProductConnector, process_document_to_salesforce


# --- 2. Data Models: Structuring our world with lovable dataclasses ---

@dataclass
class ExtractionField:
    """A single extracted piece of information from a document."""
    name: str
    value: Any
    confidence: float
    bbox: List[float] = field(default_factory=list)  # [x1, y1, x2, y2]
    page: int = 0
    strategy: str = "mock"

@dataclass
class Segment:
    """Represents a segment of the original document, like a single invoice in a multi-page PDF."""
    segment_id: str
    page_start: int
    page_end: int
    image_path: str = ""  # Path to the saved image file for this segment
    # The 'text' field is removed, as we will now work directly with image data per segment.

@dataclass
class ClassificationResult:
    """The output of the Classifier Agent for a single segment."""
    segment_id: str
    doc_type: str
    vendor: str
    template_id: str
    confidence: float

@dataclass
class ExtractionResult:
    """Structured data extracted from a single document segment."""
    segment_id: str
    doc_type: str
    fields: List[ExtractionField] = field(default_factory=list)

@dataclass
class ValidationResult:
    """Validation verdicts for a single extraction result."""
    segment_id: str
    verdicts: Dict[str, Literal["ok", "warn", "fail"]] = field(default_factory=dict)
    is_fatal: bool = False # A fatal error means instant rejection

@dataclass
class ReviewResult:
    """Results from the human-in-the-loop review simulation."""
    segment_id: str
    corrected_fields: List[ExtractionField] = field(default_factory=list)
    human_feedback: str = "Mock human approved corrections."

@dataclass
class ApplicationResult:
    """The result of attempting to upsert data into an external system (e.g., ERP)."""
    segment_id: str
    success: bool
    external_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class LearningEvent:
    """An event capturing data for model/template improvement."""
    event_type: str # e.g., 'correction', 'rejection'
    segment_id: str
    details: Dict[str, Any]

@dataclass
class Alert:
    """A notification to be sent out."""
    level: Literal["info", "warn", "error"]
    message: str
    recipient: str = "ops-team@example.com"


# --- 3. State Definition: The shared memory of our agent graph ---

class GraphState(TypedDict):
    """The central state that flows through our graph."""
    document_id: str
    file_bytes: bytes
    metadata: Dict[str, Any]
    page_count: int
    needs_split: bool
    segments: List[Segment]
    image_paths: List[str]  # List of all generated image paths
    classifications: List[ClassificationResult]
    extraction_results: List[ExtractionResult]
    validation_results: List[ValidationResult]
    review_results: List[ReviewResult]
    application_results: List[ApplicationResult]
    learning_events: List[LearningEvent]
    alerts: List[Alert]
    final_decision: Optional[Literal["processed", "rejected"]]


# --- 4. Service Integration Placeholders ---
# In a real application, you would import and initialize your external services here.
# For example:
# from .services import ocr_service, llm_service, erp_service, alert_service, learning_service
#
# ocr = ocr_service.OCRService()
# llm = llm_service.LLMService(api_key="...")
# erp = erp_service.ERPService(credentials="...")
# alerts = alert_service.AlertService()
# learning = learning_service.LearningService()


# --- 5. Agent Nodes: The heart of our system, one function at a time ---

def inspection_agent(state: GraphState) -> Dict[str, Any]:
    #Done implementating this
    """Validates the input file and determines if splitting is needed."""
    print("\n--- Running Inspection Node ✨ ---")
    # Honor a pre-assigned document_id if provided by upstream (API), else create one
    doc_id = state.get("document_id") or f"doc_{uuid.uuid4().hex[:6]}"

    incoming_metadata: Dict[str, Any] = state.get("metadata", {}) or {}
    file_bytes: bytes = state.get("file_bytes", b"") or b""

    # Always compute authoritative metadata here using tools.get_file_metadata
    filename = incoming_metadata.get("filename") or incoming_metadata.get("file_name") or "unknown"
    metadata = get_file_metadata(filename, file_bytes)

    mime_type = metadata.get("mime_type") or "application/octet-stream"
    file_extension = metadata.get("file_extension") or ""
    file_size = metadata.get("file_size") or len(file_bytes)

    # Prefer page_count provided by upstream (e.g., PDFs via PyMuPDF). Fallback to 1.
    page_count = int(metadata.get("page_count") or 1)

    print("  Incoming file metadata:")
    print(f"    - filename: {filename}")
    print(f"    - mime_type: {mime_type}")
    print(f"    - file_extension: {file_extension}")
    print(f"    - file_size_bytes: {file_size}")
    if "pdf_info" in metadata and metadata.get("pdf_info"):
        print("    - pdf_info present (keys):", list(metadata["pdf_info"].keys()))
    print(f"    - page_count (derived): {page_count}")

    needs_split = page_count > 1
    print(f"  Document ID: {doc_id}, Pages: {page_count}, Needs Split: {needs_split}")
    return {
        "document_id": doc_id,
        "page_count": page_count,
        "needs_split": needs_split,
        "metadata": metadata
    }

def splitter_agent(state: GraphState) -> Dict[str, Any]:
    #Done implementating this
    """Splits the document into logical segments and converts each to images."""
    print("\n--- Running Splitter Node ✂️ ---")
    
    document_id = state["document_id"]
    file_bytes = state["file_bytes"]
    metadata = state["metadata"]
    file_extension = metadata.get("file_extension", ".pdf")
    
    segments = []
    image_paths = []
    
    if not state["needs_split"]:
        print("  No splitting needed - single page document.")
        # Convert the single page to image
        image_path = convert_segment_to_image(
            file_bytes, document_id, 1, file_extension
        )
        segments = [Segment(
            segment_id=f"{document_id}_seg_1", 
            page_start=1, 
            page_end=state['page_count'],
            image_path=image_path
        )]
        image_paths = [image_path]
    else:
        # Split into one segment per page and convert each to image
        print(f"  Splitting document into {state['page_count']} segments and converting to images.")
        for i in range(state['page_count']):
            page_num = i + 1
            print(f"  Converting page {page_num} to image...")
            
            # Convert this page to an image
            image_path = convert_segment_to_image(
                file_bytes, document_id, page_num, file_extension
            )
            
            segments.append(
                Segment(
                    segment_id=f"{document_id}_seg_{page_num}",
                    page_start=page_num,
                    page_end=page_num,
                    image_path=image_path
                )
            )
            image_paths.append(image_path)
    
    print(f"  Created {len(segments)} segments with images saved.")
    return {
        "segments": segments,
        "image_paths": image_paths
    }

#text extraction agent removed

def classifier_agent(state: GraphState) -> Dict[str, Any]:
    #Done implementating this
    """Classifies each segment to determine its type and vendor using Gemini API."""
    print("\n--- Running Classifier Node 🤖 ---")
    classifications = []
    
    # Get the first segment (first page) for document type classification
    first_segment = state["segments"][0] if state["segments"] else None
    
    if not first_segment or not first_segment.image_path:
        print("  Error: No segments or image paths available for classification")
        return {"classifications": []}
    
    print(f"  Classifying document using first page image: {first_segment.image_path}")
    
    # Create a prompt for document classification
    classification_prompt = """
    Analyze this document image and determine:
    1. What type of document is this? (e.g., Invoice, Purchase Order, Receipt, Contract, etc.)
    2. What is the vendor/company name if visible?
    3. How confident are you in this classification? (rate 0.0 to 1.0)
    
    Please respond in this exact JSON format:
    {
        "doc_type": "document_type_here",
        "vendor": "vendor_name_here", 
        "confidence": 0.95
    }
    
    If you cannot determine the vendor, use "Unknown". Be specific about the document type.
    """
    
    try:
        # Call Gemini API with the first page image
        gemini_response = ask_gemini_about_image(first_segment.image_path, classification_prompt)
        print(f"  Gemini API Response: {gemini_response}")
        
        # Parse the JSON response
        import json
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = gemini_response.find('{')
            json_end = gemini_response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = gemini_response[json_start:json_end]
                result_dict = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Error parsing Gemini response as JSON: {e}")
            # Fallback: try to extract document type from text response
            response_lower = gemini_response.lower()
            if "invoice" in response_lower:
                result_dict = {"doc_type": "invoice", "vendor": "Unknown", "confidence": 0.7}
            elif "purchase order" in response_lower or "po" in response_lower:
                result_dict = {"doc_type": "purchase_order", "vendor": "Unknown", "confidence": 0.7}
            elif "receipt" in response_lower:
                result_dict = {"doc_type": "receipt", "vendor": "Unknown", "confidence": 0.7}
            else:
                result_dict = {"doc_type": "unknown", "vendor": "Unknown", "confidence": 0.3}
        
        # Generate template_id based on doc_type
        doc_type = result_dict.get("doc_type", "unknown")
        template_id = f"TPL_{doc_type.upper().replace(' ', '_')}"
        result_dict["template_id"] = template_id
        
        print(f"  Parsed result: {result_dict}")
        
    except Exception as e:
        print(f"  Error calling Gemini API: {e}")
        result_dict = {"doc_type": "unknown", "vendor": "Unknown", "template_id": "TPL_UNKNOWN", "confidence": 0.1}
    
    # Apply the same classification to all segments (since it's the same document)
    for segment in state["segments"]:
        classifications.append(ClassificationResult(
            segment_id=segment.segment_id,
            doc_type=result_dict.get("doc_type", "unknown"),
            vendor=result_dict.get("vendor", "Unknown"),
            template_id=result_dict.get("template_id", "TPL_UNKNOWN"),
            confidence=result_dict.get("confidence", 0.1)
        ))

    print(f"  Classifications complete: {[c.doc_type for c in classifications]}")
    return {"classifications": classifications}

def extraction_agent(state: GraphState) -> Dict[str, Any]:
    #   Done implementation
    # LLM should be called attaching the segment-wise image data along with a prompt and a target JSON schema.
    # The LLM should be prompted to calculate and return a confidence score for every extracted field.
    """Extracts structured data from each classified segment using Gemini on per-page images."""
    print("\n--- Running Extraction Node 🧩 ---")

    # Build a quick lookup for doc_type by segment_id from classifier outputs
    segment_id_to_doc_type: Dict[str, str] = {
        c.segment_id: c.doc_type for c in state.get("classifications", [])
    }

    extraction_results: List[ExtractionResult] = []
    
    # JSON logging setup
    import json
    from pathlib import Path
    from datetime import datetime
    document_id = state.get("document_id", "unknown")
    json_log_path = Path(f"extraction_logs/{document_id}_extraction_log.json")
    json_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Structured outputs folder setup
    structured_outputs_dir = Path(f"document_images/{document_id}/structured_outputs")
    structured_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing log or create new one
    try:
        with open(json_log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = {
            "document_id": document_id,
            "extraction_timestamp": "",
            "segments": []
        }

    for segment in state.get("segments", []):
        image_path = getattr(segment, "image_path", "")
        if not image_path:
            print(f"  Skipping segment {segment.segment_id}: missing image path")
            extraction_results.append(ExtractionResult(segment_id=segment.segment_id, doc_type=segment_id_to_doc_type.get(segment.segment_id, "unknown"), fields=[]))
            continue

        doc_type_for_segment = segment_id_to_doc_type.get(segment.segment_id, "unknown")
        print(f"  Extracting from segment {segment.segment_id} (type: {doc_type_for_segment}) using image: {image_path}")

        # Dynamic prompt: base prompt + required schema and document context
        base_prompt = get_extraction_prompt()
        prompt = f"""
        {base_prompt}

        The document type is: {doc_type_for_segment}.
        Return ONLY valid JSON (no markdown, no explanations) in this exact schema:
        {{
          "segment_id": "{segment.segment_id}",
          "doc_type": "{doc_type_for_segment}",
          "fields": [
            {{ "name": "string (use dot-notation for nested)", "value": "string|number|boolean", "confidence": 0.0 }}
          ]
        }}
        """

        try:
            print(f"  Calling Gemini API for {segment.segment_id}...")
            print(f"  Image path exists: {Path(image_path).exists()}")
            print(f"  Image path: {image_path}")
            
            response_text = ask_gemini_about_image(image_path, prompt)
            print(f"  Gemini API call completed for {segment.segment_id}")
            print(f"  Response length: {len(response_text) if response_text else 0}")
            print(f"  Gemini extraction raw response for {segment.segment_id}: {response_text[:200] if response_text else 'None'}...")

            # Log the raw response
            segment_log = {
                "segment_id": segment.segment_id,
                "image_path": str(image_path),
                "doc_type": doc_type_for_segment,
                "raw_response": response_text,
                "parsed_successfully": False,
                "extracted_fields": []
            }

            # Parse JSON, even if wrapped by extra text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                segment_log["parsed_successfully"] = True
                
                # Save structured JSON for this segment
                segment_image_name = Path(image_path).stem  # Get filename without extension
                structured_json_path = structured_outputs_dir / f"{segment_image_name}-structured.json"
                
                # Create structured output with metadata
                structured_output = {
                    "segment_id": segment.segment_id,
                    "document_id": document_id,
                    "doc_type": doc_type_for_segment,
                    "image_path": str(image_path),
                    "extraction_timestamp": datetime.now().isoformat(),
                    "raw_gemini_response": response_text,
                    "parsed_data": parsed
                }
                
                try:
                    with open(structured_json_path, 'w', encoding='utf-8') as f:
                        json.dump(structured_output, f, indent=2, ensure_ascii=False)
                    print(f"  Structured JSON saved: {structured_json_path}")
                    segment_log["structured_json_path"] = str(structured_json_path)
                except Exception as e:
                    print(f"  Warning: Could not save structured JSON for {segment.segment_id}: {e}")
                    segment_log["structured_json_path"] = None
            else:
                raise ValueError("No JSON found in response")

            # Normalize into list[ExtractionField]
            fields_data = parsed.get("fields") if isinstance(parsed, dict) else None
            fields: List[ExtractionField] = []
            if isinstance(fields_data, list):
                for f in fields_data:
                    try:
                        name = str(f.get("name", "unknown"))
                        value = f.get("value")
                        confidence_raw = f.get("confidence", 0.7)
                        try:
                            confidence = float(confidence_raw)
                        except Exception:
                            confidence = 0.7
                        # Clamp confidence to [0,1]
                        if confidence < 0.0: confidence = 0.0
                        if confidence > 1.0: confidence = 1.0
                        field_obj = ExtractionField(name=name, value=value, confidence=confidence)
                        fields.append(field_obj)
                        # Log each field
                        segment_log["extracted_fields"].append({
                            "name": name,
                            "value": value,
                            "confidence": confidence
                        })
                    except Exception as e:
                        print(f"    Error processing field: {e}")
                        continue
            else:
                # Fallback: if model returned a flat dict, coerce to fields
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if k in ("segment_id", "doc_type"):  # skip meta keys
                            continue
                        field_obj = ExtractionField(name=str(k), value=v, confidence=0.7)
                        fields.append(field_obj)
                        segment_log["extracted_fields"].append({
                            "name": str(k),
                            "value": v,
                            "confidence": 0.7
                        })

            extraction_results.append(
                ExtractionResult(segment_id=segment.segment_id, doc_type=doc_type_for_segment, fields=fields)
            )
            
        except Exception as e:
            print(f"  Extraction failed for {segment.segment_id}: {e}")
            segment_log = {
                "segment_id": segment.segment_id,
                "image_path": str(image_path),
                "doc_type": doc_type_for_segment,
                "raw_response": f"Error: {str(e)}",
                "parsed_successfully": False,
                "extracted_fields": []
            }
            
            # Still save structured JSON even for failed extractions
            segment_image_name = Path(image_path).stem if image_path else segment.segment_id
            structured_json_path = structured_outputs_dir / f"{segment_image_name}-structured.json"
            
            structured_output = {
                "segment_id": segment.segment_id,
                "document_id": document_id,
                "doc_type": doc_type_for_segment,
                "image_path": str(image_path) if image_path else "",
                "extraction_timestamp": datetime.now().isoformat(),
                "raw_gemini_response": f"Error: {str(e)}",
                "parsed_data": None,
                "error": str(e)
            }
            
            try:
                with open(structured_json_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_output, f, indent=2, ensure_ascii=False)
                print(f"  Error JSON saved: {structured_json_path}")
                segment_log["structured_json_path"] = str(structured_json_path)
            except Exception as save_error:
                print(f"  Warning: Could not save error JSON for {segment.segment_id}: {save_error}")
                segment_log["structured_json_path"] = None
            
            extraction_results.append(
                ExtractionResult(segment_id=segment.segment_id, doc_type=doc_type_for_segment, fields=[])
            )
        
        # Append segment log to the main log data
        log_data["segments"].append(segment_log)

    # Update timestamp and save the complete log
    log_data["extraction_timestamp"] = datetime.now().isoformat()
    log_data["total_segments"] = len(extraction_results)
    log_data["successful_extractions"] = sum(1 for seg in log_data["segments"] if seg["parsed_successfully"])
    
    try:
        with open(json_log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"  Extraction log saved to: {json_log_path}")
    except Exception as e:
        print(f"  Warning: Could not save extraction log: {e}")

    print(f"  Extraction complete for {len(extraction_results)} segments.")
    return {"extraction_results": extraction_results}

def validator_agent(state: GraphState) -> Dict[str, Any]:
    #   Done implementation 
    """Confidence-based validation - use extraction confidence scores."""
    print("\n--- Running Confidence-Based Validator Node ⚡ ---")
    
    validation_results = []
    
    # Confidence thresholds
    FAIL_THRESHOLD = 0.3
    WARN_THRESHOLD = 0.7
    
    for result in state["extraction_results"]:
        verdicts = {}
        
        for field in result.fields:
            if field.confidence < FAIL_THRESHOLD:
                verdicts[field.name] = "fail"
            elif field.confidence < WARN_THRESHOLD:
                verdicts[field.name] = "warn"
            else:
                verdicts[field.name] = "ok"
        
        is_fatal = any(v == "fail" for v in verdicts.values())
        
        validation_results.append(
            ValidationResult(
                segment_id=result.segment_id,
                verdicts=verdicts,
                is_fatal=is_fatal
            )
        )
    
    print(f"  ⚡ Lightning-fast validation complete!")
    return {"validation_results": validation_results}


def reviewer_agent(state: GraphState) -> Dict[str, Any]:
    #   Done implementation
    """Identifies fields needing human review and prepares for frontend correction."""
    print("\n--- Running Reviewer Node 👀 ---")
    T_REVIEW_LOW = 0.60
    T_AUTO = 0.95

    review_results = []
    learning_events = state.get("learning_events", [])
    
    # Setup review data collection
    from pathlib import Path
    from datetime import datetime
    import json
    document_id = state.get("document_id", "unknown")
    review_data_dir = Path(f"review_data/{document_id}")
    review_data_dir.mkdir(parents=True, exist_ok=True)

    for extraction in state["extraction_results"]:
        print(f"  Analyzing segment {extraction.segment_id} (type: {extraction.doc_type})...")
        
        # Identify fields that need review
        fields_to_review = [f for f in extraction.fields if T_REVIEW_LOW <= f.confidence < T_AUTO]
        
        if not fields_to_review:
            print(f"    ✅ No fields need review (all confidences >= {T_AUTO})")
            # Record a review result indicating no review needed
            review_results.append(ReviewResult(
                segment_id=extraction.segment_id,
                corrected_fields=[],
                human_feedback="No review needed - all fields high confidence"
            ))
            # Create an empty review JSON placeholder so API returns {}
            review_file = review_data_dir / f"{extraction.segment_id}_review.json"
            try:
                with open(review_file, 'w', encoding='utf-8') as f:
                    f.write("{}")
                print(f"    📝 Empty review placeholder saved: {review_file}")
            except Exception as e:
                print(f"    ⚠️  Could not save empty review placeholder: {e}")
            continue

        print(f"    🔍 {len(fields_to_review)} fields need human review:")
        for field in fields_to_review:
            print(f"      - {field.name}: '{field.value}' (confidence: {field.confidence:.2f})")

        # Prepare review data for frontend
        review_data = {
            "segment_id": extraction.segment_id,
            "doc_type": extraction.doc_type,
            "fields_needing_review": [
                {
                    "field_name": field.name,
                    "current_value": field.value,
                    "current_confidence": field.confidence,
                    "field_type": type(field.value).__name__,
                    "needs_correction": True
                }
                for field in fields_to_review
            ],
            "all_fields": [
                {
                    "field_name": field.name,
                    "current_value": field.value,
                    "current_confidence": field.confidence,
                    "field_type": type(field.value).__name__,
                    "needs_correction": field in fields_to_review
                }
                for field in extraction.fields
            ],
            "review_status": "pending",
            "created_timestamp": datetime.now().isoformat(),
            "updated_timestamp": None
        }

        # Save review data for frontend
        review_file = review_data_dir / f"{extraction.segment_id}_review.json"
        try:
            with open(review_file, 'w', encoding='utf-8') as f:
                json.dump(review_data, f, indent=2, ensure_ascii=False)
            print(f"    📝 Review data saved: {review_file}")
        except Exception as e:
            print(f"    ⚠️  Could not save review data: {e}")

        # Create review result (fields will be updated by frontend later)
        review_results.append(ReviewResult(
            segment_id=extraction.segment_id,
            corrected_fields=[],  # Will be populated by frontend corrections
            human_feedback=f"Review pending for {len(fields_to_review)} fields"
        ))

        # Create learning event for tracking
        learning_events.append(LearningEvent(
            event_type='review_required',
            segment_id=extraction.segment_id,
            details={
                'fields_count': len(fields_to_review),
                'review_file': str(review_file),
                'timestamp': datetime.now().isoformat()
            }
        ))

    print(f"  Review analysis complete for {len(review_results)} segments.")
    print(f"  Review files created in: {review_data_dir}")
    
    # Pass through essential state so downstream consumers (API layer) have full context
    return {
        "document_id": state.get("document_id"),
        "segments": state.get("segments", []),
        "classifications": state.get("classifications", []),
        "extraction_results": state.get("extraction_results", []),
        "metadata": state.get("metadata", {}),
        "alerts": state.get("alerts", []),
        "thresholds": {"review_low": T_REVIEW_LOW, "auto": T_AUTO},
        "review_results": review_results,
        "learning_events": learning_events
    }


def apply_human_corrections(state_or_document_id, corrections: Dict[str, Any]) -> Dict[str, Any]:
    # External function - not part of graph
    # Not required, need to be removed
    """Applies human corrections from frontend to extraction results and saves learning data."""
    print("\n--- Applying Human Corrections 🔧 ---")
    
    from pathlib import Path
    from datetime import datetime
    import json
    
    # Support being called with either full state or just a document_id (from API)
    if isinstance(state_or_document_id, dict):
        state = state_or_document_id
        document_id = state.get("document_id", "unknown")
    else:
        state = None
        document_id = str(state_or_document_id)
    learning_data_dir = Path(f"learning_data/{document_id}")
    learning_data_dir.mkdir(parents=True, exist_ok=True)
    
    learning_events = (state.get("learning_events", []) if isinstance(state, dict) else [])
    corrected_extractions = []
    
    if not isinstance(state, dict):
        # API-mode: no in-memory state to mutate. Persist corrections only.
        from datetime import datetime
        import json
        # Save one combined corrections file
        learning_file = learning_data_dir / f"document_human_corrections.json"
        learning_data = {
            "document_id": document_id,
            "corrections": corrections,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"    ⚠️  Could not save learning data: {e}")
        return {
            "status": "success",
            "document_id": document_id,
            "corrections": corrections
        }

    for extraction in state["extraction_results"]:
        segment_id = extraction.segment_id
        
        # Check if there are corrections for this segment
        if segment_id not in corrections:
            corrected_extractions.append(extraction)
            continue
            
        print(f"  Applying corrections for segment {segment_id}...")
        
        segment_corrections = corrections[segment_id]
        corrected_fields = []
        learning_corrections = []
        
        # Apply corrections to fields
        for field in extraction.fields:
            field_name = field.name
            
            if field_name in segment_corrections:
                correction = segment_corrections[field_name]
                corrected_value = correction.get("corrected_value", field.value)
                # corrected_confidence = correction.get("corrected_confidence", field.confidence)
                
                # Create corrected field
                corrected_field = ExtractionField(
                    name=field_name,
                    value=corrected_value,
                    confidence=field.confidence
                )
                corrected_fields.append(corrected_field)
                
                # Record for learning
                learning_corrections.append({
                    "field_name": field_name,
                    "original_value": field.value,
                    "original_confidence": field.confidence,
                    "corrected_value": corrected_value,
                    # "corrected_confidence": corrected_confidence,
                    "human_feedback": correction.get("human_feedback", ""),
                    "timestamp": datetime.now().isoformat()
                })
                
                print(f"    ✅ {field_name}: '{field.value}' → '{corrected_value}' (conf: {field.confidence:.2f})")
            else:
                corrected_fields.append(field)
        
        # Update extraction with corrected fields
        extraction.fields = corrected_fields
        corrected_extractions.append(extraction)
        
        # Save learning data
        if learning_corrections:
            learning_file = learning_data_dir / f"{segment_id}_human_corrections.json"
            learning_data = {
                "segment_id": segment_id,
                "doc_type": extraction.doc_type,
                "corrections": learning_corrections,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                with open(learning_file, 'w', encoding='utf-8') as f:
                    json.dump(learning_data, f, indent=2, ensure_ascii=False)
                print(f"    📚 Learning data saved: {learning_file}")
            except Exception as e:
                print(f"    ⚠️  Could not save learning data: {e}")
            
            # Create learning event
            learning_events.append(LearningEvent(
                event_type='human_correction',
                segment_id=segment_id,
                details={
                    'corrections_count': len(learning_corrections),
                    'learning_file': str(learning_file),
                    'timestamp': datetime.now().isoformat()
                }
            ))
    
    print(f"  Human corrections applied to {len(corrected_extractions)} segments.")
    
    return {
        "extraction_results": corrected_extractions,
        "learning_events": learning_events
    }

# def application_agent(state: GraphState) -> Dict[str, Any]:
#     # Common plug for any ERP system; can be adapted for Salesforce, SAP, etc.
#     # Using MCP is another option, but we will use this for now, will integrate Salesforce.
#     """Upserts the final, validated data into the target system."""
#     print("\n--- Running Application Node 🚀 ---")
#     application_results = []
#     alerts = state.get("alerts", [])
#     for extraction in state["extraction_results"]:
#         # TODO: Implement real ERP upsert logic here.
#         # Example: result = erp.upsert(extraction)
#         print(f"  (Placeholder) Upserting data for segment {extraction.segment_id}...")
#         result = ApplicationResult(
#             segment_id=extraction.segment_id,
#             success=True,
#             external_id=f"ERP-{uuid.uuid4().hex[:10].upper()}"
#         )

#         application_results.append(result)
#         if not result.success:
#             alerts.append(Alert(
#                 level="error",
#                 message=f"ERP Upsert Failed for segment {result.segment_id}: {result.error_message}"
#             ))

#     print("  Application upsert simulation complete.")
#     return {
#         "application_results": application_results,
#         "alerts": alerts,
#         "final_decision": "processed"
#     }

def application_agent(state: GraphState) -> Dict[str, Any]:
    """Simplified application agent for Product2 integration only."""
    print("\n--- Running Application Node 🚀 (Product2 Only) ---")
    
    # Initialize Salesforce Product connector
    sf_connector = SalesforceProductConnector(
        instance_url=os.getenv('SF_INSTANCE_URL', 'https://your-instance.salesforce.com'),
        email=os.getenv('SF_EMAIL'),
        password=os.getenv('SF_PASSWORD'),
        security_token=os.getenv('SF_SECURITY_TOKEN')
    )
    
    application_results = []
    alerts = state.get("alerts", [])
    
    # Test connection
    try:
        if not sf_connector.test_connection():
            raise Exception("Failed to connect to Salesforce")
        
        print("  ✅ Connected to Salesforce successfully")
        
    except Exception as e:
        error_msg = f"Salesforce connection failed: {str(e)}"
        print(f"  ❌ {error_msg}")
        
        # Return failure for all segments
        for extraction in state["extraction_results"]:
            application_results.append(ApplicationResult(
                segment_id=extraction.segment_id,
                success=False,
                error_message=error_msg
            ))
        
        alerts.append(Alert(level="error", message=error_msg))
        
        return {
            "application_results": application_results,
            "alerts": alerts,
            "final_decision": "rejected"
        }
    
    # Process each extraction result
    for extraction in state["extraction_results"]:
        segment_id = extraction.segment_id
        print(f"  Processing segment {segment_id} for Product2 integration...")
        
        try:
            # Convert extraction fields to dictionary and normalize complex values to strings
            def _normalize(v: Any) -> Any:
                if isinstance(v, (dict, list, tuple)):
                    try:
                        import json as _json
                        return _json.dumps(v, ensure_ascii=False)
                    except Exception:
                        return str(v)
                return v
            fields_dict = {field.name: _normalize(field.value) for field in extraction.fields}
            
            # Process document to Product2
            result = process_document_to_salesforce(
                document_data=fields_dict,
                document_id=segment_id,
                sf_connector=sf_connector
            )
            
            if result['success']:
                application_result = ApplicationResult(
                    segment_id=segment_id,
                    success=True,
                    external_id=result.get('sku')
                )
                print(f"    ✅ Product created/updated: {result.get('sku')}")
            else:
                application_result = ApplicationResult(
                    segment_id=segment_id,
                    success=False,
                    error_message=result.get('error')
                )
                alerts.append(Alert(
                    level="error",
                    message=f"Product2 Integration Failed for {segment_id}: {result.get('error')}"
                ))
                print(f"    ❌ Integration failed: {result.get('error')}")
            
            application_results.append(application_result)
            
        except Exception as e:
            error_msg = f"Error processing {segment_id}: {str(e)}"
            print(f"    ❌ {error_msg}")
            
            application_results.append(ApplicationResult(
                segment_id=segment_id,
                success=False,
                error_message=error_msg
            ))
            
            alerts.append(Alert(level="error", message=error_msg))
    
    success_count = sum(1 for result in application_results if result.success)
    print(f"  ✅ Product2 integration complete: {success_count}/{len(application_results)} segments successful")
    
    return {
        "application_results": application_results,
        "alerts": alerts,
        "final_decision": "processed" if success_count > 0 else "rejected"
    }


def learning_agent(state: GraphState) -> Dict[str, Any]:
    """Reads human corrections and refines the extraction prompt using an LLM."""
    print("\n--- Running Learning Node 🧠 ---")
    document_id = state.get("document_id", "unknown")
    learning_dir = Path(f"learning_data/{document_id}")
    if not learning_dir.exists():
        print("  No learning data directory for this document.")
        return {}

    # Gather human correction items across segment correction files
    import json
    corrections_summary: List[Dict[str, Any]] = []
    try:
        # Read per-segment review updates saved by API: *_updated_review.json
        for file_path in learning_dir.glob("*_updated_review.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc_type_val = data.get("doc_type") or "unknown"
                    # Support both keys: updated_fields (from review-submit) and corrections (older format)
                    items = data.get("updated_fields") or data.get("corrections") or []
                    for c in items:
                        corrections_summary.append({
                            "doc_type": doc_type_val,
                            "field_name": c.get("field_name"),
                            "original_value": c.get("original_value"),
                            "corrected_value": c.get("updated_value") if "updated_value" in c else c.get("corrected_value"),
                        })
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")

        # Also read in-memory corrections saved by apply_human_corrections: *_human_corrections.json
        for file_path in learning_dir.glob("*_human_corrections.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc_type_val = data.get("doc_type") or "unknown"
                    for c in data.get("corrections", []):
                        corrections_summary.append({
                            "doc_type": doc_type_val,
                            "field_name": c.get("field_name"),
                            "original_value": c.get("original_value"),
                            "corrected_value": c.get("corrected_value"),
                        })
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")

        # Additionally read human overrides written by API: human_overrides_*.json
        for file_path in learning_dir.glob("human_overrides_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for c in data.get("corrections", []):
                        corrections_summary.append({
                            "doc_type": "unknown",
                            "field_name": c.get("field_name"),
                            "original_value": c.get("original_value"),
                            "corrected_value": c.get("corrected_value"),
                        })
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
    except Exception as e:
        print(f"  Error reading learning data: {e}")

    if not corrections_summary:
        print("  No human corrections found to learn from.")
        return {}

    # Build an instruction to refine the base extraction prompt
    base_prompt = get_extraction_prompt()
    improvement_instructions = [
        "You are refining an extraction prompt to reduce future errors.",
        "Use the following human corrections to add targeted rules.",
        "Focus on clarifying field naming, normalization, and disambiguation.",
        "Do not include any example data in the final prompt; write generalizable rules.",
    ]

    # Prepare a compact JSON list of corrections (doc_type, field_name, original_value, corrected_value)
    import json as _json
    corrections_json = _json.dumps(corrections_summary[:100], ensure_ascii=False, indent=2)

    refinement_prompt = (
        "\n".join(improvement_instructions)
        + f"\n\nCurrent base prompt:\n---\n{base_prompt}\n---\n"
        + "\nHuman corrections (samples):\n"
        + corrections_json
        + "\n\nTask: Rewrite and improve the base prompt. Keep structure, add precise rules to avoid the above mistakes.\n"
        + "Return only the improved prompt text with no explanations."
    )

    try:
        print(f"  Learning: using {len(corrections_summary)} corrections to refine prompt")
        updated_prompt = ask_gemini_text_only(refinement_prompt)

        # Guard: if the provider returned an error-like payload as text, do not update
        def _looks_like_error(text: str) -> bool:
            if not text:
                return True
            lowered = text.lower()
            if '"error"' in lowered or 'status":' in lowered or 'unavailable' in lowered:
                return True
            if lowered.strip().startswith('{') and 'error' in lowered:
                return True
            return False

        if isinstance(updated_prompt, str) and not _looks_like_error(updated_prompt) and len(updated_prompt.strip()) > 50:
            set_extraction_prompt(updated_prompt)
            print("  ✅ Extraction prompt updated from learning data.")
        else:
            print("  ⚠️ LLM returned an insufficient or error-like prompt; keeping existing prompt.")
    except Exception as e:
        # Explicitly avoid updating the prompt when provider errors (e.g., 503)
        print(f"  ⚠️ Prompt refinement failed (provider error). Keeping existing prompt. Details: {e}")

    return {}

def alert_agent(state: GraphState) -> Dict[str, Any]:
    """Sends alerts based on workflow outcomes."""
    print("\n--- Running Alert Node 🔔 ---")
    alerts = state.get("alerts", [])
    if alerts:
        for alert in alerts:
            # TODO: Implement real alert sending logic (e.g., email, Slack API call).
            # Example: alerts.send(alert)
            print(f"  🚨 (Placeholder) Sending {alert.level.upper()} alert to {alert.recipient}: '{alert.message}'")
    else:
        print("  No alerts to send.")
    return {} # No state change needed

def rejection_node(state: GraphState) -> Dict[str, Any]:
    """Handles rejected documents, creating alerts and learning events."""
    print("\n--- Running Rejection Node ❌ ---")
    alerts = state.get("alerts", [])
    learning_events = state.get("learning_events", [])
    alerts.append(Alert(
        level="error",
        message=f"Document {state['document_id']} was rejected due to fatal validation errors or low confidence."
    ))
    learning_events.append(LearningEvent(
        event_type='rejection',
        segment_id="document-wide",
        details={'reason': 'Fatal validation error or confidence below rejection threshold.'}
    ))
    return {"alerts": alerts, "learning_events": learning_events, "final_decision": "rejected"}


# --- 6. Routing Logic: The decision-maker of our graph ---

def route_after_validation(state: GraphState) -> Literal["auto_approve", "review_needed", "reject"]:
    """Simplified router - relies on validation agent's decisions."""
    print("\n--- Router: Deciding based on validation outcomes... ---")
    
    # 1. Check for fatal validation errors (immediate reject)
    if any(vr.is_fatal for vr in state["validation_results"]):
        print("  Decision: REJECT (fatal validation errors detected).")
        return "reject"
    
    # 2. Check for any warnings (needs human review)
    has_warnings = False
    for validation_result in state["validation_results"]:
        if any(verdict == "warn" for verdict in validation_result.verdicts.values()):
            has_warnings = True
            break
    
    if has_warnings:
        print("  Decision: REVIEW NEEDED (validation warnings detected).")
        return "review_needed"
    
    # 3. All validations passed - auto approve
    print("  Decision: AUTO APPROVE (all validations OK).")
    return "auto_approve"



# --- 7. Graph Construction: Assembling our team of agents ---

def build_graph():
    """Builds and configures the LangGraph StateGraph."""
    workflow = StateGraph(GraphState)

    # Add all our agent nodes to the graph
    workflow.add_node("inspection", inspection_agent)
    workflow.add_node("splitter", splitter_agent)
    # The text_extraction_node is now removed.
    workflow.add_node("classifier", classifier_agent)
    workflow.add_node("extractor", extraction_agent)
    workflow.add_node("validator", validator_agent)
    workflow.add_node("reviewer", reviewer_agent)
    # Application, rejection, learning and alerting nodes are disconnected for now
    # to support human approval gating.

    # Set the entry point of the workflow
    workflow.set_entry_point("inspection")

    # Define the standard, linear flow
    workflow.add_edge("inspection", "splitter")
    # The splitter now connects directly to the classifier
    workflow.add_edge("splitter", "classifier")
    workflow.add_edge("classifier", "extractor")
    workflow.add_edge("extractor", "validator")

    # Add the conditional branching logic after validation
    # Route both paths to reviewer so the graph halts for human approval.
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "auto_approve": "reviewer",
            "review_needed": "reviewer",
            "reject": "reviewer",  # do not reject for now; await human
        }
    )

    # Define flows after the branches
    # End after reviewer until human approval happens via API
    # After human approval (triggered via API), we recommend calling learning_agent
    # separately at the API layer once application_agent has run. Keeping END here. 
    workflow.add_edge("reviewer", END)

    # Compile the graph into a runnable app
    return workflow.compile()


# --- 8. Example Runner: Let's see the magic happen! ---

def run_example_flow():
    """Initializes and runs the graph with mock data, printing each step."""
    print("🚀 Starting Agentic OCR Workflow Example 🚀")
    
    app = build_graph()

    # Mock input
    initial_state = {
        "file_bytes": b"this is a mock pdf file",
        "metadata": {"filename": "multi_doc.pdf"}
    }
    
    # Stream the events to see the flow step-by-step
    for event in app.stream(initial_state):
        # The key of the dictionary is the name of the node that just ran
        node_name = list(event.keys())[0]
        state_snapshot = event[node_name]
        print(f"\n✅ Finished Node: {node_name}")
        print("   State Snapshot:")
        # Pretty print relevant parts of the state
        for key, value in state_snapshot.items():
             if value: # Only print keys with values
                print(f"     - {key}: {value}")

    print("\n🏁 Agentic OCR Workflow Complete! 🏁")


if __name__ == "__main__":
    run_example_flow()


