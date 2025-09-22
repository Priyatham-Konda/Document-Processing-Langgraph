from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import io
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from pydantic import BaseModel
from datetime import datetime
import uuid
import mimetypes
import time

# Import your LangGraph workflow
from doc_process_gemini_v2 import build_graph, apply_human_corrections
from doc_process_gemini_v2 import application_agent
from doc_process_gemini_v2 import learning_agent
from redis_state import set_job_status, set_job_state, get_job_state, update_job_state
from tasks import process_document_task



app = FastAPI(
    title="Document Processing System",
    description="AI-powered document processing with LangGraph workflow and human review",
    version="2.0.0"
)

# CORS - allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = "temp_uploads"

# Ensure folders exist
for folder in [UPLOAD_FOLDER, "review_data", "document_images", "extraction_logs", "learning_data", "raw documents"]:
    os.makedirs(folder, exist_ok=True)

# Removed in-memory review sessions; state is persisted in Redis per document_id

# Supported file types
ALLOWED_EXTENSIONS = {
    'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif',
    'webp', 'svg', 'doc', 'docx', 'txt'
}

# Pydantic models
class FileProcessingResult(BaseModel):
    filename: str
    status: str
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    workflow_results: Dict[str, Any] = {}
    review_needed: bool = False
    error: Optional[str] = None

class ProcessingResponse(BaseModel):
    status: str
    processed_files: int
    results: List[FileProcessingResult]

class HumanCorrectionRequest(BaseModel):
    document_id: str
    corrections: Dict[str, Dict[str, Any]]  # segment_id -> field_name -> correction_data

class ReviewStatus(BaseModel):
    document_id: str
    status: str
    segments: List[Dict[str, Any]]

# Utility: JSON-safe serialization for workflow results
def _to_jsonable(obj):
    from dataclasses import is_dataclass, asdict
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # Datetime-like
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    try:
        import json as _json
        return _json.loads(_json.dumps(obj))
    except Exception:
        return str(obj)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def convert_to_bytes(file_content: bytes, filename: str) -> bytes:
    """Convert various file types to bytes for processing."""
    file_extension = Path(filename).suffix.lower()
    
    try:
        if file_extension == '.pdf':
            return file_content
        elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp']:
            with Image.open(io.BytesIO(file_content)) as img:
                if img.mode in ('RGBA', 'P'):
                    img.convert('RGB')
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=95)
                return img_bytes.getvalue()
        else:
            return file_content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error converting file to bytes: {str(e)}")

# Enhanced HTML template with full editing interface
UPLOAD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Processing System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
        }
        
        .header h1 { margin-bottom: 10px; font-size: 2.5rem; }
        .header p { opacity: 0.9; font-size: 1.1rem; }
        
        .upload-section {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .upload-area { 
            border: 2px dashed #ccc; 
            padding: 40px; 
            text-align: center; 
            border-radius: 8px; 
            transition: all 0.3s ease;
            margin: 20px 0;
        }
        
        .upload-area.dragover { 
            border-color: #007bff; 
            background-color: #f0f8ff; 
            transform: scale(1.02);
        }
        
        .btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 6px;
            font-size: 1rem;
            transition: all 0.3s ease;
            margin: 5px;
        }
        
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,123,255,0.3); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .btn.success { background: linear-gradient(135deg, #28a745, #20c997); }
        .btn.warning { background: linear-gradient(135deg, #ffc107, #fd7e14); }
        .btn.danger { background: linear-gradient(135deg, #dc3545, #c82333); }
        
        .result {
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .result.error { background-color: #f8d7da; color: #721c24; border-color: #dc3545; }
        .result.success { background-color: #d4edda; color: #155724; border-color: #28a745; }
        .result.processing { background-color: #fff3cd; color: #856404; border-color: #ffc107; }
        
        .file-info {
            margin: 10px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }
        
        .processing-results {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .document-card {
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin: 15px 0;
            overflow: hidden;
        }
        
        .document-header {
            background-color: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .document-content {
            padding: 20px;
        }
        
        .segment-card {
            margin: 15px 0;
            border: 1px solid #e9ecef;
            border-radius: 6px;
        }
        
        .segment-header {
            background-color: #f1f3f4;
            padding: 12px 15px;
            font-weight: 600;
            border-bottom: 1px solid #e9ecef;
        }
        
        .fields-container {
            padding: 15px;
        }
        
        .field-editor {
            margin: 10px 0;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            display: flex;
            align-items: center;
            gap: 15px;
            transition: all 0.3s ease;
        }
        
        .field-editor:hover {
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .field-editor.needs-review {
            background-color: #fff8e1;
            border-left: 4px solid #ff9800;
        }
        
        .field-editor.high-confidence {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
        }
        
        .field-label {
            min-width: 180px;
            font-weight: 600;
            color: #495057;
        }
        
        .field-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .field-input:focus {
            border-color: #007bff;
            outline: none;
            box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
        }
        
        .field-input.changed {
            border-color: #28a745;
            background-color: #f8fff8;
        }
        
        .confidence-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            min-width: 80px;
            text-align: center;
        }
        
        .confidence-high { background-color: #d4edda; color: #155724; }
        .confidence-medium { background-color: #fff3cd; color: #856404; }
        .confidence-low { background-color: #f8d7da; color: #721c24; }
        
        .review-flag {
            color: #ff6b35;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .actions {
            margin-top: 20px;
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .stats {
            display: flex;
            gap: 20px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        
        .stat {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 10px 15px;
            border-radius: 6px;
            font-weight: 600;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        
        .modal-content {
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 8px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover { color: black; }
        
        @media (max-width: 768px) {
            body { padding: 10px; }
            .header h1 { font-size: 2rem; }
            .field-editor { flex-direction: column; align-items: stretch; }
            .field-label { min-width: auto; }
        }
    </style>
</head>
<body>
    <div class="header">
    <h1>🚀 Document Processing System</h1>
        <p>AI-powered document processing with human review and correction capabilities</p>
    </div>
    
    <div class="upload-section">
        <h2>📁 Upload Documents</h2>
    <div class="upload-area" id="uploadArea">
            <p>📄 Drag and drop your files here, or click to select</p>
        <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.gif,.bmp,.tiff,.webp,.doc,.docx,.txt" multiple style="display: none;">
        <button type="button" class="btn" onclick="document.getElementById('fileInput').click()">Choose Files</button>
    </div>
    
    <div id="fileList"></div>
    
        <div style="text-align: center; margin-top: 20px;">
            <button type="button" class="btn" id="processBtn" onclick="processFiles()" disabled>
                🔄 Process Documents
            </button>
        </div>
    </div>
    
    <div id="results"></div>
    
    <!-- Modal for detailed editing -->
    <div id="editModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="modalContent"></div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileList = document.getElementById('fileList');
        const processBtn = document.getElementById('processBtn');
        const results = document.getElementById('results');
        let selectedFiles = [];
        let currentDocumentData = {};

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFiles(Array.from(e.dataTransfer.files));
        });

        fileInput.addEventListener('change', (e) => {
            handleFiles(Array.from(e.target.files));
        });

        function handleFiles(files) {
            selectedFiles = files;
            updateFileDisplay();
            processBtn.disabled = files.length === 0;
        }

        function updateFileDisplay() {
            if (selectedFiles.length === 0) {
                fileList.innerHTML = '';
                return;
            }
            
            const fileInfos = selectedFiles.map(file => 
                `<div class="file-info">
                    📄 <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)
                    <br><small>Type: ${file.type || 'Unknown'}</small>
                </div>`
            ).join('');
            
            fileList.innerHTML = fileInfos;
        }

        async function processFiles() {
            if (selectedFiles.length === 0) {
                results.innerHTML = '<div class="result error">Please select at least one file.</div>';
                return;
            }

            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });

            results.innerHTML = `
                <div class="result processing">
                    <div class="loading"></div>
                    🔄 Processing your documents... This may take a few moments.
                </div>`;
            processBtn.disabled = true;

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                
                if (response.ok) {
                    currentDocumentData = data;
                    renderProcessingResults(data);
                } else {
                    results.innerHTML = `<div class="result error">
                        <h3>❌ Error</h3>
                        <p>${data.detail}</p>
                    </div>`;
                }
            } catch (error) {
                results.innerHTML = `<div class="result error">
                    <h3>❌ Network Error</h3>
                    <p>${error.message}</p>
                </div>`;
            } finally {
                processBtn.disabled = false;
            }
        }

        function renderProcessingResults(data) {
            const successfulResults = data.results.filter(r => r.status === 'success');
            const errorResults = data.results.filter(r => r.status === 'error');
            
            let html = '<div class="processing-results">';
            html += '<h2>📊 Processing Results</h2>';
            
            // Stats
            html += `<div class="stats">
                <div class="stat">📄 Total Files: ${data.processed_files}</div>
                <div class="stat">✅ Successful: ${successfulResults.length}</div>
                <div class="stat">❌ Errors: ${errorResults.length}</div>
            </div>`;
            
            // Error results
            if (errorResults.length > 0) {
                html += '<h3>❌ Failed Files</h3>';
                errorResults.forEach(result => {
                    html += `<div class="result error">
                        <strong>${result.filename}</strong>: ${result.error}
                    </div>`;
                });
            }
            
            // Successful results
            if (successfulResults.length > 0) {
                html += '<h3>✅ Successfully Processed Documents</h3>';
                successfulResults.forEach(result => {
                    html += renderDocumentCard(result);
                });
            }
            
            html += '</div>';
            results.innerHTML = html;
        }

        function renderDocumentCard(result) {
            const documentId = result.document_id || 'unknown';
            const workflowResults = result.workflow_results || {};
            const extractionResults = workflowResults.extraction_results || [];
            
            let totalFields = 0;
            let reviewNeededFields = 0;
            
            const thresholds = (workflowResults.thresholds || {});
            const autoThresh = thresholds.auto ?? 0.85;
            extractionResults.forEach(segment => {
                totalFields += segment.fields ? segment.fields.length : 0;
                if (segment.fields) {
                    reviewNeededFields += segment.fields.filter(f => f.confidence < autoThresh).length;
                }
            });
            
            // Initialize card HTML and header
            let html = `<div class="document-card">
                <div class="document-header">
                    <div>
                        <strong>📄 ${result.filename}</strong>
                        <div style="font-size: 0.9em; color: #6c757d; margin-top: 5px;">
                            Document ID: ${documentId}
                        </div>
                    </div>
                    <div>
                        ${result.review_needed ? `<button class="btn warning" onclick="loadReviewInterface('${documentId}')">📝 Review Required (${reviewNeededFields}) fields</button>` : `<span class="stat">✅ All fields look good</span>`}
                        <button class="btn success" onclick="approveAndSubmit('${documentId}')">✔️ Approve and Submit</button>
                    </div>
                </div>
                <div class="document-content">`;
            
            // Classification info
            if (workflowResults.classifications && workflowResults.classifications.length > 0) {
                const classification = workflowResults.classifications[0];
                html += `<div style="margin-bottom: 15px;">
                    <strong>Document Type:</strong> ${classification.doc_type} | 
                    <strong>Vendor:</strong> ${classification.vendor} | 
                    <strong>Confidence:</strong> ${(classification.confidence * 100).toFixed(1)}%
                </div>`;
            }
            
            // Quick stats
            html += `<div class="stats">
                <div class="stat">📄 Segments: ${extractionResults.length}</div>
                <div class="stat">🔍 Total Fields: ${totalFields}</div>
                <div class="stat">⚠️ Need Review: ${reviewNeededFields}</div>
            </div>`;
            
            // Quick preview of extracted fields
            if (extractionResults.length > 0 && extractionResults[0].fields) {
                html += '<h4>🔍 Extracted Data Preview</h4>';
                const previewFields = extractionResults[0].fields.slice(0, 5);
                const thresholds = (workflowResults.thresholds || {});
                const auto = thresholds.auto ?? 0.85;
                const reviewLow = thresholds.review_low ?? 0.6;
                previewFields.forEach(field => {
                    const confidenceClass = field.confidence >= auto ? 'confidence-high' : 
                                          field.confidence >= reviewLow ? 'confidence-medium' : 'confidence-low';
                    html += `<div class="field-editor ${field.confidence < auto ? 'needs-review' : 'high-confidence'}">`;
                        html += `<div class="field-label">${field.name}:</div>`;
                        html += `<div class="field-input" style="border: none; background: transparent;">${field.value}</div>`;
                        html += `<span class="confidence-badge ${confidenceClass}">${(field.confidence * 100).toFixed(1)}%</span>`;
                        html += `</div>`;
                });
                
                if (extractionResults[0].fields.length > 5) {
                    html += `<p style="color: #6c757d; font-style: italic;">... and ${extractionResults[0].fields.length - 5} more fields</p>`;
                }
            }
            
            html += `<div style="margin-top: 15px; text-align: center;">
                <button class="btn" onclick="viewAllFields('${documentId}')">👁️ View All Fields</button>
                <button class="btn" onclick="downloadResults('${documentId}')">💾 Download Results</button>
            </div>`;
            
            html += '</div></div>';
            return html;
        }

        async function loadReviewInterface(documentId) {
            try {
                const response = await fetch(`/review-data/${documentId}`);
                const reviewData = await response.json();
                
                if (reviewData.status === 'success') {
                    showEditableInterface(reviewData.data, documentId);
                } else {
                    alert('No review data found for this document');
                }
            } catch (error) {
                console.error('Error loading review data:', error);
                alert('Error loading review interface');
            }
        }

        function showEditableInterface(reviewData, documentId) {
            let html = `<h2>📝 Review & Edit: ${documentId}</h2>`;
            html += '<div style="margin-bottom: 20px; color: #856404; background-color: #fff3cd; padding: 15px; border-radius: 6px;">';
            html += '⚠️ <strong>Review Required:</strong> Some extracted fields have low confidence and need human verification.';
            html += '</div>';
            
            reviewData.forEach(segment => {
                html += `<div class="segment-card">
                    <div class="segment-header">
                        📄 Segment: ${segment.segment_id} | Type: ${segment.doc_type}
                    </div>
                    <div class="fields-container">`;
                
                const thresholds = (currentDocumentData.results?.find(r => r.document_id === documentId)?.workflow_results?.thresholds) || {};
                const auto = thresholds.auto ?? 0.85;
                const reviewLow = thresholds.review_low ?? 0.6;
                segment.all_fields.forEach(field => {
                    const confidenceClass = field.current_confidence >= auto ? 'confidence-high' : 
                                          field.current_confidence >= reviewLow ? 'confidence-medium' : 'confidence-low';
                    
                    html += `<div class="field-editor ${field.needs_correction ? 'needs-review' : 'high-confidence'}">`
                        + `<div class="field-label">${field.field_name}:</div>`
                        + `<input type="text" 
                               class="field-input"
                               id="field_${segment.segment_id}_${field.field_name}"
                               value="${field.current_value}" 
                               data-original="${field.current_value}"
                               data-segment="${segment.segment_id}"
                               data-field="${field.field_name}"
                               onchange="markFieldChanged(this)">`
                        + `<span class="confidence-badge ${confidenceClass}">${(field.current_confidence * 100).toFixed(1)}%</span>`
                        + `${field.needs_correction ? '<span class="review-flag">⚠️ Review</span>' : ''}`
                        + `</div>`;
                });
                
                html += '</div></div>';
            });
            
            html += `<div class="actions">
                <button class="btn success" onclick="approveAndSubmit('${documentId}')">✔️ Approve and Submit</button>
                <button class="btn" onclick="closeModal()">❌ Cancel</button>
                <button class="btn" onclick="resetAllFields()">🔄 Reset All</button>
            </div>`;
            
            document.getElementById('modalContent').innerHTML = html;
            document.getElementById('editModal').style.display = 'block';
        }

        function markFieldChanged(input) {
            const original = input.dataset.original;
            const current = input.value;
            
            if (original !== current) {
                input.classList.add('changed');
            } else {
                input.classList.remove('changed');
            }
        }

        function resetAllFields() {
            document.querySelectorAll('.field-input').forEach(input => {
                input.value = input.dataset.original;
                input.classList.remove('changed');
            });
        }

        async function approveAndSubmit(documentId) {
            alert('Approve & Submit is disabled for now.');
            return;
        }

        async function refreshDocumentStatus(documentId) {
            // In a real implementation, you might want to refresh the document status
            console.log('Refreshing status for:', documentId);
        }

        function viewAllFields(documentId) {
            // Find the document in current data
            const docResult = currentDocumentData.results.find(r => r.document_id === documentId);
            if (!docResult) return;
            
            let html = `<h2>👁️ All Extracted Fields: ${docResult.filename}</h2>`;
            
            const extractionResults = docResult.workflow_results.extraction_results || [];
            extractionResults.forEach(segment => {
                html += `<div class="segment-card">
                    <div class="segment-header">
                        📄 ${segment.segment_id} | Type: ${segment.doc_type}
                    </div>
                    <div class="fields-container">`;
                
                if (segment.fields) {
                    segment.fields.forEach(field => {
                        const confidenceClass = field.confidence >= 0.85 ? 'confidence-high' : 
                                              field.confidence >= 0.6 ? 'confidence-medium' : 'confidence-low';
                        
                        html += `<div class="field-editor ${field.confidence < 0.85 ? 'needs-review' : 'high-confidence'}">
                            <div class="field-label">${field.name}:</div>
                            <div class="field-input" style="border: none; background: transparent; padding: 5px;">${field.value}</div>
                            <span class="confidence-badge ${confidenceClass}">${(field.confidence * 100).toFixed(1)}%</span>
                        </div>`;
                    });
                }
                
                html += '</div></div>';
            });
            
            html += `<div class="actions">
                <button class="btn" onclick="closeModal()">✅ Close</button>
            </div>`;
            
            document.getElementById('modalContent').innerHTML = html;
            document.getElementById('editModal').style.display = 'block';
        }

        function downloadResults(documentId) {
            const docResult = currentDocumentData.results.find(r => r.document_id === documentId);
            if (!docResult) return;
            
            const dataStr = JSON.stringify(docResult, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${documentId}_results.json`;
            link.click();
            
            URL.revokeObjectURL(url);
        }

        function closeModal() {
            document.getElementById('editModal').style.display = 'none';
        }

        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('editModal');
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        }
    </script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the enhanced upload form with editing capabilities."""
    return UPLOAD_TEMPLATE

@app.post("/process", response_model=ProcessingResponse)
async def process_documents(files: List[UploadFile] = File(...)):
    """Process uploaded documents through the LangGraph workflow."""
    
    if not files or all(file.filename == '' for file in files):
        raise HTTPException(status_code=400, detail="No files selected")

    results = []
    
    # Build the LangGraph app once
    try:
        workflow_app = build_graph()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize workflow: {str(e)}")
    
    for file in files:
        if not file.filename:
            results.append(FileProcessingResult(
                filename="unknown",
                status="error",
                error="No filename provided"
            ))
            continue
            
        if not allowed_file(file.filename):
            results.append(FileProcessingResult(
                filename=file.filename,
                status="error",
                error="File type not supported"
            ))
            continue
            
        try:
            # Check file size
            file_content = await file.read()
            if len(file_content) > MAX_FILE_SIZE:
                results.append(FileProcessingResult(
                    filename=file.filename,
                    status="error",
                    error=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
                ))
                continue
            
            # Convert to bytes
            processed_bytes = await convert_to_bytes(file_content, file.filename)
            
            # Prepare initial state for LangGraph
            initial_state = {
                "file_bytes": processed_bytes,
                "metadata": {"filename": file.filename}
            }
            
            print(f"\n🚀 Processing {file.filename} through LangGraph workflow...")
            
            # Collect all events from the workflow
            final_state = None
            
            try:
                for event in workflow_app.stream(initial_state):
                    final_state = event
                
                # Extract final results
                if final_state:
                    # Get the final state from the last event
                    last_node = list(final_state.keys())[0]
                    final_results = final_state[last_node] or {}
                    
                    # Check if review is needed based on reviewer output files (authoritative)
                    review_dir = Path(f"review_data/{final_results.get('document_id')}")
                    review_needed = False
                    try:
                        if review_dir.exists():
                            review_needed = any(review_dir.glob("*_review.json"))
                    except Exception:
                        # Fallback to heuristic if filesystem is not available
                        review_needed = any(
                            any(f.confidence < 0.85 for f in e.fields)
                            for e in final_results.get('extraction_results', [])
                        )
                    
                    workflow_results = {
                        'document_id': final_results.get('document_id'),
                        'final_decision': final_results.get('final_decision'),
                        'segments_processed': len(final_results.get('segments', [])),
                        'thresholds': (final_results.get('thresholds') or {}),
                        'classifications': [
                            {
                                'segment_id': c.segment_id, 
                                'doc_type': c.doc_type, 
                                'vendor': c.vendor, 
                                'confidence': c.confidence
                            }
                            for c in final_results.get('classifications', [])
                        ],
                        'extraction_results': [
                            {
                                'segment_id': e.segment_id,
                                'doc_type': e.doc_type,
                                'fields': [
                                    {
                                        'name': f.name, 
                                        'value': f.value, 
                                        'confidence': f.confidence
                                    }
                                    for f in e.fields
                                ]
                            }
                            for e in final_results.get('extraction_results', [])
                        ],
                        'application_results': [],
                        'alerts': [
                            {'level': alert.level, 'message': alert.message}
                            for alert in final_results.get('alerts', [])
                        ]
                    }
                    
                    enriched_metadata = final_results.get('metadata') or {"filename": file.filename}
                    # Persist results in Redis for approval phase
                    if final_results.get('document_id'):
                        update_job_state(final_results.get('document_id'), {
                            'status': 'completed',
                            'results': _to_jsonable(final_results),
                            'completed_at': datetime.now().isoformat(),
                        })
                    results.append(FileProcessingResult(
                        filename=file.filename,
                        status="success",
                        document_id=final_results.get('document_id'),
                        metadata=enriched_metadata,
                        workflow_results=workflow_results,
                        review_needed=review_needed
                    ))
                else:
                    results.append(FileProcessingResult(
                        filename=file.filename,
                        status="error",
                        error="Workflow completed but no final state received"
                    ))
                    
            except Exception as workflow_error:
                results.append(FileProcessingResult(
                    filename=file.filename,
                    status="error",
                    error=f"Workflow execution failed: {str(workflow_error)}"
                ))
                
        except Exception as e:
            results.append(FileProcessingResult(
                filename=file.filename,
                status="error",
                error=str(e)
            ))

    return ProcessingResponse(
        status="completed",
        processed_files=len(results),
        results=results
    )

def _run_workflow_background(file_bytes: bytes, filename: str, document_id: str):
    # Deprecated: background tasks replaced by Celery. Retained as no-op for backward compatibility.
    pass

@app.post("/process-init")
async def process_init(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Immediately return assigned document_ids and run processing in the background.

    Returns: { status, results: [{ filename, document_id, queued: true }] }
    """
    if not files or all(file.filename == '' for file in files):
        raise HTTPException(status_code=400, detail="No files selected")

    print("\n[process-init] Received request")
    print(f"[process-init] Files count: {0 if not files else len(files)}")
    results: List[Dict[str, Any]] = []

    for file in files:
        print(f"[process-init] Handling file: name='{file.filename}', content_type='{file.content_type}'")
        if not file.filename:
            results.append({
                "filename": "unknown",
                "status": "error",
                "error": "No filename provided"
            })
            continue

        # Read and validate size
        file_content = await file.read()
        print(f"[process-init] Read bytes: {len(file_content)}")
        if len(file_content) > MAX_FILE_SIZE:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            })
            continue

        # Convert and assign document_id
        try:
            processed_bytes = await convert_to_bytes(file_content, file.filename)
            print(f"[process-init] Converted bytes length: {len(processed_bytes)}")
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": e.detail,
            })
            continue

        document_id = f"doc_{uuid.uuid4().hex[:6]}"
        print(f"[process-init] Assigned document_id: {document_id}")

        # Save original raw document bytes for frontend rendering
        try:
            raw_dir = Path(f"raw documents/{document_id}")
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / file.filename
            with open(raw_path, 'wb') as rf:
                rf.write(file_content)
            print(f"[process-init] Raw document saved: {raw_path}")
        except Exception as e:
            print(f"[background] Warning: could not persist raw document for {document_id}: {e}")

        # Initialize job in Redis and enqueue Celery task
        set_job_status(document_id, "queued", {"filename": file.filename, "created_at": datetime.now().isoformat()})
        # Celery requires bytes; ensure processed_bytes is bytes already
        process_document_task.delay(document_id=document_id, file_bytes=processed_bytes, filename=file.filename)
        print(f"[process-init] Celery task enqueued for {document_id}")

        results.append({
            "filename": file.filename,
            "status": "queued",
            "document_id": document_id
        })

    print(f"[process-init] Returning response for {len(results)} files")
    return {"status": "accepted", "processed_files": len(results), "results": results}

@app.get("/raw-document/{document_id}")
async def get_raw_document(document_id: str):
    """Return the originally uploaded document bytes for rendering in the frontend."""
    raw_dir = Path(f"raw documents/{document_id}")
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail="Raw document not found")
    # Choose the first file in the folder
    files = [p for p in raw_dir.iterdir() if p.is_file()]
    if not files:
        raise HTTPException(status_code=404, detail="Raw document file missing")
    raw_path = files[0]
    mime_type, _ = mimetypes.guess_type(str(raw_path))
    return FileResponse(path=str(raw_path), media_type=mime_type or 'application/octet-stream', filename=raw_path.name)

class ReviewSubmitRequest(BaseModel):
    document_id: str
    data: Optional[List[Dict[str, Any]]] = None  # segment-wise review objects with updated_value per field
    consolidated_fields: Optional[Dict[str, Any]] = None  # flat map: field_name -> updated_value

@app.post("/review-submit")
async def review_submit(payload: ReviewSubmitRequest):
    """Apply human corrections from a consolidated flat map and upsert, then learn.

    - Compare against in-memory extracted fields, update only changed ones.
    - Persist human overrides with original and corrected values.
    - Upsert to Salesforce with corrected values.
    - Trigger learning agent.
    """
    document_id = payload.document_id
    consolidated_updates: Dict[str, Any] = payload.consolidated_fields or {}
    segments_payload = payload.data or []

    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")

    out_dir = Path(f"learning_data/{document_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve authoritative state from Redis, waiting until results are ready
    # Block until results are present (polling), so clients don't need a separate /job-status call
    start_ts = time.time()
    max_wait_s = float(os.getenv("REVIEW_SUBMIT_MAX_WAIT_SEC", "180"))
    poll_s = max(0.05, float(os.getenv("REVIEW_SUBMIT_POLL_SEC", "0.5")))
    job_state = None
    while True:
        job_state = get_job_state(document_id)
        if job_state and job_state.get("results"):
            break
        if max_wait_s > 0 and (time.time() - start_ts) >= max_wait_s:
            raise HTTPException(status_code=404, detail="No results found for this document (timeout waiting for processing)")
        await asyncio.sleep(poll_s)
    state_snapshot = job_state.get("results")

    # Build index of current extracted fields: name -> entry
    extraction_results = state_snapshot.get("extraction_results") or []
    current_index: Dict[str, Dict[str, Any]] = {}
    for extraction in extraction_results:
        seg_id = getattr(extraction, 'segment_id', None) if not isinstance(extraction, dict) else extraction.get('segment_id')
        fields_list = getattr(extraction, 'fields', None) if not isinstance(extraction, dict) else extraction.get('fields')
        for f in fields_list or []:
            name = getattr(f, 'name', None) if not isinstance(f, dict) else f.get('name')
            value = getattr(f, 'value', None) if not isinstance(f, dict) else f.get('value')
            conf = getattr(f, 'confidence', None) if not isinstance(f, dict) else f.get('confidence')
            current_index[name] = {"segment_id": seg_id, "field": f, "current_value": value, "current_confidence": conf}

    # Back-compat: if only segment-wise payload is provided, merge to flat updates
    if segments_payload and not consolidated_updates:
        merged_updates: Dict[str, Any] = {}
        for seg in segments_payload:
            for fld in (seg.get("all_fields") or []):
                name = fld.get("field_name")
                if name is not None and "updated_value" in fld:
                    merged_updates[name] = fld.get("updated_value")
        consolidated_updates = merged_updates

    # Compute corrections
    corrections_map: Dict[str, Dict[str, Any]] = {}
    changes_for_persist: Dict[str, Dict[str, Any]] = {}
    total_changes = 0
    for name, updated_val in (consolidated_updates or {}).items():
        if name not in current_index:
            continue
        entry = current_index[name]
        seg_id = entry["segment_id"]
        current_val = entry["current_value"]
        if updated_val != current_val:
            corrections_map.setdefault(seg_id, {})[name] = {
                "original_value": current_val,
                "corrected_value": updated_val,
                "timestamp": datetime.now().isoformat(),
            }
            changes_for_persist[name] = {
                "segment_id": seg_id,
                "original_value": current_val,
                "corrected_value": updated_val
            }
            total_changes += 1

    # Persist human overrides
    if total_changes > 0:
        human_file = out_dir / f"human_overrides_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(human_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "document_id": document_id,
                    "total_changes": total_changes,
                    "corrections": [
                        {"field_name": n, **data} for n, data in changes_for_persist.items()
                    ],
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to persist human overrides: {e}")

    # Apply corrections to state
    if total_changes > 0:
        try:
            apply_result = apply_human_corrections(state_snapshot, corrections_map)
            if apply_result and isinstance(apply_result, dict) and apply_result.get("extraction_results"):
                state_snapshot["extraction_results"] = apply_result["extraction_results"]
        except Exception as e:
            print(f"Error applying corrections during review-submit: {e}")

    # Upsert then learn
    try:
        app_result = application_agent(state_snapshot)
        state_snapshot.update(app_result)
        update_job_state(document_id, {"results": state_snapshot, "application_ran_at": datetime.now().isoformat()})
        try:
            print(f"[review-submit] Triggering learning_agent; total_changes={total_changes}")
            learning_agent(state_snapshot)
        except Exception as e:
            print(f"Learning agent failed after review-submit: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Application step failed: {str(e)}")

    # Response summary
    application_results = [
        {
            "segment_id": r.segment_id,
            "success": r.success,
            "external_id": r.external_id,
            "error_message": r.error_message
        }
        for r in state_snapshot.get("application_results", [])
    ]

    return {
        "status": "success",
        "document_id": document_id,
        "total_changes": total_changes,
        "application_results": application_results,
    }

@app.get("/job-status/{document_id}")
async def job_status(document_id: str):
    state = get_job_state(document_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return state

# Disabled approve-submit endpoint (kept for future use)
# @app.post("/approve-submit")
# async def approve_and_submit(correction_request: HumanCorrectionRequest):
#     """Apply optional corrections (if any) and then upsert to Salesforce."""
#     raise HTTPException(status_code=503, detail="approve-submit is temporarily disabled")

@app.get("/review-data/{document_id}")
async def get_review_data(document_id: str):
    """Get review data for a document."""
    review_data_dir = Path(f"review_data/{document_id}")
    if not review_data_dir.exists():
        return {"status": "not_found", "message": "No review data found for this document"}
    
    review_files = list(review_data_dir.glob("*_review.json"))
    review_data = []
    
    for file_path in review_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                review_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return {
        "status": "success" if review_data else "not_found",
        "document_id": document_id,
        "review_files": len(review_data),
        "data": review_data
    }

@app.get("/extraction-structured/{document_id}")
async def get_extraction_structured(document_id: str, poll_ms: int = 500, settle_ms: int = 800, max_wait_ms: int = 0):
    """Aggregate structured extraction JSON per segment for a document.
    Reads files created by extractor in `document_images/{document_id}/structured_outputs`.

    This endpoint blocks until at least one structured output file exists for the
    given document_id, then returns the aggregated data. It will continue waiting
    (with polling) until files appear, so callers should await this endpoint.
    """
    print(f"[extraction-structured] document_id={document_id}, poll_ms={poll_ms}, settle_ms={settle_ms}, max_wait_ms={max_wait_ms}")
    base_dir = Path(f"document_images/{document_id}/structured_outputs")
    images_dir = Path(f"document_images/{document_id}")

    # Adaptive wait: wait until no new structured files appear for settle_ms
    last_count = -1
    last_change_ts = time.time()
    start_ts = time.time()
    settle_s = max(0.05, settle_ms / 1000)
    while True:
        # If max_wait_ms <= 0, wait indefinitely; otherwise honor the cap
        if max_wait_ms and (time.time() - start_ts) * 1000 >= max_wait_ms:
            print(f"[extraction-structured] timeout waiting for files for {document_id}")
            break
        expected_segments = 0
        try:
            if images_dir.exists():
                page_images = list(images_dir.glob(f"{document_id}_page_*.png"))
                if not page_images:
                    page_images = list(images_dir.glob("*.png"))
                expected_segments = len(page_images)
        except Exception:
            expected_segments = 0

        current_count = 0
        if base_dir.exists():
            current_count = len(list(base_dir.glob("*-structured.json")))

        if current_count > 0:
            if current_count != last_count:
                last_count = current_count
                last_change_ts = time.time()
            # If we know expected and have reached it, wait for quiescence then break
            if expected_segments > 0 and current_count >= expected_segments and (time.time() - last_change_ts) >= settle_s:
                break
            # If expected unknown, break when stable for settle period
            if expected_segments == 0 and (time.time() - last_change_ts) >= settle_s:
                break

        await asyncio.sleep(max(0.01, poll_ms / 1000))

    segments = []
    errors = []
    # Read structured outputs and sort by page number extracted from image_path or segment_id
    def _extract_page_index_from_path(path_str: str) -> int:
        try:
            import re
            m = re.search(r"page_(\\d+)", path_str or "")
            if m:
                return int(m.group(1))
        except Exception:
            pass
        # Fallback: try segment suffix _seg_{n}
        try:
            import re
            m = re.search(r"_seg_(\\d+)", path_str or "")
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return 10**9  # push unknowns to the end

    file_paths = list(base_dir.glob("*-structured.json"))
    # Sort by page index derived from filename first (uses contained image_path later for robust sort)
    file_paths.sort(key=lambda p: _extract_page_index_from_path(str(p)))

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Normalize a compact segment view
                parsed = data.get("parsed_data") or {}
                fields = []
                if isinstance(parsed, dict) and isinstance(parsed.get("fields"), list):
                    for fld in parsed.get("fields", []):
                        try:
                            fields.append({
                                "name": fld.get("name"),
                                "value": fld.get("value"),
                                "confidence": float(fld.get("confidence", 0.0))
                            })
                        except Exception:
                            continue
                segments.append({
                    "segment_id": data.get("segment_id"),
                    "doc_type": data.get("doc_type"),
                    "image_path": data.get("image_path"),
                    "page_index": _extract_page_index_from_path(str(data.get("image_path", ""))),
                    "fields": fields,
                    "raw_gemini_response_present": bool(data.get("raw_gemini_response"))
                })
        except Exception as e:
            errors.append({"file": str(file_path), "error": str(e)})

    # Final sort of segments by page_index to ensure correct ordering
    try:
        segments.sort(key=lambda seg: seg.get("page_index", 10**9))
        # Remove helper key from output
        for seg in segments:
            if "page_index" in seg:
                del seg["page_index"]
    except Exception:
        pass

    # Build a consolidated fields array in page order for convenience
    consolidated_fields = []
    for seg in segments:
        for fld in (seg.get("fields") or []):
            consolidated_fields.append(fld)

    print(f"Consolidated fields: {consolidated_fields}")
    return {
        "status": "success" if segments else "not_found",
        "document_id": document_id,
        "segments_count": len(segments),
        "segments": segments,
        "fields": consolidated_fields,
        "errors": errors
    }

@app.get("/review-low-confidence/{document_id}")
async def get_review_low_confidence(document_id: str, poll_ms: int = 500, settle_ms: int = 800):
    """Return only the low-confidence fields prepared by reviewer for a document.
    Reads files created by reviewer in `review_data/{document_id}/*_review.json`.
    """
    review_data_dir = Path(f"review_data/{document_id}")
    images_dir = Path(f"document_images/{document_id}")

    # If review data folder doesn't exist yet, wait until it is created by the reviewer node
    # Determine expected number of segments (pages) from generated images, if available
    expected_segments = 0
    try:
        if images_dir.exists():
            page_images = list(images_dir.glob(f"{document_id}_page_*.png"))
            if not page_images:
                page_images = list(images_dir.glob("*.png"))
            expected_segments = len(page_images)
    except Exception:
        expected_segments = 0

    # Adaptive wait: return once no new review files appear for settle_ms
    last_count = -1
    last_change_ts = time.time()
    settle_s = max(0.05, settle_ms / 1000)
    while True:
        try:
            if images_dir.exists():
                page_images = list(images_dir.glob(f"{document_id}_page_*.png"))
                if not page_images:
                    page_images = list(images_dir.glob("*.png"))
                expected_segments = len(page_images)
        except Exception:
            expected_segments = 0

        current_count = 0
        if review_data_dir.exists():
            current_count = len(list(review_data_dir.glob("*_review.json")))

        if current_count > 0:
            if current_count != last_count:
                last_count = current_count
                last_change_ts = time.time()
            if expected_segments > 0 and current_count >= expected_segments and (time.time() - last_change_ts) >= settle_s:
                break
            if expected_segments == 0 and (time.time() - last_change_ts) >= settle_s:
                break

        await asyncio.sleep(max(0.01, poll_ms / 1000))

    segments = []
    errors = []
    # Helper to extract page index for ordering
    def _extract_page_index(identifier: str) -> int:
        try:
            import re
            m = re.search(r"_seg_(\\d+)", identifier or "")
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return 10**9

    file_paths = list(review_data_dir.glob("*_review.json"))
    file_paths.sort(key=lambda p: _extract_page_index(str(p)))

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                fields_needing_review = data.get("fields_needing_review") or []
                segments.append({
                    "segment_id": data.get("segment_id"),
                    "doc_type": data.get("doc_type"),
                    "page_index": _extract_page_index(str(data.get("segment_id", ""))),
                    "fields_needing_review": fields_needing_review
                })
        except Exception as e:
            errors.append({"file": str(file_path), "error": str(e)})

    # Final sort by page_index and remove helper key
    try:
        segments.sort(key=lambda seg: seg.get("page_index", 10**9))
        for seg in segments:
            if "page_index" in seg:
                del seg["page_index"]
    except Exception:
        pass

    # Consolidate all fields_needing_review in page order
    consolidated_fields = []
    for seg in segments:
        for fld in (seg.get("fields_needing_review") or []):
            consolidated_fields.append(fld)

    return {
        "status": "success" if segments else "not_found",
        "document_id": document_id,
        "segments_count": len(segments),
        "segments": segments,
        "fields_needing_review": consolidated_fields,
        "errors": errors
    }

@app.get("/document-status/{document_id}")
async def get_document_status(document_id: str):
    """Get the current status of a document."""
    # Check various directories for document data
    status_info = {
        "document_id": document_id,
        "has_images": False,
        "has_extraction_logs": False,
        "has_review_data": False,
        "has_learning_data": False,
        "status": "unknown"
    }
    
    # Check for document images
    image_dir = Path(f"document_images/{document_id}")
    if image_dir.exists():
        status_info["has_images"] = True
        status_info["image_count"] = len(list(image_dir.glob("*.png")))
    
    # Check for extraction logs
    log_file = Path(f"extraction_logs/{document_id}_extraction_log.json")
    if log_file.exists():
        status_info["has_extraction_logs"] = True
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                status_info["extraction_timestamp"] = log_data.get("extraction_timestamp")
                status_info["segments_processed"] = log_data.get("total_segments", 0)
        except:
            pass
    
    # Check for review data
    review_dir = Path(f"review_data/{document_id}")
    if review_dir.exists():
        status_info["has_review_data"] = True
        status_info["review_files"] = len(list(review_dir.glob("*_review.json")))
    
    # Check for learning data
    learning_dir = Path(f"learning_data/{document_id}")
    if learning_dir.exists():
        status_info["has_learning_data"] = True
        status_info["correction_files"] = len(list(learning_dir.glob("corrections_*.json")))
    
    # Determine overall status
    if status_info["has_extraction_logs"] and not status_info["has_review_data"]:
        status_info["status"] = "processed"
    elif status_info["has_review_data"]:
        status_info["status"] = "needs_review"
    elif status_info["has_learning_data"]:
        status_info["status"] = "reviewed"
    
    return status_info

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "message": "Document processing service is running",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "supported_extensions": sorted(list(ALLOWED_EXTENSIONS)),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "total_formats": len(ALLOWED_EXTENSIONS)
    }

@app.get("/stats")
async def get_system_stats():
    """Get system processing statistics."""
    stats = {
        "documents_processed": 0,
        "total_segments": 0,
        "corrections_made": 0,
        "storage_usage": {}
    }
    
    # Count documents processed
    extraction_logs = Path("extraction_logs")
    if extraction_logs.exists():
        stats["documents_processed"] = len(list(extraction_logs.glob("*_extraction_log.json")))
    
    # Count learning events
    learning_data = Path("learning_data")
    if learning_data.exists():
        correction_files = list(learning_data.glob("*/corrections_*.json"))
        stats["corrections_made"] = len(correction_files)
    
    # Storage usage
    for folder in ["document_images", "extraction_logs", "review_data", "learning_data"]:
        folder_path = Path(folder)
        if folder_path.exists():
            total_size = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file())
            stats["storage_usage"][folder] = f"{total_size / (1024*1024):.2f} MB"
    
    return stats

# Error handlers
@app.exception_handler(413)
async def payload_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"detail": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."}
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Enhanced Document Processing FastAPI Application")
    print("📁 Supported file types:", ', '.join(sorted(ALLOWED_EXTENSIONS)))
    print("🌐 Web interface: http://localhost:8000")
    print("📊 API documentation: http://localhost:8000/docs")
    print("📈 System stats: http://localhost:8000/stats")
    print("❤️ Health check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
