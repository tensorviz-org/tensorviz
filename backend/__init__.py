"""PyTorch Model Visualization Backend"""

__version__ = "2.0.0"
__author__ = "PyTorch Viz Team"

# Core modules are available but not auto-imported to avoid circular dependencies
# Use: from backend.services.template_service import TemplateService

__all__ = [
    "TemplateService",
    "ShapeInferenceService", 
    "CodeGenerationService",
    "ModelValidationService",
    "CodeParserService"
]
