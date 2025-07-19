"""
PDF processing utilities for Vision-RAG
Extracts text and images from PDF files
"""

import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import tempfile
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing to extract text and images"""
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize PDF processor
        
        Args:
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                text_content += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            doc.close()
            
            logger.info(f"✅ Extracted text from {pdf_path} ({len(text_content)} characters)")
            return text_content
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_images_from_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Extract images from PDF file
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of paths to extracted image files
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get PDF filename without extension
            pdf_name = Path(pdf_path).stem
            
            # Method 1: Extract embedded images using PyMuPDF
            embedded_images = self._extract_embedded_images(pdf_path, output_dir, pdf_name)
            
            # Method 2: Convert PDF pages to images
            page_images = self._convert_pages_to_images(pdf_path, output_dir, pdf_name)
            
            # Combine both methods
            all_images = embedded_images + page_images
            
            logger.info(f"✅ Extracted {len(all_images)} images from {pdf_path}")
            return all_images
            
        except Exception as e:
            logger.error(f"❌ Error extracting images from {pdf_path}: {e}")
            return []
    
    def _extract_embedded_images(self, pdf_path: str, output_dir: str, pdf_name: str) -> List[str]:
        """Extract embedded images from PDF using PyMuPDF"""
        image_paths = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Skip if image is too small or has unusual characteristics
                    if pix.width < 50 or pix.height < 50:
                        pix = None
                        continue
                    
                    # Convert to RGB if necessary
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix1.tobytes("png")
                        pix1 = None
                    
                    # Save image
                    img_filename = f"{pdf_name}_page{page_num + 1}_img{img_index + 1}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_data)
                    
                    image_paths.append(img_path)
                    pix = None
            
            doc.close()
            
        except Exception as e:
            logger.error(f"❌ Error extracting embedded images: {e}")
        
        return image_paths
    
    def _convert_pages_to_images(self, pdf_path: str, output_dir: str, pdf_name: str) -> List[str]:
        """Convert PDF pages to images using pdf2image"""
        image_paths = []
        
        try:
            # Convert PDF to images
            pages = convert_from_path(
                pdf_path,
                dpi=200,  # Good quality for text recognition
                fmt='PNG'
            )
            
            for page_num, page in enumerate(pages):
                img_filename = f"{pdf_name}_page{page_num + 1}_full.png"
                img_path = os.path.join(output_dir, img_filename)
                
                page.save(img_path, 'PNG')
                image_paths.append(img_path)
            
        except Exception as e:
            logger.error(f"❌ Error converting pages to images: {e}")
        
        return image_paths
    
    def process_pdf(self, pdf_path: str, output_dir: str = None) -> Dict[str, any]:
        """
        Process PDF file to extract both text and images
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images (default: temp dir)
            
        Returns:
            Dictionary with extracted text and image paths
        """
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, "extracted_images")
        
        logger.info(f"🔄 Processing PDF: {pdf_path}")
        
        # Extract text
        text_content = self.extract_text_from_pdf(pdf_path)
        
        # Extract images
        image_paths = self.extract_images_from_pdf(pdf_path, output_dir)
        
        result = {
            "pdf_path": pdf_path,
            "text_content": text_content,
            "image_paths": image_paths,
            "text_length": len(text_content),
            "image_count": len(image_paths)
        }
        
        logger.info(f"✅ PDF processing completed: {result['text_length']} chars, {result['image_count']} images")
        
        return result
    
    def is_pdf_file(self, file_path: str) -> bool:
        """Check if file is a PDF"""
        return file_path.lower().endswith('.pdf')
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, any]:
        """Get basic information about PDF file"""
        try:
            doc = fitz.open(pdf_path)
            info = {
                "page_count": len(doc),
                "metadata": doc.metadata,
                "file_size": os.path.getsize(pdf_path)
            }
            doc.close()
            return info
        except Exception as e:
            logger.error(f"❌ Error getting PDF info: {e}")
            return {"error": str(e)} 