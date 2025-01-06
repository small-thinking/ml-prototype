import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import fitz  # PyMuPDF
from PIL import Image, ImageStat
import io
import os
import hashlib
from datetime import datetime
import numpy as np


def is_black_image(image: Image.Image, threshold: float = 0.99) -> bool:
    """
    Check if an image is predominantly black.
    
    Args:
        image (Image.Image): PIL Image object to check
        threshold (float): Threshold for determining blackness (default: 0.99)
        
    Returns:
        bool: True if image is predominantly black, False otherwise
    """
    # Convert to grayscale
    gray_image = image.convert('L')
    
    # Calculate image statistics
    stats = ImageStat.Stat(gray_image)
    
    # Get the percentage of dark pixels
    total_pixels = gray_image.width * gray_image.height
    dark_pixels = sum(1 for pixel in gray_image.getdata() if pixel < 30)  # threshold for "dark"
    dark_ratio = dark_pixels / total_pixels
    
    # Check if the image is too dark
    return dark_ratio > threshold


def analyze_image_quality(image: Image.Image) -> Tuple[bool, str]:
    """
    Analyze the quality of an image.
    
    Args:
        image (Image.Image): PIL Image object to analyze
        
    Returns:
        Tuple[bool, str]: (is_good_quality, reason)
    """
    # Convert to numpy array for analysis
    img_array = np.array(image)
    
    # Check if image is too dark
    if is_black_image(image):
        return False, "predominantly black"
    
    # Check for low contrast
    if len(image.getbands()) > 1:  # For color images
        gray_image = image.convert('L')
        stats = ImageStat.Stat(gray_image)
        if stats.var[0] < 100:  # Variance threshold for contrast
            return False, "low contrast"
    
    # Check if image is too small
    if image.size[0] < 50 or image.size[1] < 50:
        return False, "too small"
    
    # Check if image is mostly uniform color
    std = np.std(img_array)
    if std < 20:
        return False, "uniform color"
        
    return True, "good quality"


def extract_images_from_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    min_size: Tuple[int, int] = (100, 100),
    supported_formats: Optional[List[str]] = None,
    log_level: int = logging.INFO,
    quality_threshold: float = 0.99  # Threshold for black image detection
) -> Dict[str, List[str]]:
    """
    Extract images from a PDF file and save them to the specified directory.
    
    Args:
        pdf_path (str | Path): Path to the PDF file
        output_dir (str | Path): Directory where images will be saved
        min_size (Tuple[int, int]): Minimum width and height for extracted images (default: (100, 100))
        supported_formats (Optional[List[str]]): List of supported image formats (default: None, accepts all)
        log_level (int): Logging level (default: logging.INFO)
        quality_threshold (float): Threshold for image quality checks (default: 0.99)
    
    Returns:
        Dict[str, List[str]]: Dictionary containing:
            - 'success': List of successfully saved image paths
            - 'failed': List of images that failed to extract/save
            - 'skipped': List of images skipped due to size/format/quality constraints
    """
    # Set up logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize result tracking
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    try:
        # Validate and convert paths to Path objects
        pdf_path = os.path.expanduser(pdf_path)
        output_dir = os.path.expanduser(output_dir)
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        
        # Validate inputs
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not output_dir.exists():
            logger.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
        # Set default supported formats if none provided
        if supported_formats is None:
            supported_formats = ['jpeg', 'jpg', 'png', 'bmp']
        
        logger.info(f"Processing PDF: {pdf_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.debug(f"Minimum size: {min_size}")
        logger.debug(f"Supported formats: {supported_formats}")
        
        # Open PDF document
        doc = fitz.open(pdf_path)
        total_images = 0
        
        # Create subdirectories for different quality images
        quality_dir = output_dir / "good_quality"
        low_quality_dir = output_dir / "low_quality"
        quality_dir.mkdir(exist_ok=True)
        low_quality_dir.mkdir(exist_ok=True)
        
        # Process each page
        for page_num, page in enumerate(doc, 1):
            logger.info(f"Processing page {page_num}/{len(doc)}")
            
            # Get image list
            image_list = page.get_images()
            page_images = 0
            
            # Process each image on the page
            for img_idx, img in enumerate(image_list, 1):
                try:
                    # Get image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    if base_image is None:
                        logger.warning(f"Failed to extract image {img_idx} on page {page_num}")
                        results['failed'].append(f"Page {page_num}, Image {img_idx}")
                        continue
                    
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Check format support
                    if image_ext.lower() not in supported_formats:
                        logger.debug(f"Skipping unsupported format: {image_ext}")
                        results['skipped'].append(f"Page {page_num}, Image {img_idx} (format: {image_ext})")
                        continue
                    
                    # Process image with PIL
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Check minimum size
                    if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
                        logger.debug(f"Skipping small image: {image.size}")
                        results['skipped'].append(f"Page {page_num}, Image {img_idx} (size: {image.size})")
                        continue
                    
                    # Check image quality
                    is_good_quality, quality_reason = analyze_image_quality(image)
                    
                    # Generate unique filename using hash of image data and timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                    filename = f"page{page_num:03d}_img{img_idx:03d}_{timestamp}_{image_hash}.{image_ext}"
                    
                    # Save to appropriate directory based on quality
                    if is_good_quality:
                        output_path = quality_dir / filename
                        logger.debug(f"Saving good quality image: {output_path}")
                    else:
                        output_path = low_quality_dir / filename
                        logger.debug(f"Saving low quality image: {output_path} (reason: {quality_reason})")
                    
                    # Save image
                    image.save(output_path, format=image_ext.upper())
                    results['success'].append(str(output_path))
                    page_images += 1
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_idx} on page {page_num}: {str(e)}")
                    results['failed'].append(f"Page {page_num}, Image {img_idx}")
                    continue
            
            total_images += page_images
            logger.info(f"Extracted {page_images} images from page {page_num}")
        
        # Log summary
        logger.info(f"Processing complete. Total images extracted: {total_images}")
        logger.info(f"Successful: {len(results['success'])}")
        logger.info(f"Failed: {len(results['failed'])}")
        logger.info(f"Skipped: {len(results['skipped'])}")
        
        return results
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    
    finally:
        if 'doc' in locals():
            doc.close()


# Example usage
if __name__ == "__main__":
    try:
        # Example parameters
        pdf_file = "~/Downloads/b.pdf"
        save_dir = "~/Downloads/extracted_images"
        min_size = (200, 200)  # Minimum 200x200 pixels
        formats = ['jpeg', 'jpg', 'png']
        
        # Extract images with debug logging
        results = extract_images_from_pdf(
            pdf_file,
            save_dir,
            min_size=min_size,
            supported_formats=formats,
            log_level=logging.DEBUG
        )
        
        # Print results
        print("\nExtraction Results:")
        print(f"Successfully saved: {len(results['success'])} images")
        print(f"Failed to extract: {len(results['failed'])} images")
        print(f"Skipped: {len(results['skipped'])} images")
        
    except Exception as e:
        print(f"Error: {str(e)}")