import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
import os
import tempfile
from preprocess_and_enhance import process_page, find_document_edges, perspective_correction, sharpen_pdf

# Function to process a PDF page
def process_page(image):
    # Find document edges
    corners = find_document_edges(image)
    if corners is not None:
        image = perspective_correction(image, corners)

    # Convert to grayscale
    gray_image = ImageOps.grayscale(image)

    # Enhance and sharpen the image
    enhancer = ImageEnhance.Contrast(gray_image)
    processed_image = enhancer.enhance(2.0)  # Adjust contrast
    processed_image = processed_image.filter(ImageFilter.SHARPEN)  # Sharpen the image

    return processed_image

# Streamlit app
def main():
    st.title("PDF Page Processing with Streamlit")

    # Upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Use a temporary file to handle the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        try:
            st.write("Processing the PDF...")
            output_images = []
            with fitz.open(temp_pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # Process the page (perspective correction, grayscale, enhance, sharpen)
                    processed_image = process_page(img)
                    output_images.append(processed_image)

            # Display processed images
            st.write("Processed Images:")
            for i, img in enumerate(output_images):
                st.image(img, caption=f"Page {i+1}", use_column_width=True)

            # Optionally, sharpen the entire PDF
            sharpen_button = st.button("Sharpen PDF")
            if sharpen_button:
                with st.spinner("Sharpening the entire PDF..."):
                    sharpened_pdf_path = tempfile.mktemp(suffix=".pdf")
                    sharpen_pdf(temp_pdf_path, sharpened_pdf_path)
                    st.success("PDF sharpened.")
                    st.download_button(label="Download Sharpened PDF", 
                                       data=open(sharpened_pdf_path, "rb"), 
                                       file_name="sharpened_output.pdf", 
                                       mime="application/pdf")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

if __name__ == "__main__":
    main()
