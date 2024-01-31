#!/usr/bin/env python
# coding: utf-8

# In[7]:


import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import numpy as np
import cv2

def find_document_edges(image):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edged = cv2.Canny(gray, 75, 200)

    # Find contours and keep the largest one
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Approximate the contour
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx

def perspective_correction(image, corners):
    
    # Order points in clockwise manner
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=2)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    diff = np.diff(corners, axis=2)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    # Perspective correction
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(np.array(image), M, (maxWidth, maxHeight))

    return Image.fromarray(warped)

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

def sharpen_pdf(pdf_path, output_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        # Convert PDF page to an image
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Process the page (perspective correction, grayscale, enhance, sharpen)
        processed_image = process_page(img)

        # Save the processed image
        processed_image.save(os.path.join(output_folder, f"processed_page_{page_num}.png"))

    doc.close()


sharpen_pdf("input.pdf", "output_folder")




# In[ ]:




