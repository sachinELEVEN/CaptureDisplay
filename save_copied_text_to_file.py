import pyperclip
import os
from datetime import datetime
import sys
import importlib
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import numpy as np
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch
from PyPDF2 import PdfWriter, PdfReader

utils = importlib.import_module("utils")
get_resource_path = utils.get_resource_path
append_to_logs = utils.append_to_logs


# Keep track of the last used file name
file_name_memory = None

def save_copied_text_to_file():
    append_to_logs("save_copied_text_to_file")
    global file_name_memory
    
    # Get today's date and time
    today_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Directory for saving the files
    directory = os.path.join(
    os.path.expanduser("~"),
    "Library",
    "Application Support",
    'CaptureDisplayX77'
)

    # Ensure the 'sharable' folder exists
    os.makedirs(directory, exist_ok=True)

    # Determine the file name with today's date and a unique 3-digit number
    if file_name_memory is None:
        # Find an available unique file name within the 'sharable' folder
        for i in range(1, 1000):
            file_name = os.path.join(directory, f"capture_display_notes-{today_date}-{i:03d}.md")
            if not os.path.exists(file_name):
                file_name_memory = file_name
                break

    # Get copied content from clipboard
    copied_content = pyperclip.paste()

    # Append the copied content to the file in the specified format
    with open(file_name_memory, 'a') as file:
        file.write(f"\n## Time: {current_time}\n")
        file.write(f"**Content:**\n")
        file.write(f"```\n{copied_content}\n```\n")

    append_to_logs(f"Content saved to {file_name_memory}")

# Example usage
# save_copied_text_to_file()


file_name_memory = None  # Global variable to store the filename for appending

def save_content_as_pdf(frame=None, save_text=True):
    global file_name_memory
    
    # Define the folder path
    folder_path = os.path.join(
        os.path.expanduser("~"),
        "Library",
        "Application Support",
        "CaptureDisplayX77"
    )
    os.makedirs(folder_path, exist_ok=True)
    
    copied_text = None
    if save_text:
        # Get copied content from clipboard
        copied_text = pyperclip.paste()

    # Create a unique filename based on date and a 3-digit number
    date_str = datetime.now().strftime("%Y-%m-%d")
    counter = 1
    if file_name_memory is None:
        while True:
            filename = f"capture_display_notes-{date_str}-{counter:03}.pdf"
            file_name_memory = os.path.join(folder_path, filename)
            if not os.path.exists(file_name_memory):
                break
            counter += 1

    # Create a temporary PDF to hold new content
    temp_pdf_path = os.path.join(folder_path, "temp_append.pdf")
    pdf = canvas.Canvas(temp_pdf_path, pagesize=letter)
    width, height = letter
    padding = 40
    text_y_position = height - padding

    # Add the timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(padding, text_y_position, f"Time: {timestamp}")
    text_y_position -= padding

    # Add copied text if available, with wrapping
    if copied_text:
        pdf.drawString(padding, text_y_position, "Content:")
        text_y_position -= padding
        text_object = pdf.beginText(padding, text_y_position)
        text_object.setFont("Helvetica", 10)
        text_object.setFillColor(colors.black)
        text_object.setLeading(14)
        max_text_width = width - padding * 2
        for line in copied_text.splitlines():
            wrapped_lines = wrap_text(line, max_text_width, pdf)
            for wrapped_line in wrapped_lines:
                text_object.textLine(wrapped_line)
        pdf.drawText(text_object)
        text_y_position -= padding

    # Add frame image if available
    if frame is not None:
        temp_image_path = os.path.join(folder_path, "temp_image.png")
        image = Image.fromarray(frame)
        image.save(temp_image_path)

        # Scale image if it’s wider than the page width
        image_width, image_height = image.size
        max_image_width = width - padding * 2
        if image_width > max_image_width:
            scale_factor = max_image_width / image_width
            image_width = int(image_width * scale_factor)
            image_height = int(image_height * scale_factor)

        # Insert the image into the PDF and delete the temporary file
        pdf.drawImage(temp_image_path, padding, text_y_position - image_height, width=image_width, height=image_height) 
        os.remove(temp_image_path)

    # Finalize and save the temporary PDF
    pdf.showPage()
    pdf.save()

    # Append temp_pdf to existing file or create it if it doesn't exist
    append_pdf(temp_pdf_path, file_name_memory)
    os.remove(temp_pdf_path)
    print(f"Appended content to PDF: {file_name_memory}")

def wrap_text(text, max_width, pdf_canvas):
    """
    Helper function to wrap text to fit within the specified width.
    """
    words = text.split()
    wrapped_lines = []
    current_line = ""
    for word in words:
        # Test if adding the next word would exceed the width
        test_line = f"{current_line} {word}".strip()
        if pdf_canvas.stringWidth(test_line, "Helvetica", 10) <= max_width:
            current_line = test_line
        else:
            wrapped_lines.append(current_line)
            current_line = word
    if current_line:
        wrapped_lines.append(current_line)
    return wrapped_lines

def append_pdf(temp_pdf_path, target_pdf_path):
    """
    Append the contents of temp_pdf_path to target_pdf_path.
    """
    writer = PdfWriter()

    # If the target file already exists, read and append it
    if os.path.exists(target_pdf_path):
        reader = PdfReader(target_pdf_path)
        for page in reader.pages:
            writer.add_page(page)

    # Add the new page from the temporary PDF
    temp_reader = PdfReader(temp_pdf_path)
    for page in temp_reader.pages:
        writer.add_page(page)

    # Write the combined PDF back to the target file
    with open(target_pdf_path, "wb") as f:
        writer.write(f)
