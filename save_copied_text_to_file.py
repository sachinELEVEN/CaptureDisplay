import pyperclip
import os
from datetime import datetime
import sys
import importlib
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
    'CaptureDisplayX'
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
save_copied_text_to_file()
