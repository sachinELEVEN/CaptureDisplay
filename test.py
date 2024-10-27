import pyperclip
import os
from datetime import datetime

# Keep track of the last used file name
file_name_memory = None

def save_copied_text_to_file():
    global file_name_memory
    
    # Get today's date and time
    today_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine the file name with today's date and a unique 3-digit number
    if file_name_memory is None:
        # Find an available unique file name
        for i in range(1, 1000):
            file_name = f"capture_display_notes-{today_date}-{i:03d}.txt"
            if not os.path.exists(file_name):
                file_name_memory = file_name
                break

    # Get copied content from clipboard
    copied_content = pyperclip.paste()

    # Append the copied content to the file in the specified format
    with open(file_name_memory, 'a') as file:
        file.write(f"//ENTRY STARTS\n")
        file.write(f"//Time: {current_time}\n")
        file.write(f"//Content:\n")
        file.write(f"{copied_content}\n")
        file.write(f"//ENTRY ENDS\n")

    print(f"Content saved to {file_name_memory}")

# Example usage
save_copied_text_to_file()
