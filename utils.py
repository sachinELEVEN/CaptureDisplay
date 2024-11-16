import os
import sys
import re
import requests

def get_resource_path(relative_path):
        #Paths change after creating an artifact using pyinstaller.
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        #This should be used in only dist, in dev mode return relative path
        if getattr(sys, 'frozen', False):  # PyInstaller sets this attribute
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        # globalSpace.append_to_logs("IN PY_INSTALLER",sys._MEIPASS)
        is_prod = True
        #since prod builds are done from ./build-pipeline that is our base path, we need to remove build-pipeline so that we can access other resources in normal fashion from other files during development like the assets folder
        # base_path = modify_path(base_path,'build-pipeline') if is
        return os.path.join(base_path, relative_path)

def modify_path(path, part_to_remove):
    # Normalize the path to remove any redundant relative parts (like "../" or "./")
    normalized_path = os.path.normpath(path)
    
    # Remove the specified part using regex
    modified_path = re.sub(rf'/{re.escape(part_to_remove)}', '', normalized_path, count=1)
    
    return modified_path


def append_to_logs(*args):
    # return
    logs_folder = os.path.join(
        os.path.expanduser("~"),
        "Library",
        "Application Support",
        'CaptureDisplay'
    )
    log_file_path = os.path.join(logs_folder, "CaptureDisplay.logs")

    # Join all arguments with a space or any other separator
    text_to_append = " ".join(map(str, args))

    try:
        # Create the logs folder if it doesn't exist
        os.makedirs(logs_folder, exist_ok=True)

        # Create the log file path if it doesn't exist
        open(log_file_path, 'a').close()

        # Open the file in append mode ('a+')
        with open(log_file_path, 'a+') as file:
            # Append the provided text to the file
            file.write(text_to_append + '\n')
        print(f'Logged {log_file_path}: ', text_to_append)
    except Exception as e:
        print(f'Error appending text to {log_file_path}: {str(e)}')


def notify_server(message_type, api_url="https://backend.brainsphere.in/capture_display_events"):
    #return
    # Ensure message_type is a string
    if not isinstance(message_type, str):
        # print("Error: message_type must be a string")
        return None

    try:
        # Send a GET request to the Node.js server endpoint with message_type as a query parameter
        response = requests.get(api_url, params={"event_name": message_type,"credentials":"35ge344tgon232@1!.#3EW"})

        # Check if the request was successful
        if response.status_code == 200:
            # Parse JSON response
            # data = response.json()
            # print("API Response:", data)
            return None
        else:
            # print(f"Failed to call API. Status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        # print("Error making request:", e)
        return None
