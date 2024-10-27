from pynput import keyboard
import time
import importlib
import sys

fast_screen_recording = importlib.import_module("fast-screen-recording")
save_copied_text_to_file = importlib.import_module("save_copied_text_to_file")
zoom_increase = fast_screen_recording.zoom_increase
zoom_decrease = fast_screen_recording.zoom_decrease
window_pt_top_left = fast_screen_recording.window_pt_top_left
window_pt_bottom_right = fast_screen_recording.window_pt_bottom_right
window_show_everything = fast_screen_recording.window_show_everything
toggle_region_of_interest_hiding_approach = fast_screen_recording.toggle_region_of_interest_hiding_approach
save_copied_text_to_file = save_copied_text_to_file.save_copied_text_to_file

# Dictionary to store shortcut combinations and their corresponding functions.
# For example: {('tab', 'z'): 'my_function'}
shortcut_actions = {
    ('ctrl', 'z'): 'my_function',
    ('ctrl', 'z'): 'my_function',
    ('ctrl', 'q'): 'quit_app',
    ('ctrl', '='): 'zoom_increase',
    ('ctrl', '-'): 'zoom_decrease',
    ('ctrl', '9'): 'window_pt_top_left',#window_pt_top_left window_pt_bottom_right represents the window we want to show the user
    ('ctrl', '0'): 'window_pt_bottom_right',
    ('9', '0'): 'window_show_everything',
    ('ctrl', 'b'): 'toggle_region_of_interest_hiding_approach',
    ('ctrl', 'v'): 'save_copied_text_to_file',
}

# Track the currently pressed keys.
current_keys = set()

# Define the functions that will be triggered.
def my_function():
    print("Tab+Z was pressed!")

def quit_app():
    print("user quitted the app")
    sys.exit(1)  # Exit with status 1 (you can change the status code if needed)

# Mapping from function name to actual function.
function_map = {
    'my_function': my_function,
    'another_function': quit_app,
    'zoom_decrease': zoom_decrease,
    'zoom_increase': zoom_increase,
    'window_pt_top_left': window_pt_top_left,
    'window_pt_bottom_right': window_pt_bottom_right,
    'window_show_everything':window_show_everything,
    'toggle_region_of_interest_hiding_approach':toggle_region_of_interest_hiding_approach,
    'save_copied_text_to_file':save_copied_text_to_file

}

# Function to check if a shortcut is triggered.
def on_press(key):
    try:
        # Normalize the key name.
        key_name = key.char if hasattr(key, 'char') else key.name
    except AttributeError:
        return

    # Add the key to the set of currently pressed keys.
    current_keys.add(key_name)

    # Check for any shortcut that matches the currently pressed keys.
    for shortcut, function_name in shortcut_actions.items():
        # If the keys in the shortcut match the currently pressed keys.
        if all(k in current_keys for k in shortcut):
            # Execute the corresponding function.
            if function_name in function_map:
                function_map[function_name]()

def on_release(key):
    try:
        # Normalize the key name.
        key_name = key.char if hasattr(key, 'char') else key.name
    except AttributeError:
        return

    # Remove the key from the set of currently pressed keys.
    if key_name in current_keys:
        current_keys.remove(key_name)

def millisToSeconds(millis):
    return (1/1000) * millis

def listen_keyboard_events(): 
    # Start listening for keyboard events.
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while True:
            # print("Listening for keyboard shortcuts...")
            time.sleep(millisToSeconds(10))
        # listener.join()

# listen_keyboard_events()