from pynput import keyboard
import time
import importlib

fast_screen_recording = importlib.import_module("fast-screen-recording")
zoom_increase = fast_screen_recording.zoom_increase
zoom_decrease = fast_screen_recording.zoom_decrease
window_pt_top_left = fast_screen_recording.window_pt_top_left
window_pt_bottom_right = fast_screen_recording.window_pt_bottom_right
window_show_everything = fast_screen_recording.window_show_everything

# Dictionary to store shortcut combinations and their corresponding functions.
# For example: {('tab', 'z'): 'my_function'}
shortcut_actions = {
    ('tab', 'z'): 'my_function',
    ('opt', 'z'): 'my_function',
    ('ctrl', 'shift', 'a'): 'another_function',
    ('tab', '='): 'zoom_increase',
    ('tab', '-'): 'zoom_decrease',
    ('tab', '9'): 'window_pt_top_left',#window_pt_top_left window_pt_bottom_right represents the window we want to show the user
    ('tab', '0'): 'window_pt_bottom_right',
    ('9', '0'): 'window_show_everything',
}

# Track the currently pressed keys.
current_keys = set()

# Define the functions that will be triggered.
def my_function():
    print("Tab+Z was pressed!")

def another_function():
    print("Ctrl+Shift+A was pressed!")

# Mapping from function name to actual function.
function_map = {
    'my_function': my_function,
    'another_function': another_function,
    'zoom_decrease': zoom_decrease,
    'zoom_increase': zoom_increase,
    'window_pt_top_left': window_pt_top_left,
    'window_pt_bottom_right': window_pt_bottom_right,
    'window_show_everything':window_show_everything

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