from pynput import keyboard

# Dictionary to store shortcut combinations and their corresponding functions.
# For example: {('tab', 'z'): 'my_function'}
shortcut_actions = {
    ('tab', 'z'): 'my_function',
    ('ctrl', 'shift', 'a'): 'another_function',
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

def listen_keyboard_events(): 
    # Start listening for keyboard events.
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        print("Listening for keyboard shortcuts...")
        listener.join()

listen_keyboard_events()