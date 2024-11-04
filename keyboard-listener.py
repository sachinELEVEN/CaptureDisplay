from pynput import keyboard
import time
import importlib
import os
from settings_file_manager import SettingsManager

fast_screen_recording = importlib.import_module("fast-screen-recording")
save_copied_text_to_file = importlib.import_module("save_copied_text_to_file")
zoom_increase = fast_screen_recording.zoom_increase
zoom_decrease = fast_screen_recording.zoom_decrease
pen_mode_toggle = fast_screen_recording.pen_mode_toggle
window_pt_top_left = fast_screen_recording.window_pt_top_left
window_pt_bottom_right = fast_screen_recording.window_pt_bottom_right
window_show_everything = fast_screen_recording.window_show_everything
sleep_awake_app = fast_screen_recording.sleep_awake_app
sleep_status = fast_screen_recording.sleep_status
update_current_keys = fast_screen_recording.update_current_keys
toggle_region_of_interest_hiding_approach = fast_screen_recording.toggle_region_of_interest_hiding_approach
save_copied_text_to_file = save_copied_text_to_file.save_content_as_pdf
utils = importlib.import_module("utils")
append_to_logs = utils.append_to_logs
settings_manager = SettingsManager()


def get_shortcut(default_shortcut,action_name):
    #I dont think this conversion from tupple to list and vice-versa is required but leaving this as is for now
    #convert tupple to a list here
    default_shortcut_key_combination = list(default_shortcut)
    final_shortcut_combination = settings_manager.get_setting(action_name, default=default_shortcut_key_combination)
    append_to_logs(f"Shortcut for {action_name} is {final_shortcut_combination}")
    return tuple(final_shortcut_combination)

# Dictionary to store shortcut combinations and their corresponding functions.
# For example: {('tab', 'z'): 'my_function'}
shortcut_actions = {
    get_shortcut(('ctrl', 'z'), 'my_function'): 'my_function',
    get_shortcut(('ctrl', 'z'), 'my_function'): 'my_function',
    get_shortcut(('ctrl', 'q'),'quit_app'): 'quit_app',
    get_shortcut(('ctrl', '='),'zoom_increase'): 'zoom_increase',
    get_shortcut(('ctrl', '-'),'zoom_decrease'): 'zoom_decrease',
    get_shortcut(('ctrl', '9'),'window_pt_top_left'): 'window_pt_top_left',#window_pt_top_left window_pt_bottom_right represents the window we want to show the user
    get_shortcut(('ctrl', '0'),'window_pt_bottom_right'): 'window_pt_bottom_right',
    get_shortcut(('9', '0'),'window_show_everything'): 'window_show_everything',
    get_shortcut(('ctrl', 'b'),'toggle_region_of_interest_hiding_approach'): 'toggle_region_of_interest_hiding_approach',
    get_shortcut(('ctrl', 'v'),'save_copied_text_to_file'): 'save_copied_text_to_file',
    get_shortcut(('ctrl', 'p'),'sleep_awake_app'): 'sleep_awake_app',
    get_shortcut(('ctrl', 'o'),'pen_mode_toggle'): 'pen_mode_toggle',
}

# Track the currently pressed keys.
current_keys = set()

# Define the functions that will be triggered.
def my_function():
    append_to_logs("Tab+Z was pressed!")

def quit_app():
    append_to_logs("user quitted the app")
    #forceful termination without calling and exit handler, try-catch-finally, does not raise any exception, simply terminates
    os._exit(0) #we need to terminate the main thread, and not the keyboard thread
    # Feature	sys.exit()	os._exit()
    # Type of Exit	Raises SystemExit (graceful)	Exits immediately (forceful)
    # Cleanup Execution	Runs cleanup handlers and finally blocks	Skips cleanup handlers and finally blocks
    # Exception Handling	Can be caught with try-except	Cannot be caught
    # Use Case	For normal program exits	For abrupt termination (e.g., child processes)

# Mapping from function name to actual function.
function_map = {
    'my_function': my_function,
    'quit_app': quit_app,
    'zoom_decrease': zoom_decrease,
    'zoom_increase': zoom_increase,
    'window_pt_top_left': window_pt_top_left,
    'window_pt_bottom_right': window_pt_bottom_right,
    'window_show_everything':window_show_everything,
    'toggle_region_of_interest_hiding_approach':toggle_region_of_interest_hiding_approach,
    'save_copied_text_to_file':save_copied_text_to_file,
    'sleep_awake_app':sleep_awake_app,
    'pen_mode_toggle':pen_mode_toggle

}

# Function to check if a shortcut is triggered.
def on_press(key):
    try:
        # Normalize the key name.
        key_name = key.char if hasattr(key, 'char') else key.name
    except AttributeError as e:
        append_to_logs("Got error on press",e)
        return

    # Add the key to the set of currently pressed keys.
    current_keys.add(key_name)
    update_current_keys(current_keys)

    # Check for any shortcut that matches the currently pressed keys.
    for shortcut, function_name in shortcut_actions.items():
        # If the keys in the shortcut match the currently pressed keys.
        if all(k in current_keys for k in shortcut):
            # Execute the corresponding function.
            # when in sleep mode- all keyboard shortcuts other than sleep shortcut will be turned off
            if function_name in function_map:
                if sleep_status() == False or function_name == 'sleep_awake_app':
                    function_map[function_name]()
                else:
                    append_to_logs("keyboard shortcut ignored because sleep mode is on",function_name)

def on_release(key):
    try:
        # Normalize the key name.
        key_name = key.char if hasattr(key, 'char') else key.name
    except AttributeError:
        return

    # Remove the key from the set of currently pressed keys.
    if key_name in current_keys:
        current_keys.remove(key_name)
        update_current_keys(current_keys)

def millisToSeconds(millis):
    return (1/1000) * millis

def listen_keyboard_events(): 
    # Start listening for keyboard events.
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while True:
            # append_to_logs("Listening for keyboard shortcuts...")
            time.sleep(millisToSeconds(10))
        # listener.join()

# listen_keyboard_events()