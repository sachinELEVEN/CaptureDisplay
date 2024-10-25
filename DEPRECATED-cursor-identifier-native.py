import time
import math
from Quartz import CGEventCreate, CGEventGetLocation
from pynput import mouse
from threading import Timer

# Store the previous cursor position and time for speed calculation
previous_position = None
previous_time = None

# To store the last click time and position for detecting double-clicks
click_buffer = None
double_click_threshold = 0.3  # Time in seconds to consider a double-click

def get_cursor_position():
    """
    Get the current position of the cursor on macOS.
    """
    event = CGEventCreate(None)
    location = CGEventGetLocation(event)
    return (location.x, location.y)

def calculate_speed(pos1, pos2, time_diff):
    """
    Calculate the speed of cursor movement.
    
    Args:
        pos1 (tuple): The previous position (x, y).
        pos2 (tuple): The current position (x, y).
        time_diff (float): The time difference between the two positions in seconds.
    
    Returns:
        float: The speed in pixels per second.
    """
    if time_diff <= 0:
        return 0
    distance = math.dist(pos1, pos2)
    speed = distance / time_diff
    return speed

def process_click(click_type, position):
    """
    Process the detected click type and print the event.
    
    Args:
        click_type (str): The type of click (e.g., "Left Click", "Double Click").
        position (tuple): The position (x, y) of the click.
    """
    print(f"{click_type} at {position}")

def on_click(x, y, button, pressed):
    """
    Callback function to handle mouse click events.
    
    Args:
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        button (Button): The mouse button that was clicked.
        pressed (bool): True if the button is pressed, False if released.
    """
    global click_buffer
    click_position = (x, y)

    # Consider only left-clicks for double-click detection
    if button == mouse.Button.left and pressed:
        current_time = time.time()

        if click_buffer is None:
            # If no click is in the buffer, start a timer to wait for a possible second click
            click_buffer = (current_time, click_position)
            Timer(double_click_threshold, check_for_double_click).start()
        else:
            # If there is a click in the buffer, check the time difference for a double-click
            previous_time, _ = click_buffer
            time_diff = current_time - previous_time

            if time_diff < double_click_threshold:
                # It's a double click, process it and clear the buffer
                process_click("Double Click", click_position)
                click_buffer = None
            else:
                # If the time difference exceeds the threshold, process as a single click
                process_click("Left Click", click_position)
                click_buffer = (current_time, click_position)
    elif button == mouse.Button.right and not pressed:
        process_click( "Right Click", click_position)
    elif button != mouse.Button.right and button != mouse.Button.left:
        if not pressed:
            process_click("Other Click", click_position)


def check_for_double_click():
    """
    Checks if the click buffer should be treated as a single click.
    This function is called after the double_click_threshold time has passed.
    """
    global click_buffer
    if click_buffer is not None:
        _, click_position = click_buffer
        process_click("Left Click", click_position)
        click_buffer = None

def track_cursor():
    global previous_position, previous_time
    
    print("Tracking cursor movement and mouse events. Press Ctrl+C to stop.")
    
    # Start listening for mouse events in a separate thread
    listener = mouse.Listener(on_click=on_click)
    listener.start()
    
    try:
        while True:
            # Get the current cursor position and time
            current_position = get_cursor_position()
            current_time = time.time()
            
            # Calculate speed if we have a previous position and time
            if previous_position is not None and previous_time is not None:
                time_diff = current_time - previous_time
                speed = calculate_speed(previous_position, current_position, time_diff)
                # print(f"Cursor Position: {current_position}, Speed: {speed:.2f} px/s")
            else:
                print(f"Cursor Position: {current_position}")
            
            # Update previous position and time
            previous_position = current_position
            previous_time = current_time
            
            # Sleep for a short duration to reduce CPU usage (adjust as needed)
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("Stopped tracking.")
        listener.stop()

if __name__ == "__main__":
    track_cursor()
