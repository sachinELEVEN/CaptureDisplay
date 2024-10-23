import time
import math
from Quartz import CGEventCreate, CGEventGetLocation
from pynput import mouse

# Store the previous cursor position and time for speed calculation
previous_position = None
previous_time = None

# To store the last click time for detecting double-clicks
last_click_time = 0
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

def on_click(x, y, button, pressed):
    """
    Callback function to handle mouse click events.
    
    Args:
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        button (Button): The mouse button that was clicked.
        pressed (bool): True if the button is pressed, False if released.
    """
    global last_click_time
    current_time = time.time()
    event_type = "Press" if pressed else "Release"

    # Identify the type of click
    if button == mouse.Button.left:
        click_type = "Left Click"
    elif button == mouse.Button.right:
        click_type = "Right Click"
    else:
        click_type = "Other Click"

    # Check for double-click
    if event_type == "Release" and click_type == "Left Click":
        time_since_last_click = current_time - last_click_time
        if time_since_last_click < double_click_threshold:
            click_type = "Double Click"
        last_click_time = current_time
    
    if event_type == 'Release':
        print(f"{click_type} at ({x}, {y}) - {event_type}")

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
