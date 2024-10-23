import time
import math
from Quartz import CGEventCreate, CGEventGetLocation

# Store the previous cursor position and time for speed calculation
previous_position = None
previous_time = None

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

def track_cursor():
    global previous_position, previous_time
    
    print("Tracking cursor movement. Press Ctrl+C to stop.")
    
    while True:
        try:
            # Get the current cursor position and time
            current_position = get_cursor_position()
            current_time = time.time()
            
            # Calculate speed if we have a previous position and time
            if previous_position is not None and previous_time is not None:
                time_diff = current_time - previous_time
                speed = calculate_speed(previous_position, current_position, time_diff)
                print(f"Cursor Position: {current_position}, Speed: {speed:.2f} px/s")
            else:
                print(f"Cursor Position: {current_position}")
            
            # Update previous position and time
            previous_position = current_position
            previous_time = current_time
            
            # Sleep for a short duration to reduce CPU usage (adjust as needed)
            time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("Stopped tracking.")
            break

if __name__ == "__main__":
    track_cursor()
