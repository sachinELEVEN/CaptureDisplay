import numpy as np
import Quartz as QZ
import cv2
import time
from Quartz import CGGetActiveDisplayList, CGGetOnlineDisplayList
from CoreFoundation import CFPreferencesCopyAppValue
import importlib
from pynput import mouse
from threading import Timer

one_time_cursor_info = importlib.import_module("one-time-cursor-info")
get_cursor_info = one_time_cursor_info.get_cursor_info

# To store the last click time and position for detecting double-clicks
click_buffer = None
double_click_threshold = 0.3  # Time in seconds to consider a double-click
left_click_status = False #Press left click once to zoom, and left click again to reset zoom
# Initialize previous zoom level
prev_zoom_level = 1

#This is slow in capturing the video-> each frame takes like 0.2-0.3s
#Now when we just take 1 monitor frame -> we are getting like 2000FPS, But this is just input video, we still need to process it

class ScreenCapture:

    # constructor
    def __init__(self):
        # You can set screen dimensions if known, but typically you can retrieve them directly from the captured image.
        self.screen_width = 0
        self.screen_height = 0

    #Captures a particular monitor
    def get_monitor_screen_image(self, input_monitor_index=0,output_monitor_index=0):
        # maximum number of displays to return
        max_displays = 100
        # get active display list
        # CGGetActiveDisplayList:
        #     Provides a list of displays that are active (or drawable).
        (err, active_displays, number_of_active_displays) = CGGetActiveDisplayList(
            max_displays, None, None)
        if err:
            return False
        print(active_displays,number_of_active_displays)
        # Get the desired monitor's bounds
        if input_monitor_index < number_of_active_displays and output_monitor_index < number_of_active_displays:
            input_monitor_id = active_displays[input_monitor_index]  # Choose which monitor to capture
            output_monitor_id = active_displays[output_monitor_index] #monitor to which we will display the output- should ideally be the virtual monitor
        else:
            raise ValueError("Monitor index out of range.")

        input_monitor_bounds = QZ.CGDisplayBounds(input_monitor_id)
        output_monitor_bounds = QZ.CGDisplayBounds(output_monitor_id)

        # Capture only the specified monitor using its bounds
        core_graphics_image = QZ.CGWindowListCreateImage(
            input_monitor_bounds,  # Use the bounds of the specific monitor
            QZ.kCGWindowListOptionOnScreenOnly,  # Capture only visible screen elements
            QZ.kCGNullWindowID,  # Ignore specific windows and focus on the specified monitor
            QZ.kCGWindowImageDefault
        )

        # Extract image properties
        bytes_per_row = QZ.CGImageGetBytesPerRow(core_graphics_image)
        width = QZ.CGImageGetWidth(core_graphics_image)
        height = QZ.CGImageGetHeight(core_graphics_image)

        # Set class properties for width and height
        self.screen_width = width
        self.screen_height = height

        # Extract pixel data from the image
        core_graphics_data_provider = QZ.CGImageGetDataProvider(core_graphics_image)
        core_graphics_data = QZ.CGDataProviderCopyData(core_graphics_data_provider)

        np_raw_data = np.frombuffer(core_graphics_data, dtype=np.uint8)

        # Convert the raw data to a numpy array that OpenCV can process
        numpy_data = np.lib.stride_tricks.as_strided(
            np_raw_data,
            shape=(height, width, 3),
            strides=(bytes_per_row, 4, 1),
            writeable=False
        )

        final_output = np.ascontiguousarray(numpy_data, dtype=np.uint8)

        return (final_output,input_monitor_bounds,output_monitor_bounds)
    

    #Captures the entire screen - across all the monitors
    def get_entire_screen_image(self):
        # Capture the entire screen using CGWindowListCreateImage
        core_graphics_image = QZ.CGWindowListCreateImage(
            QZ.CGRectInfinite,  # Use CGRectInfinite to capture the entire screen
            QZ.kCGWindowListOptionOnScreenOnly,  # Capture only visible screen elements
            QZ.kCGNullWindowID,  # Ignore specific windows and focus on the whole screen
            QZ.kCGWindowImageDefault
        )

        # Extract image properties
        bytes_per_row = QZ.CGImageGetBytesPerRow(core_graphics_image)
        width = QZ.CGImageGetWidth(core_graphics_image)
        height = QZ.CGImageGetHeight(core_graphics_image)

        # Set class properties for width and height
        self.screen_width = width
        self.screen_height = height

        # Extract pixel data from the image
        core_graphics_data_provider = QZ.CGImageGetDataProvider(core_graphics_image)
        core_graphics_data = QZ.CGDataProviderCopyData(core_graphics_data_provider)

        np_raw_data = np.frombuffer(core_graphics_data, dtype=np.uint8)

        # Convert the raw data to a numpy array that OpenCV can process
        numpy_data = np.lib.stride_tricks.as_strided(
            np_raw_data,
            shape=(height, width, 3),
            strides=(bytes_per_row, 4, 1),
            writeable=False
        )

        final_output = np.ascontiguousarray(numpy_data, dtype=np.uint8)

        return final_output

#Augments the frame by adjusting the zoom level about the cursor based on the cursor movement and returns a list of frames
    
# Zoomed video preview
def zoom_at(img, zoom=1, angle=0, coord=None):
    # Set the center of zoom to the center of the image if coord is None
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
    
    # Create the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    
    # Apply the warpAffine function to zoom the image
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result

# Function to smoothly transition the zoom level
def smooth_zoom(current_zoom, target_zoom, steps=10):
    return np.linspace(current_zoom, target_zoom, steps)

def is_cursor_within_bounds(position,input_monitor_bounds):
    cursor_x, cursor_y = position
    bound_x1 = input_monitor_bounds.origin.x
    bound_y1 = input_monitor_bounds.origin.y
    bound_x2 = input_monitor_bounds.origin.x + input_monitor_bounds.size.width
    bound_y2 = input_monitor_bounds.origin.y + input_monitor_bounds.size.height
    # print(position)
    # print(bound_x1,bound_y1, "and", bound_x2,bound_y2)
    return bound_x1 <= cursor_x <= bound_x2 and bound_y1 <= cursor_y <= bound_y2

#cursor position and input_monitor_bounds and normalizes them to 0,0 origin, this is done because zooming, drawing rectangles only works for positive coordinates
def normalize_coordinate_to_0_0_origin(cursor_position,input_monitor_bounds):
    x_offset = (0 - input_monitor_bounds.origin.x)
    y_offset = (0 - input_monitor_bounds.origin.y)
    input_monitor_bounds.origin.x = input_monitor_bounds.origin.x + x_offset
    input_monitor_bounds.origin.y = input_monitor_bounds.origin.y + y_offset

    print(cursor_position)
    cursor_position = (cursor_position[0] + x_offset,
                  cursor_position[1] + y_offset)
   
    return (cursor_position,input_monitor_bounds)


def perform_zoom_augmentation(frame,cursor_info,input_monitor_bounds,output_monitor_bounds):
    global left_click_status, prev_zoom_level
    # Now, iterate through cursor_data and zoom in at cursor positions with speed less than threshold
    position = cursor_info["position"]
    speed = cursor_info["speed"]
    show_rectangle_overlay = True
    result = normalize_coordinate_to_0_0_origin(position,input_monitor_bounds)
    position = result[0]
    input_monitor_bounds = result[1]
    # print("Normalised->",position,input_monitor_bounds)
    cursor_in_bounds = False
    #Validate cursor position- basically we need to check if cursor is on the same monitor as we are interested in or not
    if not is_cursor_within_bounds(position,input_monitor_bounds):
        print("Cursor is not within bounds")
        #Making speed 0 ensures we do not perform any zooming in the augmented frames, so input and output frame will be same with no additional frames being generated
        speed = 0
        # return
    else:
        cursor_in_bounds = True
    
    # Define the speed threshold (adjust as needed)
    speed_threshold = 5000  # Pixels/second

    # Open the video file again to extract the frames for visualization
    # video = cv2.VideoCapture(video_path)


    frame_num = -1
    show_processed_video_preview = True
    if speed < speed_threshold:
        print(f"Zooming in at Frame {frame_num}: Position {position}, Speed {speed:.2f} pixels/second")

        # Set the frame position to the one where we want to zoom in
        # video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        # ret, frame = video.read()

        if frame is not None:
            # Assuming cursor_x and cursor_y are your cursor's position
            cursor_x, cursor_y = position

            #zoom level on basis of left click toggling
            target_zoom_level = 3 if left_click_status else 1
            # Calculate zoom level based on speed
            # target_zoom_level = zoom_level = 3 if speed > 10 and speed < 5000 else 1  # Dynamic zoom based on speed
            angle = 0  # No rotation

            # Smoothly interpolate zoom levels via n no. of zoom steps
            #we only want smooth transition when zoom level has changed, since adding zoom steps increases video size, we dont want to increase the video size needlessly and increasing zoom steps also slows down the video between those frames
            #I think levels should maybe according to the cursor speed, so we should normalize it for cursor speed
            #Assuming 200 is a good speed smooth and understandably cursor speed
            # good_cursor_speed = 100#in pixels per frame
            # zoom_steps = min(int(speed/good_cursor_speed),1)
            zoom_steps = 1
            if prev_zoom_level != target_zoom_level:
                print("Compared zoomed level",prev_zoom_level, "and", target_zoom_level)
                good_cursor_speed = 100#in pixels per frame
                zoom_steps = max(int(speed/good_cursor_speed),1)
                zoom_steps = 5 #when left click zoom is enabled
            print("Zooming animation steps->",zoom_steps," zoom level",target_zoom_level)
            zoom_levels = smooth_zoom(prev_zoom_level, target_zoom_level, steps=zoom_steps)
            # print(zoom_levels)
            # Apply zoom for each interpolated zoom level
            # Need to do this only when zoom level has changed
            for zoom in zoom_levels:
                zoomed_frame = zoom_at(frame, zoom=zoom, angle=angle, coord=(cursor_x, cursor_y))
                if show_processed_video_preview:
                    # Optionally, draw a rectangle around the detected cursor
                    # cv2.rectangle(zoomed_frame, (int(0), int(0)), (int(0) + 50, int(0) + 50), (0, 255, 0), 2)
                    if show_rectangle_overlay and cursor_in_bounds:
                        cv2.rectangle(zoomed_frame, (int(cursor_x), int(cursor_y)), (int(cursor_x) + 50, int(cursor_y) + 50), (0, 255, 0), 2)
                    display_frame_at_required_monitor(zoomed_frame,output_monitor_bounds)
                    # cv2.imshow("Zoomed Frame", zoomed_frame)

                # Write the zoomed frame to the output video
                # output_video.write(zoomed_frame)
                # output_video.write(zoomed_frame)
                # processed_frames.append(zoomed_frame)

                # Break on 'q' key
                if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
                    break

            # Update the previous zoom level for the next iteration
            print("setting prev_zoom_level to ",target_zoom_level)
            prev_zoom_level = target_zoom_level

        else:
            print(f"Received null frame while trying to augment the frame {frame_num}")


#Displays the frame at the correct display
def display_frame_at_required_monitor(frame,output_monitor_bounds):
    # If you want to display the frame using OpenCV (for testing purposes):
    window_name = "Screen Capture"
    cv2.imshow(window_name, frame)
    cv2.moveWindow(window_name,int(output_monitor_bounds.origin.x),int(output_monitor_bounds.origin.y))
        
    # Pause for FPS
    if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
        print("key entered")

#Cursor clicks methods
        
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
    global click_buffer, left_click_status
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
                left_click_status = not left_click_status
                click_buffer = None
            else:
                # If the time difference exceeds the threshold, process as a single click
                process_click("Left Click", click_position)
                # left_click_status = not left_click_status
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
    global click_buffer, left_click_status
    if click_buffer is not None:
        _, click_position = click_buffer
        process_click("Left Click", click_position)
        # left_click_status = not left_click_status
        click_buffer = None

# Example usage
if __name__ == "__main__":
    screen_capture = ScreenCapture()

    # Start listening for mouse events in a separate thread
    listener = mouse.Listener(on_click=on_click)
    listener.start()
    while True:
        start_time = time.time()  # Start the timer
        #This basically takes a ss of the screen and converts into a frame which can then be used by OpenCV for further analysis
        result = screen_capture.get_monitor_screen_image(1,2)
        frame = result[0]
        input_monitor_bounds = result[1]
        output_monitor_bounds = result[2]

        # Calculate the time taken to capture the frame
        elapsed_time = time.time() - start_time
        print(f"FPS: {60 * 1 / elapsed_time:.4f}")
        # print(input_monitor_bounds)

        #Augmentation of the frame
        #get cursor info
        cursor_info = get_cursor_info()
        # print("Cursor info is",cursor_info)
        perform_zoom_augmentation(frame,cursor_info,input_monitor_bounds,output_monitor_bounds)

        #we dont want too many reading to be done because then zoom abruption will be higher simply because you are sampling at a super high frequency
        #actually when we use left click based zoom, we want it to have high frequency, so changes are picked up quickly
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            
    cv2.destroyAllWindows()
