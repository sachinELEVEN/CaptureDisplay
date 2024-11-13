import numpy as np
import Quartz as QZ
import cv2
import time
from Quartz import CGGetActiveDisplayList, CGGetOnlineDisplayList
from CoreFoundation import CFPreferencesCopyAppValue
import importlib
from pynput import mouse
from threading import Timer
import copy
from settings_file_manager import SettingsManager

settings_manager = SettingsManager()
one_time_cursor_info = importlib.import_module("one-time-cursor-info")
get_cursor_info = one_time_cursor_info.get_cursor_info
utils = importlib.import_module("utils")
get_resource_path = utils.get_resource_path
append_to_logs = utils.append_to_logs
save_copied_text_to_file = importlib.import_module("save_copied_text_to_file")
save_content_as_pdf = save_copied_text_to_file.save_content_as_pdf

# To store the last click time and position for detecting double-clicks
click_buffer = None
double_click_threshold = 0.3  # Time in seconds to consider a double-click
left_click_status = False #Press left click once to zoom, and left click again to reset zoom
# Initialize previous zoom level
prev_zoom_level = 1
default_zoom_level = 3
zoom_level_to_use = default_zoom_level
max_zoom_level = 10
min_zoom_level = 1

pt_top_left = -1 #-1 denotes uninitialized value
pt_bottom_right = -1
blur_kernel_size = 51#should be an odd number
last_in_bounds_cursor_position = (0,0)#default position is the origin
use_blur_effect = False
dimming_factor = 0#0 - blackout, 1-everything visible
initialization_done = False
screen_capture = None
mouse_event_listener = False

input_monitor = None
output_monitor = None

input_monitor_old = None
output_monitor_old = None
is_screen_augmentation_paused = False
screen_destroyed = False

pen_mode_coordinates_set_list = []
pen_mode_coordinates_curr_set = set()
pen_mode_enabled = True
#the reason for having rgb is a little stupid is that the settings manager can only understand strings so if we persist (r,g,b) as string we will need to add additional logic to convert it to a tupple of ints, so we avoid this using this simpler logic for now
pen_thickness = int(settings_manager.get_setting("pen_thickness",5))
pen_color_r = int(settings_manager.get_setting("pen_color_r",0))
pen_color_g = int(settings_manager.get_setting("pen_color_g",255))
pen_color_b = int(settings_manager.get_setting("pen_color_b",0))
#when moving across monitors pen_frame_layer will get destorted because the underlying frame with change, so you need to disable and re-enable pen mode, so pen_frame_layer is recalculated
pen_frame_layer = None
logo_watermark_path = None
cursor_img_path = "./assets/mac-cursor-4x/default@4x.png"

last_frame_displayed = None
#If display_output_mode is false then no output will be shown in the output monitor. This is generally done when you just want to screen record the input monitor and take text and screenshot notes, without having to output something to output monitor
display_output_mode = True if settings_manager.get_setting("display_output_mode","enabled")=="enabled" else False
pending_window_destroy = False

current_keys = set()
current_keys_with_history_for_certain_time = {}
#for this many iterations current_keys_with_history_for_certain_time will maintain the record of key in its map
current_keys_history_for_iterations = 200
show_pressed_keys_on_screen = True

def display_output_mode_toggle():
    global display_output_mode, output_monitor_bounds, pending_window_destroy
    display_output_mode = not display_output_mode
    if display_output_mode == False:
        output_monitor_bounds = None
        pending_window_destroy = True

    settings_manager.set_setting("display_output_mode", "enabled "if display_output_mode else "disabled")
    append_to_logs("Display output mode is toggled to ",display_output_mode)

def display_output_mode_status():
    global display_output_mode
    return display_output_mode

def take_screenshot():
    global last_frame_displayed
    if last_frame_displayed is not None:
        save_content_as_pdf(last_frame_displayed,save_text=False)
        append_to_logs("Adding screenshot to notes")
    else:
        append_to_logs("Tried to take a screenshot but last frame was null")

def pen_mode_toggle():
    global pen_mode_enabled
    pen_mode_enabled = not pen_mode_enabled
    append_to_logs("pen mode status is:",pen_mode_enabled,)

def add_keys_to_current_keys_with_history():
    global show_pressed_keys_on_screen
    if show_pressed_keys_on_screen == False:
        return
    
    global current_keys, current_keys_with_history_for_certain_time,current_keys_history_for_iterations
    # For each character in current_keys, add it to the map with value 1000
    for key in current_keys:
        current_keys_with_history_for_certain_time[key] = current_keys_history_for_iterations
    return current_keys_with_history_for_certain_time

def decrement_value_current_keys_with_history():
    global current_keys_with_history_for_certain_time,show_pressed_keys_on_screen
    
    if show_pressed_keys_on_screen == False:
        return
    # List of keys to remove after decrementing
    keys_to_remove = []
    
    # Decrement each key's value by 1, and mark for removal if it becomes 0
    for key in current_keys_with_history_for_certain_time:
        current_keys_with_history_for_certain_time[key] -= 1
        if current_keys_with_history_for_certain_time[key] == 0:
            keys_to_remove.append(key)
    
    # Remove keys with value 0
    for key in keys_to_remove:
        del current_keys_with_history_for_certain_time[key]
    
    return current_keys_with_history_for_certain_time

def update_current_keys(key_set):
    global current_keys
    current_keys = key_set
    add_keys_to_current_keys_with_history()

def is_key_pressed(key_name):
    if key_name in current_keys:
        # append_to_logs(f"is_key_pressed: {key_name}: yes")
        return True
    return False                                                                                                                                        

def set_input_monitor(id):
    global input_monitor, input_monitor_old
    if sleep_status():
        #if system is in sleep mode we cache the value of input_monitor in input_monitor_old, we probably could have made this whole thing a little better but its fine
        append_to_logs("system in sleep mode: set_input_monitor_old to ",id)
        input_monitor_old = id
    else:
        append_to_logs("set_input_monitor to ",id)
        input_monitor = id
    
    settings_manager.set_setting("input_monitor", f"{id}")

def set_output_monitor(id):
    global output_monitor, output_monitor_old
    if sleep_status():
        append_to_logs("system in sleep mode: set_input_monitor_old to ",id)
        output_monitor_old = id
    else:
        append_to_logs("set_input_monitor to ",id)
        output_monitor = id

    settings_manager.set_setting("output_monitor", f"{id}")
    

#####KEYBOARD SHORTCUT METHODS ABOVE
    
def sleep_status():
    global is_screen_augmentation_paused
    return is_screen_augmentation_paused
    
def sleep_awake_app():
    #when in sleep mode- all keyboard shortcuts other than sleep shortcut will be turned off
    global input_monitor, input_monitor_old, output_monitor, output_monitor_old, is_screen_augmentation_paused, screen_destroyed

    if is_screen_augmentation_paused:
        #we need to restart it
        input_monitor = input_monitor_old
        output_monitor = output_monitor_old
        is_screen_augmentation_paused = False
        screen_destroyed = False
        append_to_logs(f"restarting screen augmentation at input monitor {input_monitor}, and output monitor {output_monitor}")
    else:
        append_to_logs("Pausing screen augmentation")
        input_monitor_old = input_monitor
        input_monitor = None
        output_monitor_old = output_monitor
        output_monitor = None
        #we will destroy the screen from the main thread
        is_screen_augmentation_paused = True


def toggle_region_of_interest_hiding_approach():
    global use_blur_effect
    use_blur_effect = not use_blur_effect

#Methods to show only a particular section of the screen aka window cropping methods

#First top left corner should be provided only then bottom right can be provided
def window_show_everything():
    global pt_top_left, pt_bottom_right
    append_to_logs("showing entire screen")
    #resetting the points
    pt_top_left = -1
    pt_bottom_right = -1

#Do we need to check if this points are in bounds or not?? i dont think so because our blur region will probably be not be on the frame
def window_pt_top_left():
    global pt_top_left,pt_bottom_right, input_monitor
    append_to_logs("window_pt_top_left")
    pt_top_left =  get_cursor_info()["position"]
    input_monitor_bounds = QZ.CGDisplayBounds(int(input_monitor))
    pt_top_left = scaleAccordingToInputDisplayFactor((screen_capture.screen_width,screen_capture.screen_height),input_monitor_bounds,pt_top_left)
    append_to_logs("window_pt_top_left is ",pt_top_left)


def window_pt_bottom_right():
    global pt_top_left,pt_bottom_right,input_monitor
    if pt_top_left == -1:
        append_to_logs("pt_top_left needs to be set first")
        return   
    append_to_logs("window_pt_bottom_right")
    pt_bottom_right =  get_cursor_info()["position"]
    input_monitor_bounds = QZ.CGDisplayBounds(int(input_monitor))
    pt_bottom_right = scaleAccordingToInputDisplayFactor((screen_capture.screen_width,screen_capture.screen_height),input_monitor_bounds,pt_bottom_right)
    append_to_logs("window_pt_bottom_right is ",pt_bottom_right)
    


#these need to check if zoom is supported via -left_click_status
def zoom_increase():
    global zoom_level_to_use, left_click_status

    if not left_click_status:
        append_to_logs("Zoom mode is disabled- double click to enable it")
        return
   
    if zoom_level_to_use < max_zoom_level:
        append_to_logs("zoom increased to",zoom_level_to_use)
        zoom_level_to_use +=  0.5
    else:
        append_to_logs("max zoom level reached",zoom_level_to_use)

def zoom_decrease():
    global zoom_level_to_use, left_click_status

    if not left_click_status:
        append_to_logs("Zoom mode is disabled- double click to enable it")
        return
    
    if zoom_level_to_use > min_zoom_level:
        append_to_logs("zoom decreased to",zoom_level_to_use)
        zoom_level_to_use -= 0.5
    else:
        append_to_logs("min zoom level reached",zoom_level_to_use)



#####KEYBOARD SHORTCUT METHODS ABOVE

#This is slow in capturing the video-> each frame takes like 0.2-0.3s
#Now when we just take 1 monitor frame -> we are getting like 2000FPS, But this is just input video, we still need to process it

class ScreenCapture:

    # constructor
    def __init__(self):
        # You can set screen dimensions if known, but typically you can retrieve them directly from the captured image.
        self.screen_width = 0
        self.screen_height = 0

    #Captures a particular monitor
    #if display_ids_provided it means the input_monitor_index is actually input_monitor_id and same for output_monitor_index is output_monitor_id
    def get_monitor_screen_image(self, input_monitor_index=0,output_monitor_index=0,display_ids_provided=False):
        global display_output_mode
        if display_ids_provided==False:
            # maximum number of displays to return
            max_displays = 100
            # get active display list
            # CGGetActiveDisplayList:
            #     Provides a list of displays that are active (or drawable).
            (err, active_displays, number_of_active_displays) = CGGetActiveDisplayList(
                max_displays, None, None)
            if err:
                return False
            # append_to_logs(active_displays,number_of_active_displays)
            # Get the desired monitor's bounds
            # append_to_logs("active displays",number_of_active_displays,"and",active_displays)
            if input_monitor_index < number_of_active_displays and output_monitor_index < number_of_active_displays:
                input_monitor_id = active_displays[input_monitor_index]  # Choose which monitor to capture
                output_monitor_id = active_displays[output_monitor_index] #monitor to which we will display the output- should ideally be the virtual monitor
            else:
                raise ValueError("Monitor index out of range.")
        else:
            input_monitor_id = input_monitor_index
            output_monitor_id = output_monitor_index

        input_monitor_bounds = QZ.CGDisplayBounds(input_monitor_id)
        #this can crash when input monitor is not there
        if display_output_mode and output_monitor_id is not None: 
            output_monitor_bounds = QZ.CGDisplayBounds(output_monitor_id)
        else:
            output_monitor_bounds = None

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

# Function to draw on the frame using given coordinates
def draw_pen_mode(frame, only_draw_recent_line=False, draw_lines=True, color=(0, 255, 0), thickness=5):
    """
    Draws lines or dots on a frame using specified coordinates.
    
    Parameters:
        frame (numpy.ndarray): The frame on which to draw.
        input_monitor_bounds (list of tuple): parameter used for scaling points for different screen/monitor resolution
        only_draw_recent_line: If true we only draw a line using last 2 points in the current coordinate set, this should be true for an layer with existing pen drawing, this is more performant
        draw_lines (bool): If True, draws lines between points; otherwise, draws dots.
        color (tuple): The color of the lines or dots.
        thickness (int): Thickness of lines or size of dots.
    """
    global pen_mode_coordinates_set_list, pen_mode_coordinates_curr_set, screen_capture
   
    
    if only_draw_recent_line:
        full_list = []
        if len(pen_mode_coordinates_curr_set) >= 2:
            full_list.append(pen_mode_coordinates_curr_set)
    else:  
        #create a combined list where you append pen_mode_coordinates_curr_set at the end of full_list  
        full_list = pen_mode_coordinates_set_list
        full_list.append(pen_mode_coordinates_curr_set)

    if len(full_list)>0:
        pass
        # append_to_logs("Draw lines using: ",len(full_list)," Draw mode is only_draw_recent_line:",only_draw_recent_line)
    # append_to_logs(full_list)
    for coordinates in full_list:
        # Draw lines if draw_lines is True
        # Sort coordinates by time (third element in each tuple)
        # convert to a list since set does not maintain order and we need order to draw lines
        # append_to_logs("inside",coordinates[0])
        sorted_coordinates = sorted(coordinates, key=lambda coord: coord[2])
        #need to scale and normalize the coordinate here (in that order, we should always scale first and then normalize(x,y translation))
        for i, coordinate in enumerate(sorted_coordinates):
            sorted_coordinates[i] = scaleAccordingToInputDisplayFactor((screen_capture.screen_width, screen_capture.screen_height), QZ.CGDisplayBounds(int(input_monitor)), coordinate)
            sorted_coordinates[i] = normalize_coordinate_to_0_0_origin(sorted_coordinates[i],QZ.CGDisplayBounds(int(input_monitor)))[0]

        if draw_lines and len(sorted_coordinates) > 1:
            for i in range(1, len(sorted_coordinates)):
                # Draw a line between each consecutive pair of points
                cv2.line(frame, (int(sorted_coordinates[i - 1][0]),int(sorted_coordinates[i - 1][1])),  (int(sorted_coordinates[i][0]),int(sorted_coordinates[i][1])), color, thickness)
        # else:
        #     # Draw dots if draw_lines is False or only one point is present
        #     for point in coordinates:
        #         cv2.circle(frame, (int(point[0]),int(point[1])), thickness, color, -1)
    
    return frame

def overlay_image_on_frame(frame, image_path, top_left_x, top_left_y, overlay_width=None, overlay_height=None, is_image_path_absolute=False):
    """
    Overlays an image onto a given frame at specified coordinates, with optional resizing.

    :param frame: The background frame (numpy array).
    :param image_path: Path to the image to overlay.
    :param top_left_x: X-coordinate of the top-left corner where the image should be placed.
    :param top_left_y: Y-coordinate of the top-left corner where the image should be placed.
    :param overlay_width: The desired width of the overlay image (only used if is_image_path_absolute is True).
    :param overlay_height: The desired height of the overlay image (only used if is_image_path_absolute is True).
    :param is_image_path_absolute: Boolean indicating if image_path is an absolute path.
    :return: The frame with the image overlaid.
    """
    if not is_image_path_absolute:
        image_path = get_resource_path(image_path)
    top_left_x = int(top_left_x)
    top_left_y = int(top_left_y)

    # Load the overlay image from the given path.
    overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Resize overlay image if required
    if is_image_path_absolute and overlay_width and overlay_height:
        overlay = cv2.resize(overlay, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)

    # If the overlay image has an alpha channel, we need to split it.
    if overlay.shape[2] == 4:
        overlay_rgb = overlay[:, :, :3]  # Extract RGB channels.
        alpha_channel = overlay[:, :, 3]  # Extract the alpha channel.
    else:
        overlay_rgb = overlay
        alpha_channel = np.ones((overlay.shape[0], overlay.shape[1]), dtype=np.uint8) * 255

    # Determine the region where the image will be placed.
    h, w = overlay_rgb.shape[:2]
    bottom_right_x = top_left_x + w
    bottom_right_y = top_left_y + h

    # Ensure that the coordinates are within the frame boundaries.
    if top_left_x < 0 or top_left_y < 0 or bottom_right_x > frame.shape[1] or bottom_right_y > frame.shape[0]:
        #append_to_logs("The overlay image is out of frame bounds. Not adding overlay.")
        return frame

    # Extract the region of interest (ROI) from the frame.
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Blend the overlay image with the ROI using the alpha channel as a mask.
    alpha_channel = alpha_channel[:, :, np.newaxis] / 255.0
    blended = (overlay_rgb * alpha_channel + roi * (1 - alpha_channel)).astype(np.uint8)

    # Place the blended image back into the frame.
    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = blended

    return frame


def display_characters_on_frame(frame, characters, font_scale=3, thickness=5, color=(255, 255, 255), position=(50,100), spacing=100):
    """
    Displays each character from a dictionary on a given frame in large, bold text, sorted by the integer value associated with each character.

    :param frame: The background frame (numpy array).
    :param characters: A dictionary where keys are characters and values are integers.
    :param font_scale: Font scale for the text.
    :param thickness: Thickness of the text.
    :param color: Color of the text in BGR format (default is white).
    :param position: Starting position (top-left corner) for the first character.
    :param spacing: Spacing between characters (in pixels).
    :return: The frame with the characters overlaid.
    """
    # Sort the characters based on their corresponding integer values (ascending order)
    sorted_characters = sorted(characters.items(), key=lambda item: item[1])

    # Loop through each character and display it on the frame
    x, y = position
    for char, _ in sorted_characters:

        if char == "space":
            char = ' '

        cv2.putText(
            frame,                # Frame to draw on
            char,              # Character to display
            (x, y),               # Position to place the character
            cv2.FONT_HERSHEY_SIMPLEX, # Font type
            font_scale,           # Font scale for size
            color,                # Color of the text
            thickness,            # Thickness of the text
            lineType=cv2.LINE_AA  # Anti-aliased line for better quality
        )
        # Update x-position for the next character, moving right by 'spacing' pixels

        x += spacing + spacing*int(len(char)/2)
    
    return frame


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
    # append_to_logs(position)
    # append_to_logs(bound_x1,bound_y1, "and", bound_x2,bound_y2)
    return bound_x1 <= cursor_x <= bound_x2 and bound_y1 <= cursor_y <= bound_y2

#cursor position and input_monitor_bounds and normalizes them to 0,0 origin, this is done because zooming, drawing rectangles only works for positive coordinates
def normalize_coordinate_to_0_0_origin(cursor_position,input_monitor_bounds):
    x_offset = (0 - input_monitor_bounds.origin.x)
    y_offset = (0 - input_monitor_bounds.origin.y)
    input_monitor_bounds.origin.x = input_monitor_bounds.origin.x + x_offset
    input_monitor_bounds.origin.y = input_monitor_bounds.origin.y + y_offset

    # append_to_logs(cursor_position)
    cursor_position = (cursor_position[0] + x_offset,
                  cursor_position[1] + y_offset)
   
    return (cursor_position,input_monitor_bounds)

def blur_except_region(frame,input_monitor_bounds):
    global pt_top_left, pt_bottom_right, blur_kernel_size

    top_left = pt_top_left
    bottom_right = pt_bottom_right
    #check if frame needs to be blurred
    if not (pt_bottom_right != -1 and pt_top_left != -1):
        return frame

    #Normalize the top_left and bottom_right coordinates
    result1 = normalize_coordinate_to_0_0_origin(top_left,copy.deepcopy(input_monitor_bounds))
    top_left = result1[0]
    result2 = normalize_coordinate_to_0_0_origin(bottom_right,input_monitor_bounds)
    bottom_right = result2[0]
    # append_to_logs("Normalized cropped section coordinates are",top_left,bottom_right)
    # input_monitor_bounds = result[1]

    # Create a mask with the same dimensions as the frame, initialized to zeros (black)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Define the region to keep (rectangle) with white color (255)
    top_left = (int(top_left[0]), int(top_left[1]))        # (x, y) coordinates of the top-left corner
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))   # (x, y) coordinates of the bottom-right corner
    cv2.rectangle(mask, top_left, bottom_right, 255, thickness=cv2.FILLED)

    # Blur the entire frame
    blurred_frame = cv2.GaussianBlur(frame, (blur_kernel_size, blur_kernel_size), 0)

    # Create an inverse mask to get the area to blur
    inverse_mask = cv2.bitwise_not(mask)

    # Use the inverse mask to keep only the blurred area
    blurred_area = cv2.bitwise_and(blurred_frame, blurred_frame, mask=inverse_mask)

    # Use the original frame where the mask is white (to keep the original region)
    original_area = cv2.bitwise_and(frame, frame, mask=mask)

    # Combine the original area and the blurred area
    final_output = cv2.add(original_area, blurred_area)

    return final_output


def dim_except_region(frame, input_monitor_bounds):
    global dimming_factor
    """
    Dims or blacks out all parts of the frame except a specified region.

    Parameters:
        frame (numpy.ndarray): The input image/frame.
        input_monitor_bounds (tuple): Monitor bounds for normalization.
        dimming_factor (float): A value between 0 (completely black) and 1 (no dimming).

    Returns:
        numpy.ndarray: The modified frame with dimmed regions.
    """
    global pt_top_left, pt_bottom_right

    top_left = pt_top_left
    bottom_right = pt_bottom_right

    # Check if the frame needs to be dimmed.
    if not (pt_bottom_right != -1 and pt_top_left != -1):
        return frame

    # Normalize the top_left and bottom_right coordinates.
    result1 = normalize_coordinate_to_0_0_origin(top_left, copy.deepcopy(input_monitor_bounds))
    top_left = result1[0]
    result2 = normalize_coordinate_to_0_0_origin(bottom_right, input_monitor_bounds)
    bottom_right = result2[0]

    # Create a mask with the same dimensions as the frame, initialized to zeros (black).
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Define the region to keep (rectangle) with white color (255).
    top_left = (int(top_left[0]), int(top_left[1]))  # (x, y) coordinates of the top-left corner.
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))  # (x, y) coordinates of the bottom-right corner.
    cv2.rectangle(mask, top_left, bottom_right, 255, thickness=cv2.FILLED)

    # Create a dimmed version of the frame.
    dimmed_frame = (frame * dimming_factor).astype(np.uint8)

    # Use the mask to keep the original area where the mask is white.
    original_area = cv2.bitwise_and(frame, frame, mask=mask)

    # Use the inverse mask to apply the dimmed frame to the rest of the area.
    inverse_mask = cv2.bitwise_not(mask)
    dimmed_area = cv2.bitwise_and(dimmed_frame, dimmed_frame, mask=inverse_mask)

    # Combine the original area and the dimmed area.
    final_output = cv2.add(original_area, dimmed_area)

    return final_output


def perform_zoom_augmentation(frame,cursor_info,input_monitor_bounds,output_monitor_bounds):
    global left_click_status, prev_zoom_level, last_in_bounds_cursor_position, use_blur_effect, pen_mode_enabled, pen_mode_coordinates_curr_set, pen_frame_layer, last_frame_displayed
    global logo_watermark_path, cursor_img_path, show_pressed_keys_on_screen,current_keys_with_history_for_certain_time
    # Now, iterate through cursor_data and zoom in at cursor positions with speed less than threshold
    position = cursor_info["position"]
    speed = cursor_info["speed"]
    show_rectangle_overlay = False
    input_monitor_bounds_unnormalized = copy.deepcopy(input_monitor_bounds)
    result = normalize_coordinate_to_0_0_origin(position,input_monitor_bounds)
    position = result[0]
    input_monitor_bounds = result[1]
    # append_to_logs("Normalised and unnormalised are",input_monitor_bounds,input_monitor_bounds_unnormalized)
    # append_to_logs("Normalised->",position,input_monitor_bounds)
    cursor_in_bounds = False
    #Validate cursor position- basically we need to check if cursor is on the same monitor as we are interested in or not
    if not is_cursor_within_bounds(position,input_monitor_bounds):
        # append_to_logs("Cursor is not within bounds")
        #Making speed 0 ensures we do not perform any zooming in the augmented frames, so input and output frame will be same with no additional frames being generated
        speed = 0
        # return
    else:
        last_in_bounds_cursor_position = position
        cursor_in_bounds = True
    
    # Define the speed threshold (adjust as needed)
    speed_threshold = 5000  # Pixels/second

    # Open the video file again to extract the frames for visualization
    # video = cv2.VideoCapture(video_path)


    frame_num = -1
    show_processed_video_preview = True
    if speed < speed_threshold:
        # append_to_logs(f"Zooming in at Frame {frame_num}: Position {position}, Speed {speed:.2f} pixels/second")

        # Set the frame position to the one where we want to zoom in
        # video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        # ret, frame = video.read()

        if frame is not None:
            # Assuming cursor_x and cursor_y are your cursor's position
            cursor_x, cursor_y = last_in_bounds_cursor_position

            #zoom level on basis of left click toggling
            if cursor_in_bounds:
                target_zoom_level = zoom_level_to_use if left_click_status else 1
            else:
                #you can't change the zoom state if cursor is not in bounds
                target_zoom_level = prev_zoom_level
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
                append_to_logs("Compared zoomed level",prev_zoom_level, "and", target_zoom_level)
                good_cursor_speed = 100#in pixels per frame
                zoom_steps = max(int(speed/good_cursor_speed),1)
                zoom_steps = 5 #when left click zoom is enabled
            # append_to_logs("Zooming animation steps->",zoom_steps," zoom level",target_zoom_level)
            zoom_levels = smooth_zoom(prev_zoom_level, target_zoom_level, steps=zoom_steps)
            # append_to_logs(zoom_levels)
            #Frame needs to be processed here before we display it
            if use_blur_effect:
                only_show_region_of_interest_frame = blur_except_region(frame,input_monitor_bounds_unnormalized)
            else:
                only_show_region_of_interest_frame = dim_except_region(frame,input_monitor_bounds_unnormalized)
            
            #Add logo watermark overlay
            if  logo_watermark_path is not None:
                frame_with_logo_watermark_layer_overlay = overlay_image_on_frame(only_show_region_of_interest_frame, logo_watermark_path, only_show_region_of_interest_frame.shape[1] - 100, only_show_region_of_interest_frame.shape[0] - 100, 50, 50, is_image_path_absolute=cursor_img_path != logo_watermark_path)

            #Add cursor overlay on it
            frame_with_cursor = overlay_image_on_frame(frame_with_logo_watermark_layer_overlay,"./assets/mac-cursor-4x/default@4x.png",cursor_x-20,cursor_y-20)
            
            #Add pen mode drawings- this looks like a wrong approach because we are redrawing at all the points again on every frame, without keeping anything from our memory
            if pen_mode_enabled and pen_frame_layer is not None:
                #overlay the pen_frame_layer
                #put this in try catch and on error reset the pen_frame_layer so that it gets recomputed
                try:
                    # append_to_logs("overlay pen frame")
                    frame_with_pen_layer_overlay = cv2.addWeighted(frame_with_cursor, 1, pen_frame_layer, 1, 0)
                # frame_with_pen_mode = draw_pen_mode(frame_with_cursor)
                except Exception as e:
                    #this generally happens when the input monitor is changed because at that point the dimensions of frame_with_cursor is according to the new input monitor but pen_frame_layer's dimensions are still according to the old monitor, so it gives a dimension mismatch exception, so in that case we just recalculate the pen_frame_layer according to the new input layer
                    append_to_logs("Exception in drawing mode, will reset the pen_frame_layer:",e)
                    pen_frame_layer = None
                    #for this frame calculation we will not overlay any pen layer
                    frame_with_pen_layer_overlay = frame_with_cursor
            else:
                frame_with_pen_layer_overlay = frame_with_cursor
            # Apply zoom for each interpolated zoom level
            # Need to do this only when zoom level has changed
            for zoom in zoom_levels:
                # display_characters_on_frame
                zoomed_frame = zoom_at(frame_with_pen_layer_overlay, zoom=zoom, angle=angle, coord=(cursor_x, cursor_y))
                if show_processed_video_preview:
                    # Optionally, draw a rectangle around the detected cursor
                    # cv2.rectangle(zoomed_frame, (int(0), int(0)), (int(0) + 50, int(0) + 50), (0, 255, 0), 2)
                    if show_rectangle_overlay and cursor_in_bounds:
                        cv2.rectangle(zoomed_frame, (int(cursor_x), int(cursor_y)), (int(cursor_x) + 50, int(cursor_y) + 50), (0, 255, 0), 2)
                    
                    if show_pressed_keys_on_screen:
                        zoomed_frame_with_chars = display_characters_on_frame(zoomed_frame,current_keys_with_history_for_certain_time)
                    else:
                        zoomed_frame_with_chars = zoomed_frame

                    #Frame needs to be processed here before we display it
                    last_frame_displayed = zoomed_frame_with_chars
                    display_frame_at_required_monitor(zoomed_frame_with_chars,output_monitor_bounds)
                    # cv2.imshow("Zoomed Frame", zoomed_frame)

                # Write the zoomed frame to the output video
                # output_video.write(zoomed_frame)
                # output_video.write(zoomed_frame)
                # processed_frames.append(zoomed_frame)

                # Break on 'q' key
                # if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
                #     break

            # Update the previous zoom level for the next iteration
            # append_to_logs("setting prev_zoom_level to ",target_zoom_level)
            prev_zoom_level = target_zoom_level

        else:
            append_to_logs(f"Received null frame while trying to augment the frame {frame_num}")


#Displays the frame at the correct display
def display_frame_at_required_monitor(frame,output_monitor_bounds):
    # If you want to display the frame using OpenCV (for testing purposes):
    if output_monitor_bounds is None:
        append_to_logs("Failed to output the frame because output monitor bounds is null. Probably toggling of display_output_mode happened recently")
        return

    window_name = "Capture Display"
    cv2.imshow(window_name, frame)
    cv2.moveWindow(window_name,int(output_monitor_bounds.origin.x),int(output_monitor_bounds.origin.y))
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        append_to_logs("key entered")

#Cursor clicks methods
        
def process_click(click_type, position):
    """
    Process the detected click type and print the event.
    
    Args:
        click_type (str): The type of click (e.g., "Left Click", "Double Click").
        position (tuple): The position (x, y) of the click.
    """
    # append_to_logs(f"{click_type} at {position}")

def on_click(x, y, button, pressed):
    """
    Callback function to handle mouse click events.
    
    Args:
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        button (Button): The mouse button that was clicked.
        pressed (bool): True if the button is pressed, False if released.
    """
    global click_buffer, left_click_status, zoom_level_to_use
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
                if not sleep_status() and display_output_mode_status():
                    left_click_status = not left_click_status
                else:
                    append_to_logs("zoom mode cannot be enabled/disabled in sleep mode or when display_output_mode is off")
                if left_click_status:
                    zoom_level_to_use = default_zoom_level
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

#Depends on the points on the screen and size of the frame
def scaleAccordingToInputDisplayFactor(frame_size,input_monitor_bounds,point_to_scale):
    x_scale_factor = frame_size[0]/input_monitor_bounds.size.width
    y_scale_factor = frame_size[1]/input_monitor_bounds.size.height
    point_to_scale = (point_to_scale[0] * int(x_scale_factor),point_to_scale[1] * int(y_scale_factor))
    # append_to_logs(point_to_scale)
    return point_to_scale

# This is used if you want to run the screen recording by calling this function on the main thread, but this will occupy the main thread, preventing anyother even from getting executed. Hence when we now use the menu bar app we call the screen_rec_and_mouse_click_listener which is called multiple times and does not occupy the main thread because screen_rec_and_mouse_click_listener does not contain any infinite loop
def screen_rec_and_mouse_click_listener_deprecated():
    screen_capture = ScreenCapture()

    # Start listening for mouse events in a separate thread
    listener = mouse.Listener(on_click=on_click)
    listener.start()

    
    append_to_logs("Starting to screen screen recording loop")
    while True:
        # append_to_logs("hello")
        start_time = time.time()  # Start the timer
        #This basically takes a ss of the screen and converts into a frame which can then be used by OpenCV for further analysis
        result = screen_capture.get_monitor_screen_image(1,2)
        frame = result[0]
        input_monitor_bounds = result[1]

         #get cursor info
        cursor_info = get_cursor_info()
        cursor_info["position"] = scaleAccordingToInputDisplayFactor((screen_capture.screen_width,screen_capture.screen_height),input_monitor_bounds,cursor_info["position"])
        input_monitor_bounds.size.width,input_monitor_bounds.size.height  = scaleAccordingToInputDisplayFactor((screen_capture.screen_width,screen_capture.screen_height),input_monitor_bounds,input_monitor_bounds.size)
        #should we scale output display factor as well? - not doing it for not seeing any weird results there as of now
        output_monitor_bounds = result[2]
        # append_to_logs("after scaling",input_monitor_bounds)
        # break
        # Calculate the time taken to capture the frame
        
        # append_to_logs(f"FPS: {60 * 1 / elapsed_time:.4f}")
        # append_to_logs(input_monitor_bounds)

        #Augmentation of the frame
       
        # append_to_logs("Cursor info is",cursor_info)
        perform_zoom_augmentation(frame,cursor_info,input_monitor_bounds,output_monitor_bounds)
        elapsed_time = time.time() - start_time
        append_to_logs(f"FPS: {  1 / elapsed_time:.4f}")
        # display_frame_at_required_monitor(frame,output_monitor_bounds)

        #we dont want too many reading to be done because then zoom abruption will be higher simply because you are sampling at a super high frequency
        #actually when we use left click based zoom, we want it to have high frequency, so changes are picked up quickly
        # if cv2.waitKey(1000) & 0xFF == ord('q'):
        #     break

            
    cv2.destroyAllWindows()


def setup():
    global screen_capture, mouse_event_listener,initialization_done,logo_watermark_path, cursor_img_path

    if initialization_done:
        return
    
    append_to_logs("initializing screen_rec_and_mouse_click_listener")
    screen_capture = ScreenCapture()
    # Start listening for mouse events in a separate thread
    mouse_event_listener = mouse.Listener(on_click=on_click)
    mouse_event_listener.start()

    logo_watermark_path = settings_manager.get_setting("logo_watermark_path_png","")
    if logo_watermark_path == "":
        append_to_logs("logo_watermark_path was not provided using the CaptureDisplay logo as the watermark")
        logo_watermark_path = cursor_img_path

    initialization_done = True

def screen_rec_and_mouse_click_listener():
    global screen_capture, mouse_event_listener, input_monitor, output_monitor, is_screen_augmentation_paused, screen_destroyed, pen_mode_enabled, pen_mode_coordinates_curr_set, pen_mode_coordinates_set_list, pen_frame_layer, display_output_mode, last_frame_displayed, pending_window_destroy
    setup()

    # append_to_logs("Starting to screen screen recording loop")
    if True:#replaced while True

        if pending_window_destroy:
            pending_window_destroy = False
            append_to_logs("Pending window destroy so destroying output window")
            destroy_cv2_windows()
            

        if is_screen_augmentation_paused and screen_destroyed==False:
            #we need to destroy the window from the main thread
            destroy_cv2_windows()
            screen_destroyed = True
            return
        # append_to_logs("hello")
        # start_time = time.time()  # Start the timer
        #This basically takes a ss of the screen and converts into a frame which can then be used by OpenCV for further analysis
        if input_monitor is None:
            # append_to_logs("Please select input monitor from the menu bar")
            cv2.waitKey(1)
            return

        

        result = screen_capture.get_monitor_screen_image(int(input_monitor),int(output_monitor) if (display_output_mode and output_monitor is not None) else None, display_ids_provided=True)
        frame = result[0]

        if display_output_mode and output_monitor is None:
            # append_to_logs("Please select output monitor from the menu bar")
            last_frame_displayed = frame
            cv2.waitKey(1)
            return
        if display_output_mode==False:
            #this will be our last frame we took in input, so last_frame_displayed should be this even though we are not displaying it technically, but its the last frame our pipeline saw, so this will be used in case user want to take a screen shot 
            last_frame_displayed = frame
            return


        input_monitor_bounds = result[1]

        input_monitor_bounds_unscaled = copy.deepcopy(input_monitor_bounds)

         #get cursor info
        cursor_info = get_cursor_info()
        cursor_info_unscaled = cursor_info.copy()
        cursor_info["position"] = scaleAccordingToInputDisplayFactor((screen_capture.screen_width,screen_capture.screen_height),input_monitor_bounds,cursor_info["position"])
        input_monitor_bounds.size.width,input_monitor_bounds.size.height  = scaleAccordingToInputDisplayFactor((screen_capture.screen_width,screen_capture.screen_height),input_monitor_bounds,input_monitor_bounds.size)
        #should we scale output display factor as well? - not doing it for not seeing any weird results there as of now
        output_monitor_bounds = result[2]
        # append_to_logs("after scaling",input_monitor_bounds)
        # break
        # Calculate the time taken to capture the frame
        
        # append_to_logs(f"FPS: {60 * 1 / elapsed_time:.4f}")
        # append_to_logs(input_monitor_bounds)
        is_tab_pressed = is_key_pressed('alt')
        if pen_mode_enabled:

            if is_tab_pressed:
                #we should probably add a condition so that we do not draw when cursor is out of bounds but ignoring that for now
                pen_mode_coordinates_curr_set.add((cursor_info_unscaled["position"][0],cursor_info_unscaled["position"][1],time.time()))
            else:
                #clear the current coordinate set
                if len(pen_mode_coordinates_curr_set)>1:#we compare it with 1 because we need at least 2 points to draw a line
                    pen_mode_coordinates_set_list.append(pen_mode_coordinates_curr_set)
                pen_mode_coordinates_curr_set = set()

            #please note when you switch monitor we preserve the drawing wrt to the original input monitor's coordinates, so you may not see the previous drawings on new monitor but when you switch back to the old input monitor you will see your drawings there remain intact
            #so our drawings are preserved wrt input monitor coordinates across all monitors, allowing you to draw different stuff on different input monitors
            if pen_frame_layer is None:
                append_to_logs("Creating pen frame layer")
                # Step 1: Create the base layer with dots
                pen_frame_layer = np.zeros((screen_capture.screen_height, screen_capture.screen_width, 3), dtype=np.uint8)
                pen_frame_layer = draw_pen_mode(pen_frame_layer,color=(pen_color_r,pen_color_g,pen_color_b),thickness=pen_thickness)
            else:
                #modifying existing pen_frame_layer
                # append_to_logs("modifying pen_frame_layer")
                pen_frame_layer = draw_pen_mode(pen_frame_layer,only_draw_recent_line=True,color=(pen_color_r,pen_color_g,pen_color_b),thickness=pen_thickness)       
        else:
            #clear the current coordinate set
            if len(pen_mode_coordinates_curr_set)>1:#we compare it with 1 because we need at least 2 points to draw a line
                pen_mode_coordinates_set_list.append(pen_mode_coordinates_curr_set)
            pen_mode_coordinates_curr_set = set()
            pen_frame_layer = None
        #Augmentation of the frame

        decrement_value_current_keys_with_history()
        # append_to_logs("Cursor info is",cursor_info)
        perform_zoom_augmentation(frame,cursor_info,input_monitor_bounds,output_monitor_bounds)
        # elapsed_time = time.time() - start_time
        # append_to_logs(f"FPS: {  1 / elapsed_time:.4f}") Here FPS should be calculated from Menu bar app's event loop and not from here for this function
        # display_frame_at_required_monitor(frame,output_monitor_bounds)

        #we dont want too many reading to be done because then zoom abruption will be higher simply because you are sampling at a super high frequency
        #actually when we use left click based zoom, we want it to have high frequency, so changes are picked up quickly
        # if cv2.waitKey(1000) & 0xFF == ord('q'):
        #     break
        # time.sleep(millisToSeconds(10)) - cannot put main thread to sleep

            
    # cv2.destroyAllWindows()

def destroy_cv2_windows():
    append_to_logs("destroying cv2 windows")
    cv2.destroyAllWindows()