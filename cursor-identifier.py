import cv2
import os
import numpy as np

# Path to the input video and the folder containing cursor templates
standard_cursor_vid = "/Users/sachinjeph/Desktop/CaptureDisplay/assets/standard-cursor-movement.mp4"
standard_and_type_cursor_vid = "/Users/sachinjeph/Desktop/CaptureDisplay/assets/standard-and-type-cursor-movement.mp4"

video_path = standard_cursor_vid
#Order of template in the cursor png folder should be from most likely to least likely, so standard cursor should be at the beginning
template_folder = "./assets/mac-cursor-1x"


# Load all template images from the folder in grayscale and sort by file name. Its a sorted list because order of template in the cursor png folder should be from most likely to least likely, so standard cursor should be at the beginning
templates = []
for file_name in sorted(os.listdir(template_folder)):
    if file_name.endswith(".png"):
        template_path = os.path.join(template_folder, file_name)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates.append((template, file_name))
print(f"Loaded {len(templates)} templates.")


# Open the video file
video = cv2.VideoCapture(video_path)
video_fps = 60
frame_count = 0
current_template_index = 0
next_template_index = 0
#considering a 60fps video this means 1sec of consecutive cursor missing, we probably should determine it from the video
consecutive_cursor_miss_threshold = 150
current_consecutive_cursor_misses = 0
cursor_found = False
show_cursor_identifier_preview = False

# Variables for calculating speed
previous_position = None
cursor_data = []  # List to hold frame number, cursor position, and speed

# Iterate over the video frames
while True:
    ret, frame = video.read()
    if not ret:
        break  # End of the video

    # Convert frame to grayscale (for matching templates)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the current template for detection
    template, template_name = templates[current_template_index]
    template_height, template_width = template.shape

    # Use template matching to find the cursor
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

    # Find the location with the highest match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Set a threshold for detecting a match
    threshold = 0.7
    if max_val >= threshold:
        # Get the top-left corner of the matched area
        cursor_x, cursor_y = max_loc
        current_position = (cursor_x + template_width // 2, cursor_y + template_height // 2)  # Center of the cursor

        # Resetting the next template to the beginning of the template list
        next_template_index = 0
        current_consecutive_cursor_misses = 0
        if not cursor_found:
            cursor_found = True
            print(f"Frame {frame_count}: Cursor detected using template '{template_name}' at position {current_position} with confidence {max_val}")

        # Calculate speed if previous_position exists
        if previous_position is not None:
            distance = np.sqrt((current_position[0] - previous_position[0]) ** 2 + (current_position[1] - previous_position[1]) ** 2)
            time_between_frames = 1 / video_fps  # 60 fps
            speed = distance / time_between_frames
        else:
            speed = -1

        # Store frame data (frame number, position, speed)
        cursor_data.append((frame_count, current_position, speed))

        # Update previous_position for the next frame
        previous_position = current_position

        # Optionally, draw a rectangle around the detected cursor
        cv2.rectangle(frame, (cursor_x, cursor_y), 
                      (cursor_x + template_width, cursor_y + template_height), 
                      (0, 255, 0), 2)
        
        # Show the frame with the detected cursor (for debugging)
        if show_cursor_identifier_preview:
            cv2.imshow("Cursor Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        # No match found, move to the next template
        if current_consecutive_cursor_misses > consecutive_cursor_miss_threshold:
            current_consecutive_cursor_misses = 0
            cursor_found = False
            next_template_index = (next_template_index + 1) % len(templates)
            print(f"Frame {frame_count}: No match found with template '{template_name}', with current_template_index {current_template_index}. Now going to check with template_index {next_template_index}")
            current_template_index = next_template_index
        else:   
            current_consecutive_cursor_misses += 1
            print(f"Cursor cursor misses for template: {current_template_index} with count {current_consecutive_cursor_misses}")
        
    if frame_count>500:#remove this later
        break
    frame_count += 1

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()

# Print cursor data
for data in cursor_data:
    print(f"Frame {data[0]}: Position {data[1]}, Speed {data[2]:.2f} pixels/second")

#Zoomed video preview
#Zooming logic picked from here - https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
def zoom_at(img, zoom=1, angle=0, coord=None):
    # Set the center of zoom to the center of the image if coord is None
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
    
    # Create the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    
    # Apply the warpAffine function to zoom the image
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result

def calculate_zoom_level(speed, max_speed=5000):
    """
    Calculate zoom level based on speed.
    
    :param speed: The speed of the cursor in pixels/second.
    :param max_speed: The maximum speed at which zoom level is at its minimum (1).
    :return: A zoom level between 1 and 3.
    """
    # Ensure speed is not negative
    speed = max(speed, 0)
    
    # Calculate zoom level inversely related to speed
    # The formula ensures zoom level is between 1 and 3
    zoom_level = 3 - (2 * (speed / max_speed))
    
    # Clamp the zoom level to be between 1 and 3
    zoom_level = max(1, min(zoom_level, 3))
    
    return zoom_level
# Define the speed threshold (adjust as needed)
speed_threshold = 5000  # Pixels/second

# Open the video file again to extract the frames for visualization
video = cv2.VideoCapture(video_path)

# Now, iterate through cursor_data and zoom in at cursor positions with speed less than threshold
for frame_num, position, speed in cursor_data:
    if speed < speed_threshold:
        print(f"Zooming in at Frame {frame_num}: Position {position}, Speed {speed:.2f} pixels/second")

        # Set the zoom level and size of the zoomed area
        # zoom_scale = 2  # Zoom factor zoom_size = max(800 + speed/100,2500)  # Size of 
        # zoom_size = 800  # Size of the area around the cursor to zoom into

        # Set the frame position to the one where we want to zoom in
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()

        # Assuming you have your frame and cursor position
        if ret:
            # Assuming cursor_x and cursor_y are your cursor's position
            cursor_x, cursor_y = position

            # Define zoom level and angle if needed
            #Zoom level should depend on the speed
            #0 speed -> zoom in, more speed - zoom out
            zoom_level = calculate_zoom_level(speed)  # 2x zoom
            angle = 0  # No rotation

            # Zoom into the frame at the cursor position
            zoomed_frame = zoom_at(frame, zoom=zoom_level, angle=angle, coord=(cursor_x, cursor_y))

            # Show the zoomed frame
            cv2.imshow("Zoomed Frame", zoomed_frame)

            # Break on 'q' key
            if cv2.waitKey(int(1000/60)) & 0xFF == ord('q'):
                break

        else:
            print(f"Frame {frame_num} could not be read.")

# Release the video capture
video.release()
cv2.destroyAllWindows()
