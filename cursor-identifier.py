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
        

    frame_count += 1

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()

# Print cursor data
for data in cursor_data:
    print(f"Frame {data[0]}: Position {data[1]}, Speed {data[2]:.2f} pixels/second")
