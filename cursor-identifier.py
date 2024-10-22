import cv2
import os
import numpy as np

# Path to the input video and the folder containing cursor templates
video_path = "/Users/sachinjeph/Desktop/CaptureDisplay/assets/cursor-movement-onscreen.mp4"
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
frame_count = 0
current_template_index = 0
next_template_index = 0
consecutive_cursor_miss_threshold = 60
current_consecutive_cursor_misses = 0

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
    threshold = 0.8
    if max_val >= threshold:
        # Get the top-left corner of the matched area
        cursor_x, cursor_y = max_loc
        #Resetting the next template we are going to check to the beginning of the template list
        next_template_index = 0
        print(f"Frame {frame_count}: Cursor detected using template '{template_name}' at position ({cursor_x}, {cursor_y}) at current_template_index {current_template_index}")
        
        # Optionally, draw a rectangle around the detected cursor for visualization
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
            next_template_index = (next_template_index + 1) % len(templates)
            print(f"Frame {frame_count}: No match found with template '{template_name}', with current_template_index {current_template_index}. Now going to check with template_index {next_template_index}")
            current_template_index = next_template_index
        else:
            current_consecutive_cursor_misses += 1
        

    frame_count += 1

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
