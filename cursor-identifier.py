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
show_processed_video_preview = False
processed_frames = []

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
# video.release()
# cv2.destroyAllWindows()

# Print cursor data
for data in cursor_data:
    print(f"Frame {data[0]}: Position {data[1]}, Speed {data[2]:.2f} pixels/second")

####ZOOM PREVIEW HERE

# Zoomed video preview
def zoom_at(img, zoom=1, angle=0, coord=None):
    # Set the center of zoom to the center of the image if coord is None
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
    
    # Create the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    
    # Apply the warpAffine function to zoom the image
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result

def DEPRECATED_calculate_zoom_level(speed, max_speed=5000):
    """
    Calculate zoom level based on speed.
    
    :param speed: The speed of the cursor in pixels/second.
    :param max_speed: The maximum speed at which zoom level is at its minimum (1).
    :return: A zoom level between 1 and 3.
    """
    # Ensure speed is not negative
    speed = max(speed, 0)
    
    # Calculate zoom level inversely related to speed
    zoom_level = 3 - (2 * (speed / max_speed))
    
    # Clamp the zoom level to be between 1 and 3
    zoom_level = max(1, min(zoom_level, 3))
    
    return zoom_level

# Function to smoothly transition the zoom level
def smooth_zoom(current_zoom, target_zoom, steps=10):
    return np.linspace(current_zoom, target_zoom, steps)

# Define the speed threshold (adjust as needed)
speed_threshold = 5000  # Pixels/second

# Open the video file again to extract the frames for visualization
# video = cv2.VideoCapture(video_path)

# Initialize previous zoom level
prev_zoom_level = 1

#Saving the output file locally
# Setup VideoWriter to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
output_file_path = "processed_output_video.mp4"
output_video = cv2.VideoWriter(output_file_path, fourcc, video_fps, (frame.shape[1], frame.shape[0]))

# Now, iterate through cursor_data and zoom in at cursor positions with speed less than threshold
for frame_num, position, speed in cursor_data:
    if speed < speed_threshold:
        print(f"Zooming in at Frame {frame_num}: Position {position}, Speed {speed:.2f} pixels/second")

        # Set the frame position to the one where we want to zoom in
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()

        if ret:
            # Assuming cursor_x and cursor_y are your cursor's position
            cursor_x, cursor_y = position

            # Calculate zoom level based on speed
            target_zoom_level = zoom_level = 1 if speed < 100 else 3  # Dynamic zoom based on speed
            angle = 0  # No rotation

            # Smoothly interpolate zoom levels via n no. of zoom steps
            #we only want smooth transition when zoom level has changed, since adding zoom steps increases video size, we dont want to increase the video size needlessly and increasing zoom steps also slows down the video between those frames
            #I think levels should maybe according to the cursor speed, so we should normalize it for cursor speed
            #Assuming 300 is a good speed smooth and understandably cursor speed
            good_cursor_speed = 100#in pixels per frame
            zoom_steps = min(int(speed/good_cursor_speed),1)
            # zoom_steps = 3
            print("Zooming animation steps->",zoom_steps)
            # if prev_zoom_level != target_zoom_level:
            #     zoom_steps = 10
            zoom_levels = smooth_zoom(prev_zoom_level, target_zoom_level, steps=zoom_steps)

            # Apply zoom for each interpolated zoom level
            # Need to do this only when zoom level has changed
            for zoom in zoom_levels:
                zoomed_frame = zoom_at(frame, zoom=zoom, angle=angle, coord=(cursor_x, cursor_y))
                if show_processed_video_preview:
                    cv2.imshow("Zoomed Frame", zoomed_frame)

                # Write the zoomed frame to the output video
                # output_video.write(zoomed_frame)
                output_video.write(zoomed_frame)
                # processed_frames.append(zoomed_frame)

                # Break on 'q' key
                if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
                    break

            # Update the previous zoom level for the next iteration
            prev_zoom_level = target_zoom_level

        else:
            print(f"Could not read frame {frame_num}")

# Release the video capture and output writer
video.release()


# Release the output writer
output_video.release()
print(f"Processed video saved as: {output_file_path}")

# Cleanup
cv2.destroyAllWindows()
