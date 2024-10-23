import numpy as np
import Quartz as QZ
import cv2
import time
from Quartz import CGGetActiveDisplayList, CGGetOnlineDisplayList
from CoreFoundation import CFPreferencesCopyAppValue


#This is slow in capturing the video-> each frame takes like 0.2-0.3s
#Now when we just take 1 monitor frame -> we are getting like 2000FPS, But this is just input video, we still need to process it

class ScreenCapture:

    # constructor
    def __init__(self):
        # You can set screen dimensions if known, but typically you can retrieve them directly from the captured image.
        self.screen_width = 0
        self.screen_height = 0

    #Captures a particular monitor
    def get_monitor_screen_image(self, monitor_index=0):
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
        if monitor_index < number_of_active_displays:
            monitor_id = active_displays[monitor_index]  # Choose which monitor to capture
        else:
            raise ValueError("Monitor index out of range.")

        monitor_bounds = QZ.CGDisplayBounds(monitor_id)

        # Capture only the specified monitor using its bounds
        core_graphics_image = QZ.CGWindowListCreateImage(
            monitor_bounds,  # Use the bounds of the specific monitor
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

        return final_output
    

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

# Example usage
if __name__ == "__main__":
    screen_capture = ScreenCapture()
    while True:
        start_time = time.time()  # Start the timer
        #This basically takes a ss of the screen and converts into a frame which can then be used by OpenCV for further analysis
        frame = screen_capture.get_monitor_screen_image(1)

        # Calculate the time taken to capture the frame
        elapsed_time = time.time() - start_time
        print(f"FPS: {60 * 1 / elapsed_time:.4f}")

        # If you want to display the frame using OpenCV (for testing purposes):
        cv2.imshow("Screen Capture", frame)
        
        # Pause for FPS
        if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
