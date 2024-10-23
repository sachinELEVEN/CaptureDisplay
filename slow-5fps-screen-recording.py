import numpy as np
import Quartz as QZ
import cv2
import time

#This is slow in capturing the video-> each frame takes like 0.2-0.3s

class ScreenCapture:

    # constructor
    def __init__(self):
        # You can set screen dimensions if known, but typically you can retrieve them directly from the captured image.
        self.screen_width = 0
        self.screen_height = 0

    def get_screen_image(self):
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
        frame = screen_capture.get_screen_image()

        # Calculate the time taken to capture the frame
        elapsed_time = time.time() - start_time
        print(f"Time to capture frame: {elapsed_time:.4f} seconds")

        # If you want to display the frame using OpenCV (for testing purposes):
        cv2.imshow("Screen Capture", frame)
        
        # Pause for FPS
        if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
