import numpy as np
import Quartz as QZ
import cv2
import time
from Quartz import CGGetActiveDisplayList
from CoreFoundation import CFPreferencesCopyAppValue
import AppKit
from PyObjCTools import AppHelper

class ScreenCapture:
    def __init__(self):
        self.screen_width = 0
        self.screen_height = 0

    def get_monitor_screen_image(self, monitor_index=0):
        max_displays = 100
        (err, active_displays, number_of_active_displays) = CGGetActiveDisplayList(
            max_displays, None, None)
        if err:
            return False
        print(active_displays, number_of_active_displays)
        if monitor_index < number_of_active_displays:
            monitor_id = active_displays[monitor_index]
        else:
            raise ValueError("Monitor index out of range.")

        monitor_bounds = QZ.CGDisplayBounds(monitor_id)

        core_graphics_image = QZ.CGWindowListCreateImage(
            monitor_bounds,
            QZ.kCGWindowListOptionOnScreenOnly,
            QZ.kCGNullWindowID,
            QZ.kCGWindowImageDefault
        )

        bytes_per_row = QZ.CGImageGetBytesPerRow(core_graphics_image)
        width = QZ.CGImageGetWidth(core_graphics_image)
        height = QZ.CGImageGetHeight(core_graphics_image)

        self.screen_width = width
        self.screen_height = height

        core_graphics_data_provider = QZ.CGImageGetDataProvider(core_graphics_image)
        core_graphics_data = QZ.CGDataProviderCopyData(core_graphics_data_provider)

        np_raw_data = np.frombuffer(core_graphics_data, dtype=np.uint8)

        numpy_data = np.lib.stride_tricks.as_strided(
            np_raw_data,
            shape=(height, width, 3),
            strides=(bytes_per_row, 4, 1),
            writeable=False
        )

        final_output = np.ascontiguousarray(numpy_data, dtype=np.uint8)

        return final_output


class DisplayWindow:
    def __init__(self, display_id):
        self.display_id = display_id
        self.window = None
        self.create_window()

    def create_window(self):
        screen = AppKit.NSScreen.screens()[self.display_id]
        screen_frame = screen.frame()

        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            screen_frame,
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False
        )
        self.window.setLevel_(AppKit.NSMainMenuWindowLevel + 1)
        self.window.setBackgroundColor_(AppKit.NSColor.blackColor())
        self.window.setOpaque_(True)
        self.window.makeKeyAndOrderFront_(None)

    def display_frame(self, frame):
        height, width, _ = frame.shape

        # Convert OpenCV frame (BGR) to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a bitmap image representation
        image_rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            None, width, height, 8, 3, False, False, AppKit.NSCalibratedRGBColorSpace, width * 3, 24
        )

        # Copy the frame data into the bitmap
        frame_data = frame_rgb.tobytes()
        frame_data_memoryview = memoryview(image_rep.bitmapData())
        frame_data_memoryview[:] = frame_data

        # Create an NSImage from the bitmap representation
        image = AppKit.NSImage.alloc().initWithSize_((width, height))
        image.addRepresentation_(image_rep)

        # Update the window content
        image_view = AppKit.NSImageView.alloc().initWithFrame_(self.window.contentView().frame())
        image_view.setImage_(image)
        self.window.contentView().addSubview_(image_view)
        AppHelper.callLater(0.01, image_view.display)


# Example usage
if __name__ == "__main__":
    screen_capture = ScreenCapture()
    display_window = DisplayWindow(display_id=0)  # Display window on monitor with ID 4

    while True:
        start_time = time.time()

        frame = screen_capture.get_monitor_screen_image(monitor_index=1)

        elapsed_time = time.time() - start_time
        print(f"FPS: {60 * 1 / elapsed_time:.4f}")

        display_window.display_frame(frame)

        # Pause for FPS
        if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
            break

    AppHelper.runEventLoop()  # Run the macOS event loop for window updates
