import Cocoa
from Cocoa import NSApplication, NSApp, NSEvent, NSWindow, NSWindowStyleMask, NSMakeRect, NSObject
import objc

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        # Create a window to detect gestures
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100.0, 100.0, 400.0, 300.0),
            Cocoa.NSWindowStyleMaskClosable | Cocoa.NSWindowStyleMaskTitled | Cocoa.NSWindowStyleMaskResizable,
            Cocoa.NSBackingStoreBuffered,
            False
        )
        self.window.setTitle_("Pinch Detector")
        self.window.makeKeyAndOrderFront_(None)

    def magnifyWithEvent_(self, event):
        # Called when a pinch gesture is detected
        magnification = event.magnification()
        if magnification > 0:
            print("Zooming In")
        elif magnification < 0:
            print("Zooming Out")

if __name__ == "__main__":
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    NSApp.run()
