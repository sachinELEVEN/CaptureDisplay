import sys
import objc
from Cocoa import NSApplication, NSStatusBar, NSMenu, NSMenuItem, NSApp
from Quartz import CGGetActiveDisplayList

class MenuBarApp:
    def __init__(self):
        # Create the application instance
        self.app = NSApplication.sharedApplication()

        # Create a status bar item
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(-1)
        self.status_item.setTitle_("Monitor Selector")

        # Create a menu
        self.menu = NSMenu.alloc().init()

        # Get the list of monitors
        self.monitor_list = self.get_monitors()
        
        # Add input monitor selection
        self.input_menu_item = NSMenuItem.alloc().initWithTitle("Select Input Monitor", action=None, keyEquivalent="")
        self.menu.addItem(self.input_menu_item)
        for idx, monitor in enumerate(self.monitor_list):
            item = NSMenuItem.alloc().initWithTitle(monitor, action=self.select_input_monitor, keyEquivalent="")
            item.setTag(idx)
            self.menu.addItem(item)

        self.menu.addItem(NSMenuItem.separatorItem())
        
        # Add output monitor selection
        self.output_menu_item = NSMenuItem.alloc().initWithTitle("Select Output Monitor", action=None, keyEquivalent="")
        self.menu.addItem(self.output_menu_item)
        for idx, monitor in enumerate(self.monitor_list):
            item = NSMenuItem.alloc().initWithTitle(monitor, action=self.select_output_monitor, keyEquivalent="")
            item.setTag(idx)
            self.menu.addItem(item)

        self.menu.addItem(NSMenuItem.separatorItem())
        
        # Add quit option
        quit_item = NSMenuItem.alloc().initWithTitle("Quit", action=self.quit_app, keyEquivalent="")
        self.menu.addItem(quit_item)

        # Set the menu to the status item
        self.status_item.setMenu(self.menu)
        
    def get_monitors(self):
        """Get a list of available monitors."""
        max_displays = 10  # Maximum number of displays
        (err, active_displays, number_of_active_displays) = CGGetActiveDisplayList(max_displays, None, None)
        if err:
            return []

        monitor_names = []
        for display_id in active_displays:
            display_name = display_id #CGDisplayCopyDisplayName(display_id)
            monitor_names.append(display_name)

        return monitor_names

    def select_input_monitor(self, sender):
        """Handle input monitor selection."""
        monitor_idx = sender.tag()
        print(f"Input Monitor Selected: {self.monitor_list[monitor_idx]}")

    def select_output_monitor(self, sender):
        """Handle output monitor selection."""
        monitor_idx = sender.tag()
        print(f"Output Monitor Selected: {self.monitor_list[monitor_idx]}")

    def quit_app(self, sender):
        """Quit the application."""
        NSApp.terminate_(self)

if __name__ == "__main__":
    app = MenuBarApp()
    NSApp.run()
