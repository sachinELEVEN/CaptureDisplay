import rumps
from Quartz import CGGetActiveDisplayList
import importlib
import time

fast_screen_recording = importlib.import_module("fast-screen-recording")
screen_rec_and_mouse_click_listener = fast_screen_recording.screen_rec_and_mouse_click_listener
destroy_cv2_windows = fast_screen_recording.destroy_cv2_windows
set_input_monitor = fast_screen_recording.set_input_monitor
set_output_monitor = fast_screen_recording.set_output_monitor
start_time = None

class MonitorSelectorApp(rumps.App):
    def __init__(self):
        super(MonitorSelectorApp, self).__init__("Capture Display", icon=None)
        self.input_monitor = None
        self.output_monitor = None
        self.monitor_list = self.get_monitors()
        self.update_menu()  # Initialize the menu
        self.start_loop_function_timer()
        self.show_monitor_selection_alert()

    def start_loop_function_timer(self):
        global start_time
        """Start the infinite loop function timer"""
        start_time = time.time()
        rumps.timer(0.001)(self.loop_function)()

    def loop_function(self, *args):
        global start_time
        # This below screen recording method also needs to run on the main thread that is why we are running it using rumps.timer
        screen_rec_and_mouse_click_listener()
        elapsed_time = time.time() - start_time
        # print(f"FPS: {1 / elapsed_time:.4f}")
        start_time = time.time()

    def get_monitors(self):
        """Get a list of available monitors."""
        max_displays = 10  # Maximum number of displays
        (err, active_displays, number_of_active_displays) = CGGetActiveDisplayList(max_displays, None, None)
        if err:
            return []

        monitor_names = []
        for display_id in active_displays:
            display_name = str(display_id)  # Convert display_id to string for display
            monitor_names.append(display_name)

        return monitor_names

    def update_menu(self):
        """Refresh the menu with current monitor selections."""
        # Clear the current menu
        self.menu.clear()
        
        # Add Input Monitor header
        input_monitor_label = f"Input Monitor ({self.input_monitor})" if self.input_monitor else "Input Monitor"
        input_menu = rumps.MenuItem(input_monitor_label)
        
        for monitor in self.monitor_list:
            selected = " (Selected)" if monitor == self.input_monitor else ""
            input_menu.add(rumps.MenuItem(f"{monitor}{selected}", callback=self.select_input_monitor))
        
        self.menu.add(input_menu)

        # Add Output Monitor header
        output_monitor_label = f"Output Monitor ({self.output_monitor})" if self.output_monitor else "Output Monitor"
        output_menu = rumps.MenuItem(output_monitor_label)
        
        for monitor in self.monitor_list:
            selected = " (Selected)" if monitor == self.output_monitor else ""
            output_menu.add(rumps.MenuItem(f"{monitor}{selected}", callback=self.select_output_monitor))
        
        self.menu.add(output_menu)

        # Add Shortcuts section
        shortcuts_menu = rumps.MenuItem("Shortcuts")
        shortcuts = {
            'Double click': 'Enable/disable zoom mode',
            ('ctrl', '+      '): 'Zoom in when in zoom mode',
            ('ctrl', '-       '): 'Zoom out when in zoom mode',
            ('ctrl', '(       '): 'Top left corner of the screen section to be visible',
            ('ctrl', ')       '): 'Bottom right corner of the screen section to be visible',
            ('(', ')           '): 'Show entire screen',
            ('ctrl', 'b      '): 'Switch between blur and complete blackout',
            ('ctrl', 'v      '): 'Save copied text to a markdown file',
            ('ctrl', 'p      '): 'Sleep/awake',
            ('ctrl', 'q      '): 'Quit',
        }

        for keys, description in shortcuts.items():
            keys_display = " ".join(keys) if isinstance(keys, tuple) else keys
            shortcuts_menu.add(rumps.MenuItem(f"{keys_display}   {description}"))

        self.menu.add(shortcuts_menu)

        # Add separator and quit option
        self.menu.add(rumps.separator)
        #For some reason we get a default Quit button when using rumps to create menu bar app, which gets destroyed when we refresh the menu, so at the point we need to enable our own Quit option
        if self.input_monitor is not None or self.output_monitor is not None:
            self.menu.add(rumps.MenuItem("Quit", callback=self.quit_app))

    def select_input_monitor(self, sender):
        print("Input monitor is", sender.title)
        self.input_monitor = sender.title.split(" ")[0]  # Get monitor name without (Selected)
        set_input_monitor(self.input_monitor)
        self.refresh_menu()  # Refresh the menu after selection

    def select_output_monitor(self, sender):
        print("Output monitor is", sender.title)
        self.output_monitor = sender.title.split(" ")[0]  # Get monitor name without (Selected)
        set_output_monitor(self.output_monitor)
        self.refresh_menu()  # Refresh the menu after selection

    def show_monitor_selection_alert(self):
        if self.input_monitor is None or self.output_monitor is None:
            rumps.alert(f"Capture Display\n\nSelect your Input and Output Display from the menu bar. \n\n\n Input Display- The screen where your content is.\n\nOutput Display- The screen you’ll need to share with others during the call.\n\n Capture Display will monitor the input display, apply enhancements, and present the final result on the output display. Simply share the output display screen with others during the call.")

    def refresh_menu(self):
        self.update_menu()  # Call to refresh the menu

    def quit_app(self, _):
        destroy_cv2_windows()
        rumps.quit_application()

# should be run on the main thread since it's a UI event loop
def menu_bar_app():
    MonitorSelectorApp().run()
