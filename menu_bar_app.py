import rumps
from Quartz import CGGetActiveDisplayList
import importlib
import time
import os
import subprocess

fast_screen_recording = importlib.import_module("fast-screen-recording")
screen_rec_and_mouse_click_listener = fast_screen_recording.screen_rec_and_mouse_click_listener
destroy_cv2_windows = fast_screen_recording.destroy_cv2_windows
set_input_monitor = fast_screen_recording.set_input_monitor
set_output_monitor = fast_screen_recording.set_output_monitor
sleep_awake_app = fast_screen_recording.sleep_awake_app
sleep_status = fast_screen_recording.sleep_status
start_time = None

class MonitorSelectorApp(rumps.App):
    def __init__(self):
        super(MonitorSelectorApp, self).__init__("Capture Display", icon='/Users/sachinjeph/Desktop/CaptureDisplay/assets/capturedisplay.ico')
        self.input_monitor = None
        self.output_monitor = None
        self.last_sleep_awake_status = None
        #setting self.quit_button to None which is the default quit button provided by rumps, because we will have our own quit button
        self.quit_button = None 
        self.monitor_list = self.get_monitors()
        self.update_menu()  # Initialize the menu
        self.start_loop_function_timer()
        self.show_monitor_selection_alert()
        self.update_menu()
        
        

    def menu_update_pending(self):
        
        if self.last_sleep_awake_status != sleep_status():
            self.last_sleep_awake_status = sleep_status()
            return True

    def start_loop_function_timer(self):
        global start_time
        """Start the infinite loop function timer"""
        start_time = time.time()
        rumps.timer(0.001)(self.loop_function)()

    def loop_function(self, *args):
        global start_time
        # This below screen recording method also needs to run on the main thread that is why we are running it using rumps.timer
        screen_rec_and_mouse_click_listener()
        if self.menu_update_pending():
            print("refreshing menu because of a pending refresh")
            self.refresh_menu()
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

        self.menu.add(rumps.MenuItem("Notes", callback=self.open_notes_folder))

        # Add separator and quit option
        self.menu.add(rumps.separator)

        sleep_awake_menu_label = "Awake" if sleep_status() else "Sleep"      
        self.menu.add(rumps.MenuItem(sleep_awake_menu_label, callback=self.sleep_awake_action))

        #For some reason we get a default Quit button when using rumps to create menu bar app, which gets destroyed when we refresh the menu, so at the point we need to enable our own Quit option
        #The above issue can be solved by setting the self.quit_button to None which is the default quit button provided by rumps
        # if self.input_monitor is not None or self.output_monitor is not None or self.last_sleep_awake_status is not None:
        self.menu.add(rumps.MenuItem("Quit", callback=self.quit_app))

    def sleep_awake_action(self,sender):
        print("Sleep awake action invoked from menu bar")
        sleep_awake_app()
        #Do not update the last_sleep_awake_status status here, we want the menu bar refresh logic to set that so it the menu bar refreshes
        self.refresh_menu() 

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

    def open_notes_folder(self,sender):
        # Get the path to the notes folder in the current directory
        notes_path = os.path.join(os.getcwd(), 'notes')
        
        # Create the notes folder if it doesn't exist
        if not os.path.exists(notes_path):
            os.makedirs(notes_path)
        
        # Open the folder in Finder on macOS
        subprocess.run(['open', notes_path])

# should be run on the main thread since it's a UI event loop
def menu_bar_app():
    MonitorSelectorApp().run()
