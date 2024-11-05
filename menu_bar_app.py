import rumps
from Quartz import CGGetActiveDisplayList
import importlib
import time
import os
import subprocess
import sys
from settings_file_manager import SettingsManager
from datetime import datetime

utils = importlib.import_module("utils")
get_resource_path = utils.get_resource_path
fast_screen_recording = importlib.import_module("fast-screen-recording")
screen_rec_and_mouse_click_listener = fast_screen_recording.screen_rec_and_mouse_click_listener
destroy_cv2_windows = fast_screen_recording.destroy_cv2_windows
set_input_monitor = fast_screen_recording.set_input_monitor
set_output_monitor = fast_screen_recording.set_output_monitor
sleep_awake_app = fast_screen_recording.sleep_awake_app
sleep_status = fast_screen_recording.sleep_status
display_output_mode_toggle = fast_screen_recording.display_output_mode_toggle
display_output_mode_status = fast_screen_recording.display_output_mode_status
start_time = None
utils = importlib.import_module("utils")
append_to_logs = utils.append_to_logs
settings_manager = SettingsManager()


class MonitorSelectorApp(rumps.App):
    def __init__(self):
        super(MonitorSelectorApp, self).__init__("Capture Display", icon='/Users/sachinjeph/Desktop/CaptureDisplay/assets/CaptureDisplayX.ico')
        self.input_monitor = None
        self.output_monitor = None
        self.last_sleep_awake_status = None
        self.last_display_mode_status = None
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
        
        if self.last_display_mode_status != display_output_mode_status():
            self.last_display_mode_status = display_output_mode_status()
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
            append_to_logs("refreshing menu because of a pending refresh")
            self.refresh_menu()
        elapsed_time = time.time() - start_time
        # append_to_logs(f"FPS: {1 / elapsed_time:.4f}")
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

    def get_shortcut(self,action_name):
        #the output of result is like this ['ctrl','o'] which does not look good in ui, so we show it as a single string
        result = settings_manager.get_setting(action_name)
        return " ".join(result)

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

        if sleep_status():
            shortcuts = {
                "Sleep mode is enabled. Only the following shortcut is available":"",
                (self.get_shortcut('sleep_awake_app'),'      '): 'Sleep/awake',
            }
        elif display_output_mode_status()==False:
            shortcuts = {
            "Output Mode is Disabled. Only the following shortcuts are available":"",
            (self.get_shortcut('save_copied_text_to_file'),'      '): 'Save copied text to the notes pdf file',
            (self.get_shortcut('take_screenshot'),'      '): 'Take screenshot and save it to the notes pdf file',
            (self.get_shortcut('display_output_mode_toggle'), '      '): 'Enable/disable output screen. Use this when you want to take notes. Notes can be screenshots of your input monitor or copied text',
        }
        else:
            shortcuts = {
            'Double click': 'Enable/disable zoom mode',
            (self.get_shortcut('zoom_increase'),'      '): 'Zoom in when in zoom mode',
            (self.get_shortcut('zoom_decrease'),'       '): 'Zoom out when in zoom mode',
            (self.get_shortcut('window_pt_top_left'),'      '): 'Top left corner of the screen section to be visible',
            (self.get_shortcut('window_pt_bottom_right'),'      '): 'Bottom right corner of the screen section to be visible',
            (self.get_shortcut('window_show_everything'),'         '): 'Show entire screen',
            (self.get_shortcut('toggle_region_of_interest_hiding_approach'),'      '): 'Switch between blur and complete blackout',
            (self.get_shortcut('save_copied_text_to_file'),'      '): 'Save copied text to the notes pdf file',
            (self.get_shortcut('take_screenshot'),'      '): 'Take screenshot and save it to the notes pdf file',
            (self.get_shortcut('sleep_awake_app'),'      '): 'Sleep/awake',
            (self.get_shortcut('quit_app'), '      '): 'Quit',
            (self.get_shortcut('pen_mode_toggle'), '      '): 'Enable/disable pen mode. Use option + trackpad to draw',
            (self.get_shortcut('display_output_mode_toggle'), '      '): 'Enable/disable output screen. Use this when you want to take notes. Notes can be screenshots of your input monitor or copied text',
        }

        for keys, description in shortcuts.items():
            keys_display = " ".join(keys) if isinstance(keys, tuple) else keys
            shortcuts_menu.add(rumps.MenuItem(f"{keys_display}   {description}"))

        self.menu.add(shortcuts_menu)

        self.menu.add(rumps.MenuItem("Notes", callback=self.open_notes_folder))

        # Add separator and quit option
        self.menu.add(rumps.separator)
        display_output_mode_label_suffix = ". This will have no effect as no output monitor is selected" if self.output_monitor is None else ""
        display_output_mode_label = f"Disable Output Mode{display_output_mode_label_suffix}" if display_output_mode_status() else f"Enable Output Mode{display_output_mode_label_suffix}"
        self.menu.add(rumps.MenuItem(display_output_mode_label, callback=self.display_output_mode_toggle))

        # Add separator and quit option
        self.menu.add(rumps.separator)

        sleep_awake_menu_label = "Awake" if sleep_status() else "Sleep"      
        self.menu.add(rumps.MenuItem(sleep_awake_menu_label, callback=self.sleep_awake_action))

        #For some reason we get a default Quit button when using rumps to create menu bar app, which gets destroyed when we refresh the menu, so at the point we need to enable our own Quit option
        #The above issue can be solved by setting the self.quit_button to None which is the default quit button provided by rumps
        # if self.input_monitor is not None or self.output_monitor is not None or self.last_sleep_awake_status is not None:
        self.menu.add(rumps.MenuItem("Quit", callback=self.quit_app))

    def display_output_mode_toggle(self,sender):
        append_to_logs("display_output_mode_toggle called from menu bar")
        display_output_mode_toggle()
        self.refresh_menu()
    
    def sleep_awake_action(self,sender):
        append_to_logs("Sleep awake action invoked from menu bar")
        sleep_awake_app()
        #Do not update the last_sleep_awake_status status here, we want the menu bar refresh logic to set that so it the menu bar refreshes
        self.refresh_menu() 

    def select_input_monitor(self, sender):
        append_to_logs("Input monitor is", sender.title)
        self.input_monitor = sender.title.split(" ")[0]  # Get monitor name without (Selected)
        set_input_monitor(self.input_monitor)
        self.refresh_menu()  # Refresh the menu after selection

    def select_output_monitor(self, sender):
        append_to_logs("Output monitor is", sender.title)
        self.output_monitor = sender.title.split(" ")[0]  # Get monitor name without (Selected)
        set_output_monitor(self.output_monitor)
        self.refresh_menu()  # Refresh the menu after selection

    def is_license_expired(self):
        target_date = datetime(2025, 1, 31)
        current_date = datetime.now()
        if current_date > target_date:
            append_to_logs("License Information: License is expired. Please download the newer version of CaptureDisplay")
        else:
            append_to_logs("License Information: Capture Display License is valid")
        return current_date > target_date
    
    def try_load_input_output_monitors_from_settings(self):
        retrieved_input_monitor = settings_manager.get_setting('input_monitor')
        retrieved_output_monitor = settings_manager.get_setting('output_monitor')
        retrieved_display_output_mode = True if settings_manager.get_setting("display_output_mode","enabled")=="enabled" else False
        append_to_logs("Retrieved I/O monitors: ",retrieved_input_monitor,retrieved_output_monitor,retrieved_display_output_mode)

        #Need to validate the monitors, the monitor id can change when user's monitor configuration changes, or user makes manual edit to the app.settings file
        
        # maximum number of displays to return
        max_displays = 100
        # get active display list
        # CGGetActiveDisplayList:
        #     Provides a list of displays that are active (or drawable).
        (err, active_displays, number_of_active_displays) = CGGetActiveDisplayList(
            max_displays, None, None)
        if err:
            append_to_logs("Error retrieving system monitor information, could not validate monitor information in the app.settings")
            return False
        
        append_to_logs("system monitor setup information: ",active_displays,number_of_active_displays)
        # Get the desired monitor's bounds
        # append_to_logs("active displays",number_of_active_displays,"and",active_displays)
        for active_display in active_displays:
            if str(active_display) == str(retrieved_input_monitor):
                self.input_monitor = retrieved_input_monitor
                append_to_logs("Input monitor validated and set to ",self.input_monitor)
            
            if str(active_display) == str(retrieved_output_monitor):
                self.output_monitor = retrieved_output_monitor
                append_to_logs("Output monitor validated and set to ",self.output_monitor)

        #Update the app with I/O monitor information if applicable

        if self.input_monitor is not None:
            set_input_monitor(self.input_monitor)
        if self.output_monitor is not None:
           set_output_monitor(self.output_monitor)
           
        self.refresh_menu()
        
        return self.input_monitor is not None and (self.output_monitor is not None or retrieved_display_output_mode==False)


    def show_monitor_selection_alert(self):

        if self.is_license_expired():
            self.quit_app()

        if self.try_load_input_output_monitors_from_settings() == False:
            append_to_logs("Failed to retrieve I/O monitor from app.settings")
            if self.input_monitor is None or self.output_monitor is None:
                rumps.alert(f"Capture Display\n\nSelect your Input and Output Display from the menu bar. \n\n\n Input Display- The screen where your content is.\n\nOutput Display- The screen you’ll need to share with others during the call.\n\n Capture Display will monitor the input display, apply enhancements, and present the final result on the output display. Simply share the output display screen with others during the call.")

    def refresh_menu(self):
        self.update_menu()  # Call to refresh the menu

    def quit_app(self, _):
        destroy_cv2_windows()
        rumps.quit_application()
    
    def open_notes_folder(self,sender):
        # Get the path to the notes folder in the current directory
        notes_path = os.path.join(
    os.path.expanduser("~"),
    "Library",
    "Application Support",
    'CaptureDisplay'
)
        
        # Create the notes folder if it doesn't exist
        if not os.path.exists(notes_path):
            os.makedirs(notes_path)
        
        # Open the folder in Finder on macOS
        subprocess.run(['open', notes_path])

# should be run on the main thread since it's a UI event loop
def menu_bar_app():
    MonitorSelectorApp().run()
