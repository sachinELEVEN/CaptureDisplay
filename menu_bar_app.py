import rumps
from Quartz import CGGetActiveDisplayList

class MonitorSelectorApp(rumps.App):
    def __init__(self):
        super(MonitorSelectorApp, self).__init__("Capture Display", icon=None)
        self.input_monitor = None
        self.output_monitor = None
        self.monitor_list = self.get_monitors()
        self.update_menu()  # Initialize the menu

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
        
        self.menu.add(input_menu)  # Use add() instead of append

        # Add Output Monitor header
        output_monitor_label = f"Output Monitor ({self.output_monitor})" if self.output_monitor else "Output Monitor"
        output_menu = rumps.MenuItem(output_monitor_label)
        
        for monitor in self.monitor_list:
            selected = " (Selected)" if monitor == self.output_monitor else ""
            output_menu.add(rumps.MenuItem(f"{monitor}{selected}", callback=self.select_output_monitor))
        
        self.menu.add(output_menu)  # Use add() instead of append

        # Add separator and quit option
        self.menu.add(rumps.separator)

        #For some reason we get a default Quit button when using rumps to create menu bar app, which gets destroyed when we refresh the menu, so at the point we need to enable our own Quit option
        if self.input_monitor is not None or self.output_monitor is not None:
            self.menu.add(rumps.MenuItem("Quit", callback=self.quit_app))

    def select_input_monitor(self, sender):
        print("Input monitor is", sender.title)
        self.input_monitor = sender.title.split(" ")[0]  # Get monitor name without (Selected)
        self.refresh_menu()  # Refresh the menu after selection
        rumps.alert(f"Input monitor set to: {self.input_monitor}")
        

    def select_output_monitor(self, sender):
        print("Output monitor is", sender.title)
        self.output_monitor = sender.title.split(" ")[0]  # Get monitor name without (Selected)
        self.refresh_menu()  # Refresh the menu after selection
        rumps.alert(f"Output monitor set to: {self.output_monitor}")
        

    def refresh_menu(self):
        self.update_menu()  # Call to refresh the menu

    # @rumps.clicked("Quit")
    def quit_app(self, _):
        rumps.quit_application()

if __name__ == "__main__":
    MonitorSelectorApp().run()
