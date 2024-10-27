import rumps
from Quartz import CGGetActiveDisplayList

class MonitorSelectorApp(rumps.App):
    def __init__(self):
        super(MonitorSelectorApp, self).__init__("Monitor Selector", icon=None)
        self.input_monitor = None
        self.output_monitor = None
        self.monitor_list = self.get_monitors()

        # Create the menu
        self.menu = [self.create_input_monitor_menu(), 
                     self.create_output_monitor_menu(),
                     rumps.separator,
                     "Quit"]

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

    def create_input_monitor_menu(self):
        input_menu = rumps.MenuItem("Input Monitor")
        for monitor in self.monitor_list:
            input_menu.add(rumps.MenuItem(monitor, callback=self.select_input_monitor))
        return input_menu

    def create_output_monitor_menu(self):
        output_menu = rumps.MenuItem("Output Monitor")
        for monitor in self.monitor_list:
            output_menu.add(rumps.MenuItem(monitor, callback=self.select_output_monitor))
        return output_menu

    def select_input_monitor(self, sender):
        print("Input monitor is",sender.title)
        self.input_monitor = sender.title
        rumps.alert(f"Input monitor set to: {self.input_monitor}")

    def select_output_monitor(self, sender):
        print("Output monitor is",sender.title)
        self.output_monitor = sender.title
        rumps.alert(f"Output monitor set to: {self.output_monitor}")

    @rumps.clicked("Quit")
    def quit_app(self, _):
        rumps.quit_application()

if __name__ == "__main__":
    MonitorSelectorApp().run()
