import os
import importlib
utils = importlib.import_module("utils")
append_to_logs = utils.append_to_logs

class SettingsManager:
    def __init__(self):

        # Usage example
        file_path = os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "CaptureDisplayX77",
            "app.settings"
        )

        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        append_to_logs("Settings manager initialized")
        
    def _load_settings(self):
        settings = {}
        try:
            with open(self.file_path, "r") as file:
                for line in file:
                    # Ignore empty lines or lines without an '='
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        if value.startswith("[") and value.endswith("]"):
                            # Parse as list of strings
                            value = value[1:-1].split(",")  # remove brackets and split by commas
                            value = [v.strip() for v in value]  # strip whitespace around each item
                        settings[key.strip()] = value
        except FileNotFoundError:
            # If the settings file does not exist, return an empty dictionary
            pass
        return settings

    def _write_settings(self, settings):
        with open(self.file_path, "w") as file:
            for key, value in settings.items():
                if isinstance(value, list):
                    value = "[" + ", ".join(value) + "]"
                file.write(f"{key}={value}\n")

    def get_setting(self, key, default=None, save_to_file_if_not_exists=True):
        append_to_logs(f"Settings.get_setting {key} and default is {default}")
        settings = self._load_settings()
        result_with_default = settings.get(key,default)
        result_with_none_default = settings.get(key, None)
        if save_to_file_if_not_exists and result_with_none_default is None and default is not None:
            #this mean the key is not present in file as it result with a none, and the default supplied to us is not none, so we will save that to the config file
            append_to_logs("Tried to get a setting which was not present, saving the default value in the file, so user can override later on")
            settings[key] = default
            self._write_settings(settings)

        return result_with_default


    def set_setting(self, key, value):
        append_to_logs(f"Settings.set_setting {key} with value {value}")
        # Ensure the settings file exists by opening in append mode
        open(self.file_path, 'a').close()

        settings = self._load_settings()
        settings[key] = value
        self._write_settings(settings)


# # Usage example
# settings_file_path = os.path.join(
#     os.path.expanduser("~"),
#     "Library",
#     "Application Support",
#     "CaptureDisplayX77",
#     "app.settings"
# )

# settings_manager = SettingsManager(settings_file_path)
# append_to_logs("Settings manager initialized")

# Get a setting with a default value if the key doesn't exist
# result = settings_manager.get_setting("some_key", default="default_value")
# append_to_logs("Retrieved value:", result)

# # Set or update a setting
# settings_manager.set_setting("some_key", ["value1", "value2", "value3"])
# settings_manager.set_setting("some_key", ["value1", "value2", "value4"])
# settings_manager.set_setting("another_key", "single_value")
