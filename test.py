from Quartz import (
    CGGetActiveDisplayList,
    CGDisplayVendorNumber,CGDisplayModelNumber,CGDisplaySerialNumber
)

def get_connected_monitors():
    """Get a list of connected monitor names."""
    max_displays = 10  # Maximum number of displays
    (err, active_displays, number_of_active_displays) = CGGetActiveDisplayList(max_displays, None, None)
    
    if err:
        print("Error retrieving monitor information.")
        return []

    monitor_names = []
    for display_id in active_displays:
        # Get the display name
        display_name = CGDisplayVendorNumber(display_id)
        print(display_id)
        print(CGDisplayModelNumber(display_id))
        print(CGDisplaySerialNumber(display_id))
        monitor_names.append(display_name if display_name else f"Display {display_id}")

    return monitor_names

if __name__ == "__main__":
    monitors = get_connected_monitors()
    print("Connected Monitors:")
    # for monitor in monitors:
    #     print(f"- {monitor}")
