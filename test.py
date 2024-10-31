import tkinter as tk

# Sample coordinates and their corresponding numbers
coordinates = [
    (50, 50, "1"),
    (150, 50, "2"),
    (50, 150, "3"),
    (150, 150, "4"),
    (100, 100, "5"),
]

def create_number_box(x, y, number):
    # Create a new window for each number
    box_window = tk.Toplevel()
    box_window.geometry("100x100")  # Size of the window
    box_window.overrideredirect(True)  # Remove title bar
    box_window.geometry(f"+{x}+{y}")  # Position at (x, y)

    # Create a box at the specified coordinates
    box = tk.Frame(box_window, width=80, height=80, bg="lightblue", bd=2, relief="groove")
    box.pack_propagate(False)  # Prevent the frame from resizing
    box.pack()

    # Create a label with the number
    label = tk.Label(box, text=number, font=("Arial", 24))
    label.pack(expand=True)

# Create the main window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Create separate number boxes
for x, y, number in coordinates:
    create_number_box(x, y, number)

# Start the Tkinter event loop
root.mainloop()
 