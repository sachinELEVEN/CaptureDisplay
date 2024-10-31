import threading
import importlib

keyboard_listener = importlib.import_module("keyboard-listener")
listen_keyboard_events = keyboard_listener.listen_keyboard_events
fast_screen_recording = importlib.import_module("fast-screen-recording")
screen_rec_and_mouse_click_listener = fast_screen_recording.screen_rec_and_mouse_click_listener
menu_bar_app = importlib.import_module("menu_bar_app")
menu_bar_app = menu_bar_app.menu_bar_app

def start_application():
    print("Starting CaptureDisplayX")

    #Listen keyboard events on a different thread
    print("Starting keyboard_listener_thread thread")
    keyboard_listener_thread = threading.Thread(target=listen_keyboard_events)
    keyboard_listener_thread.daemon = True
    keyboard_listener_thread.start()
    

    #(THIS NOW RUNS FROM MENU BAR APP'S EVENT LOOP)Screen recording and mouse click listener thread needs to be on the main thread because i think it uses screen recording ui apis
    # print("Starting screen_rec_and_mouse_click_listener_main_thread on main thread")
    # screen_rec_and_mouse_click_listener_main_thread = screen_rec_and_mouse_click_listener
    # screen_rec_and_mouse_click_listener_main_thread()
    

    print("starting menu_bar_app on main thread")
    menu_bar_app()

    keyboard_listener_thread.join()
    print("Terminating CaptureDisplayX")

if __name__ == "__main__":
    start_application()

    

    