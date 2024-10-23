import cv2 as cv
from time import time
from windowcapture import WindowCapture

# initialise the WindowCapture class
wincap = WindowCapture('1')

loop_time = time()

while(True):
    # get an updated image of the window you want
    screenshot = wincap.get_image_from_window()

    # show that image
    cv.imshow('Computer Vision', screenshot)

    print('FPS {:.2f}'.format(round(1 / (time() - loop_time), 2)))
    loop_time = time()

    # hold 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')