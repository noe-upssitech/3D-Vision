import cv2 as cv

def displayCam():           
    capture = cv.VideoCapture(2)
    cv.namedWindow('Camera', cv.WINDOW_NORMAL)

    while(True):
        # Capture frame-by-frame
        _, frame = capture.read()

        # Display the resulting frame
        cv.imshow('Camera', frame)    
        
        # Wait 100 ms
        key = cv.waitKey(100)

        # Quit with escape or q
        if key == ord('\x1b') or key == ord('q'):
            break
        
    capture.release()
    cv.destroyAllWindows()

def displayImage(imagePath=''):
    cv.namedWindow('Image', cv.WINDOW_NORMAL)

    frame = cv.imread(imagePath)
    cv.imshow('Image', frame)
    cv.waitKey(0)

    cv.destroyAllWindows()

displayCam()

