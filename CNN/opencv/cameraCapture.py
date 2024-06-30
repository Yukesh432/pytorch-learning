import numpy as np
import cv2

cap= cv2.VideoCapture(0)  # 0 represent default webcam

while True:
    ret, frame= cap.read()   # Here frame is the numpy array of the image

    width= int(cap.get(3))  # 3 is for width 
    height= int(cap.get(4))   # 4 is for height
    image= np.zeros(frame.shape, np.uint8)
    smaller_frame= cv2.resize(frame, (0, 0), fx= 0.5, fy=0.5)
    
    image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, :width//2] = smaller_frame
    image[:height//2, width//2:] = smaller_frame
    image[height//2:, width//2:] = smaller_frame
   


    cv2.imshow('frame dd', image)

    if cv2.waitKey(1) == ord('q'):  #1 milisecond
        break

cap.release()
cv2.destroyAllWindows()