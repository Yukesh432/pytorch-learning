import cv2

"""
opencv by default open the image in BGR(blue, green, red) mode. We can define how to 
open with mode parameter

-1: cv2.IMREAD_COLOR: loads the color image. It's default mode
0:  cv2.IMREAD_GRAYSCALE: loads the image in grayscale mode
1:  cv2.IMREAD_UNCHANGED: loads the image unchanged

"""
img= cv2.imread('../data/coffee.png', -1)


"""
For resizing we can either provide the resoolution or we can use fx, and fy
to resize accoriding to the fraction of original height and width of image
"""
img= cv2.resize(img, (200, 200))
img= cv2.resize(img, (0, 0), fx= 2, fy=2)

# Rotate the image
img= cv2.rotate(img, cv2.ROTATE_180)


# Showing the image
cv2.imshow('coffee image', img)


# For saving the image
cv2.imwrite("new_saved_image.jpg", img)

# Wait for given amount of time (in sec) and destroy windows
cv2.waitKey(0)
cv2.destroyAllWindows()