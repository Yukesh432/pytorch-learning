import os
import cv2
import selectivesearch
import matplotlib.pyplot as plt
import numpy as np


image_path= 'path'
# images= os.listdir(image_path)


# img= cv2.imread(image_path)
# print(img.shape[1])
# plt.imshow(img)
# plt.show()



def selective_search_demo(image_path, num_show_rects=100):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return

    # Create a copy of the image for drawing boxes
    image_copy = image.copy()

    # Initialize OpenCV's selective search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()  # You can use switchToSelectiveSearchQuality() for better but slower results

    # Run selective search
    rects = ss.process()

    # Number of rectangles to show (limited to avoid cluttering)
    num_show_rects = min(num_show_rects, len(rects))

    # Draw rectangles on the image copy
    for i, rect in enumerate(rects[:num_show_rects]):
        x, y, w, h = rect
        color = np.random.randint(0, 255, size=3).tolist()  # Random color for each box
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, 2)

    # Display the image with bounding boxes
    cv2.imshow("Selective Search", image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__== '__main__':

    selective_search_demo(image_path)