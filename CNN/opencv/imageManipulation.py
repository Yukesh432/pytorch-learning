import cv2
import random
img= cv2.imread("../data/coffee.png")

# print(img)
print(type(img))
print(img.shape)

"""
An image with 4*3 pixels will have numpy array like this:
These are Blue, green, red in opencv

[
[[1,1,1], [2,3,4], [4,5,6]]
[[0,0,0], [1,2,3], [2,3,4]]
[[2,1,1], [2,3,4], [5,4,3]]
[[5,6,7], [7,8,9], [9,8,7]]
]
"""

# print(img[0:100])
cv2.imshow('gfgf',img[0:100])
cv2.waitKey(0)
cv2.destroyAllWindows()

## Randomly corrupting certain part of the image
for i in range(100):
    for j in range(img.shape[0]):
        img[i][j]= [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


cv2.imshow('Corrupted image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Copy part of the image . here img[row, column]

piece= img[60:80, 60:80]
img[10:30, 30:50]= piece
cv2.imshow('Corrupted image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()