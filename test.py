import os
import torch
import cv2


# print(torch.cuda.is_available())
# print(os.path.isdir('../../data/pascal_voc/'))

url = './test_img/2.jpg'
image = cv2.imread(url)
print(image.shape)
print(type(image))

# Display the image
#cv2.imshow("Image", image)

# Wait for the user to press a key
cv2.waitKey(0)

# Close all windows
# cv2.destroyAllWindows()