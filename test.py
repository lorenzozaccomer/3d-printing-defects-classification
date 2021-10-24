# Python program to explain cv2.imread() method
  
# importing cv2 
import cv2
  
# path
path = r'D:\geeks14.png'
  
# Using cv2.imread() method
img = cv2.imread(path)
  
# Displaying the image
cv2.imshow('image', img)

#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 