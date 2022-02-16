# Python program to explain cv2.imread() method
  
# importing cv2 
import cv2
  
# path
path = r'D:\NoDefects-0_jpg.rf.8df76f71bc232b121de110483cc6e24b.jpg'
  
# Using cv2.imread() method
img = cv2.imread(path)
orb = cv2.ORB_create()
  
# Displaying the image
cv2.imshow('image', img)

#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 