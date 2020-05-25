#Instruction for running 
#After file name enter image path as an argument
#OCRHD.py imagepath
#importing libraries
import cv2
import sys
import pytesseract
import os

#Loading image converting to gray scale and resizing it
path=sys.argv[1]
from PIL import Image
img=Image.open(path)

# Loading pixels and convert to binary
pix = img.load()
for y in range(img.size[1]):
    for x in range(img.size[0]):
        if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
            pix[x, y] = (0, 0,0,255)
        else:
            pix[x, y] = (255, 255, 255, 255)
img.save('temp.png')

#Smoothing ,downsizing and deionizing image
im=cv2.imread('temp.png')
img = cv2.GaussianBlur(im,(5,5),0)
img = cv2.medianBlur(img,5)
img = cv2.bilateralFilter(img,9,75,75)
img = cv2.resize(im, (780,540), interpolation = cv2.INTER_AREA)
#Deionizing image
img=cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
# img=cv2.dilate()
cv2.imwrite('temp.png',img)

#Extracting text
custom_config=r'--oem 3 --psm 6 outputbase digits'
print(img.shape)
text=pytesseract.image_to_string('temp.png',config=custom_config)
print(text)

os.remove('temp.png')
