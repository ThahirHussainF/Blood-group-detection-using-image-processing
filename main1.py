import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpg')
GrayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('in', GrayFrame)

FrameThresh = cv2.threshold(
    GrayFrame, 125, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow()
bw = cv2.bitwise_not(FrameThresh)

kernel = np.ones((10, 10), np.uint8)
FrameThresh = cv2.erode(FrameThresh, kernel, iterations=5)

FrameThresh = cv2.bitwise_not(FrameThresh)

cv2.imshow('bw', FrameThresh)

sm = np.sum(FrameThresh, axis=0)
# plt.plot(sm)
# plt.show()
print(len(sm))

print(sm[1])
# f = np.zeros(1, len(sm))
f = np.zeros(len(sm))
flag = 0
flag1 = 0
for ii in range(len(sm) - 1):
    i = ii + 1
    if (sm[i] == 0 and flag == 0 and flag1 == 0):
        f[i] = 0
        flag = 1
    elif (sm[i] == 0 and flag == 1 and flag1 == 0):
        f[i] = 0
        flag = 1
    elif (sm[i] > 0 and flag1 == 0):
        if (flag == 1 and sm[i] > 0):
            f[i] = 1
            flag = 0
            flag1 = 1

    elif (flag1 == 1 and flag == 0 and sm[i] == 0):
        f[i] = 2
        flag = 1
        flag1 = 0

# plt.plot(f)
# plt.show()
# p1 = f.index(1)
# p2 = f.index(2)
p1 = np.where(f == 1)
p2 = np.where(f == 2)

p2 = (np.asarray(p2))[0]
p1 = (np.asarray(p1))[0]

print(p1[0])
print(p2[0])

crop_img1 = bw[:, p1[0]:p2[0]]
crop_img2 = bw[:, p1[1]:p2[1]]
crop_img3 = bw[:, p1[2]:p2[2]]

cv2.imshow("cropped1", crop_img1)
cv2.imshow("cropped2", crop_img2)
cv2.imshow("cropped3", crop_img3)

crop_img1 = cv2.erode(crop_img1, None, iterations=2)
crop_img2 = cv2.erode(crop_img2, None, iterations=2)
crop_img3 = cv2.erode(crop_img3, None, iterations=2)

(contours, hierarchy) = cv2.findContours(crop_img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
count1 = len(contours)
(contours, hierarchy) = cv2.findContours(crop_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
count2 = len(contours)
(contours, hierarchy) = cv2.findContours(crop_img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
count3 = len(contours)

print("c1: " + str(count1))
print("c2: " + str(count2))
print("c3: " + str(count3))

crop_img1.dtype = 'uint8'
a1 = np.sum(crop_img1 == 255)
crop_img2.dtype = 'uint8'
a2 = np.sum(crop_img2 == 255)
crop_img3.dtype = 'uint8'
a3 = np.sum(crop_img3 == 255)

print("a1: " + str(a1))
print("a2: " + str(a2))
print("a3: " + str(a3))

if (a1 < 10000 and count1 > 10):
    antiA = 1
else:
    antiA = 0

if (a2 < 10000 and count2 > 10):
    antiB = 1
else:
    antiB = 0

if (a3 < 10000 and count3 > 10):
    antiC = 1
else:
    antiC = 0

if (antiA == 0 and antiB == 0 and antiC == 0):
    print('O-NEGATIVE');
elif (antiA == 0 and antiB == 0 and antiC == 1):
    print('O-POSITIVE');
elif (antiA == 1 and antiB == 0 and antiC == 1):
    print('A-POSITIVE');
elif (antiA == 1 and antiB == 0 and antiC == 0):
    print('A-NEGATIVE');
elif (antiA == 0 and antiB == 1 and antiC == 0):
    print('B-NEGATIVE');
elif (antiA == 0 and antiB == 1 and antiC == 1):
    print('B-POSITIVE');
elif (antiA == 1 and antiB == 1 and antiC == 0):
    print('AB-NEGATIVE');
elif (antiA == 1 and antiB == 1 and antiC == 1):
    print('AB-POSITIVE');

cv2.waitKey(0)
