import numpy as np
import cv2

I = cv2.imread('0012.jpg', 1)
src = I.copy()
Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
Ig = cv2.medianBlur(Ig, 5)
rows = Ig.shape[0]
# find circles in the image, we used hough transforms
circles = cv2.HoughCircles(Ig, cv2.HOUGH_GRADIENT, dp=1, minDist=rows / 30,
                           param1=120, param2=20,
                           minRadius=1, maxRadius=50)
# save the results into file

file = open('coordinates.txt', 'w')
if circles is not None:
    # sort circles ascending according to the radius
    circles = sorted(circles[0, :, :].tolist(), key=lambda x: x[2])  # x is [cx,cy,radius]
    circles = np.uint16(np.around(circles))
    for i in circles:
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(src, center, radius, (0, 0, 0), 3)
        file.write(f"X:{i[0]}, Y:{i[1]}\n")
file.close()


cv2.waitKey(0)
circles = circles[-2:]  # get the two bigger circles in our results
# one will be the dice and the other is the center

num_dot = 0
dice_index = -1
# for each circle we do the following:
# 	1- crop the image around the circle
# 	2- do hough transform
# 	3- count the dots
# dice will be the circle with most dots

for i, circle in enumerate(circles):
    r = int(1.5 * circle[-1])
    c = circle[:-1]
    crop_img = I[c[1] - r:c[1] + r, c[0] - r:c[0] + r, :]  # crop rectangle around the circle
    crop_img2 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # apply Hough circle with different params to detect small circles on dice
    rows = crop_img2.shape[0]
    dots = cv2.HoughCircles(crop_img2, cv2.HOUGH_GRADIENT, dp=1, minDist=rows / 5,
                            param1=120, param2=15,
                            minRadius=5, maxRadius=9)
    # dice will be the circle with most dots
    if dots is not None:
        if len(dots[0, :]) > num_dot:
            print(len(dots[0, :]))
            num_dot = len(dots[0, :])
            dice_index = i
# dice_index is the corresponding index for dice
x, y, r = circles[dice_index]
cv2.circle(src, (x, y), r, (0, 0, 0), 3)
print('Reading dice:\n value: ', num_dot)
src = cv2.resize(src, (src.shape[1] // 2, src.shape[0] // 2))
cv2.imshow('result',src)
cv2.waitKey(0)
