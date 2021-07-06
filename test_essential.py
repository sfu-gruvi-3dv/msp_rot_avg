import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    if len(img1.shape) == 2:  # grayscale input
        print("Grayscale Input (drawLines)")
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    elif len(img1.shape) == 3:  # RGB
        print("Color Input (drawLines)")

    else:
        print(len(img1.shape))

    r, c, ch = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        pt1 = pt1.astype(np.int32)
        pt2 = pt2.astype(np.int32)

        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

    return img1, img2


im_1 = cv2.imread("/Users/corsy/Downloads/AmbiguousData/cup/P1010174.jpg")
im_2 = cv2.imread("/Users/corsy/Downloads/AmbiguousData/cup/P1010171.jpg")

k = [689.70, 0, 533.50, 0, 689.70, 400.0, 0, 0, 1.0]
k = np.asarray(k).reshape(3, 3)

# convert to gray
im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)

# proceed with sparce feature matching
orb = cv2.ORB_create()

kp_1, des_1 = orb.detectAndCompute(im_1, None)
kp_2, des_2 = orb.detectAndCompute(im_2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des_1, des_2)

matches = sorted(matches, key=lambda x: x.distance)

draw_params = dict(matchColor=(20, 20, 20), singlePointColor=(200, 200, 200), matchesMask=None, flags=0)

im_3 = cv2.drawMatches(im_1, kp_1, im_2, kp_2, matches[0:20], None, **draw_params)

# select points to evaluate the fundamental matrix
pts1 = []
pts2 = []
idx = matches[1:20]

for i in idx:
    pts1.append(kp_1[i.queryIdx].pt)
    pts2.append(kp_2[i.trainIdx].pt)

pts1 = np.array(pts1)
pts2 = np.array(pts2)

# creating homegeneous coordenate
pones = np.ones((1, len(pts1))).T

pth_1 = np.hstack((pts1, pones))
pth_2 = np.hstack((pts2, pones))

k = np.array(k)
ki = np.linalg.inv(k)
# normalized the points
pthn_1 = []
pthn_2 = []

for i in range(0, len(pts1)):
    pthn_1.append((np.mat(ki) * np.mat(pth_1[i]).T).transpose())
    pthn_2.append((np.mat(ki) * np.mat(pth_2[i]).T).transpose())

ptn1 = []
ptn2 = []
for i in range(0, len(pts1)):
    ptn1.append([pthn_1[i][0, 0], pthn_1[i][0, 1]])
    ptn2.append([pthn_2[i][0, 0], pthn_2[i][0, 1]])

ptn1 = np.array(ptn1)
ptn2 = np.array(ptn2)
# evaluate the essential Matrix (using the original points, not the normilized ones)
E, mask0 = cv2.findEssentialMat(pts1, pts2, k, cv2.RANSAC)
# evaluate the fundamental matrix (using the normilized points)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)

E = np.mat(E)
F = np.mat(F)
# print(E)

_, R, t, _ = cv2.recoverPose(E, pts1, pts2, k)
print(R, t)

#
# selecting only inlier points
ptn1 = ptn1[mask.ravel() == 1]
ptn2 = ptn2[mask.ravel() == 1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(im_1, im_2, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(im_2, im_1, lines2, pts2, pts1)

plt.subplot(131)
plt.imshow(img5)
plt.subplot(132)
plt.imshow(img3)
plt.subplot(133)
plt.imshow(im_3)
#
plt.show()