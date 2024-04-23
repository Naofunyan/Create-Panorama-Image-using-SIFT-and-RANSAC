import numpy as np
import cv2
from matplotlib import pyplot as plt

#Load images and convert to grayscale
img1 = cv2.imread('test3.jpg') #RightPosition
img2 = cv2.imread('test2.jpg') #MiddlePosition
img3 = cv2.imread('test1.jpg') #LeftPosition
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

#Import SIFT
sift = cv2.SIFT_create()

#Find keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
keypoints3, descriptors3 = sift.detectAndCompute(gray3, None)

#Match keypoints
bf = cv2.BFMatcher()
matches12 = bf.knnMatch(descriptors1, descriptors2, k=2)
matches23 = bf.knnMatch(descriptors2, descriptors3, k=2)

#Test to find good matches
good_matches12 = []
for m, n in matches12:
    if m.distance < 0.75 * n.distance:
        good_matches12.append(m)
        good_matches23 = []

for m, n in matches23:
    if m.distance < 0.75 * n.distance:
        good_matches23.append(m)

#Extract matched keypoints
src_points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches12]).reshape(-1, 1, 2)
dst_points1 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches12]).reshape(-1, 1, 2)
src_points2 = np.float32([keypoints2[m.queryIdx].pt for m in good_matches23]).reshape(-1, 1, 2)
dst_points2 = np.float32([keypoints3[m.trainIdx].pt for m in good_matches23]).reshape(-1, 1, 2)

#Compute homography
homography12, _ = cv2.findHomography(src_points1, dst_points1, cv2.RANSAC, 5.0)
homography23, _ = cv2.findHomography(src_points2, dst_points2, cv2.RANSAC, 5.0)

#Warp images
img_warped1_2 = cv2.warpPerspective(img1, homography12, (img2.shape[1] + img1.shape[1], img2.shape[0]))
img_warped1_2[0:img2.shape[0], 0:img2.shape[1]] = img2
img_warped2_3 = cv2.warpPerspective(img_warped1_2, homography23, (img3.shape[1] + img_warped1_2.shape[1], img3.shape[0]))
img_warped2_3[0:img3.shape[0], 0:img3.shape[1]] = img3

#Stitch images together
panorama = img_warped2_3

#Export keypoints
keypoints_array = [np.float32([kp.pt]) for kp in keypoints1], [np.float32([kp.pt]) for kp in keypoints2], [np.float32([kp.pt]) for kp in keypoints3]

#Export panorama
plt.figure(figsize=(15, 5))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig('Export.png', dpi=300, bbox_inches='tight')
plt.show()
