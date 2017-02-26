import numpy as np
import cv2
import glob
import pickle

CAM_CAL_DIR = 'camera_cal/'

# obj points, (0,0,0), (1,0,0) ..., (7,5,0), (8,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

images = glob.glob(CAM_CAL_DIR + './calibration*.jpg')

for idx, fname in enumerate(images):
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

  print ('working on ', fname, 'idx: ', idx, 'shape: ', img.shape, 'found: ', ret)
  if ret:
    objpoints.append(objp)
    imgpoints.append(corners)

    cv2.drawChessboardCorners(img, (9,6), corners, ret)
    write_name = CAM_CAL_DIR + 'corners_found' + str(idx) + '.jpg'
    cv2.imwrite(write_name, img)

img = cv2.imread(CAM_CAL_DIR + './calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./calibration_pickle.p", "wb"))
