import cv2
import glob
from os.path import basename, splitext
from pipeline import process_image

TEST_IMG_DIR = './test_images/'
OUTPUT_IMG_DIR = './output_images/'
images = glob.glob(TEST_IMG_DIR + 'test*.jpg')

for idx, fname in enumerate(images):
  img = cv2.imread(fname)
  result, warpage, warped, preprocessImage, undistorted = process_image(img, debug=True)
  base_name = basename(fname)

  write_name = OUTPUT_IMG_DIR + splitext(base_name)[0] + '_undistorted.jpg'
  cv2.imwrite(write_name, undistorted)

  write_name = OUTPUT_IMG_DIR + splitext(base_name)[0] + '_binary.jpg'
  cv2.imwrite(write_name, preprocessImage)

  write_name = OUTPUT_IMG_DIR + splitext(base_name)[0] + '_warped.jpg'
  cv2.imwrite(write_name, warped)

  write_name = OUTPUT_IMG_DIR + splitext(base_name)[0] + '_warpape.jpg'
  cv2.imwrite(write_name, warpage)

  write_name = OUTPUT_IMG_DIR + splitext(base_name)[0] + '_result.jpg'
  cv2.imwrite(write_name, result)

