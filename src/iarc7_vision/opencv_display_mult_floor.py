import cv2
import numpy as np

IMAGE_WIDTH = 138
IMAGE_HEIGHT = 100
MAX_WIDTH = IMAGE_WIDTH * 6

def im_show_m(imgs):
    full_image = np.zeros((0,MAX_WIDTH,3), np.uint8)
    row_image = np.zeros((IMAGE_HEIGHT,0,3), np.uint8)
    current_width = 0
    max_row_height = 0
    for img in imgs:
        shape = img.shape

        # Assume its greyscale
        if len(shape) < 3:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        ih = shape[0]
        iw = shape[1]

        if iw != IMAGE_WIDTH or ih != IMAGE_HEIGHT:
            #raise ValueError('Image does not have the allowed dimensions w: {} h: {}'.format(iw, ih))
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_NEAREST)
            shape = img.shape
            ih = shape[0]
            iw = shape[1]

        if current_width+iw < MAX_WIDTH:
            current_width = current_width + iw
            row_image = np.append(row_image,img, axis=1)
        else:
            current_width = 0
            row_image = np.append(row_image,img, axis=1)
            full_image = np.append(full_image,row_image, axis=0)
            row_image = np.zeros((IMAGE_HEIGHT,0,3), np.uint8)


    rh, rw, rc = row_image.shape
    if (rw < MAX_WIDTH) and (rw != 0):
        filler = np.zeros((IMAGE_HEIGHT, MAX_WIDTH-rw, 3), np.uint8)
        row_image = np.append(row_image,filler, axis=1)
        full_image = np.append(full_image,row_image, axis=0)

    cv2.namedWindow('multishow',cv2.WINDOW_NORMAL)
    cv2.imshow('multishow', full_image)
    fh, fw, fc = full_image.shape
    cv2.resizeWindow('multishow', max(IMAGE_WIDTH, fw), max(IMAGE_HEIGHT, fh))

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    retval, im1 = camera.read()
    retval, im2 = camera.read()

    im1 = cv2.resize(im1, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_LINEAR)
    im2 = cv2.resize(im2, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_LINEAR)

    im_show_m([im1, im2, im1, im1, im2, im2, im1])
    cv2.waitKey(0)
