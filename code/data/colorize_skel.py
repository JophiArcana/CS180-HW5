# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import scipy as sc

# name of the input file
imname = 'cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.util.img_as_float(im)

# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b = im[:height]
g = im[height: 2 * height]
r = im[2 * height: 3 * height]

k = 30

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

sobel_vertical = np.array([
    [0.1875,    0., -0.1875],
    [0.625,     0., -0.625],
    [0.1875,    0., -0.1875]
])
sobel_horizontal = sobel_vertical.T


def sigmoid(arr):
    return 1. / (1. + np.exp(-arr))


def edge_detection(image):
    edge_vertical = sc.signal.convolve2d(image, sobel_vertical, mode='valid')
    edge_horizontal = sc.signal.convolve2d(image, sobel_horizontal, mode='valid')

    # return np.clip(np.linalg.norm(np.dstack([edge_vertical, edge_horizontal]), axis=-1), 0, 1)
    edge = np.linalg.norm(np.dstack([edge_vertical, edge_horizontal]), axis=-1)
    edge_min, edge_max = np.min(edge), np.max(edge)

    return sigmoid(5 * (np.linalg.norm(np.dstack([edge_vertical, edge_horizontal]), axis=-1) - 0.5 * (edge_min + edge_max)) / (edge_max - edge_min))


er = edge_detection(r[k:-k, k:-k])
eg = edge_detection(g[k:-k, k:-k])
eb = edge_detection(b[k:-k, k:-k])


def align(im1, im2, cx, cy, window=20):
    w, h = im1.shape

    max_corr, max_dx, max_dy = 0, None, None
    for dx in range(cx - window, cx + window + 1):
        for dy in range(cy - window, cy + window + 1):
            cropped_im1 = im1[max(0, dx):w + min(0, dx), max(0, dy):h + min(0, dy)]
            cropped_im2 = im2[max(0, -dx):w + min(0, -dx), max(0, -dy):h + min(0, -dy)]

            corr = np.sum(cropped_im1 * cropped_im2)
            if corr > max_corr:
                max_corr, max_dx, max_dy = corr, dx, dy
    print(max_corr, max_dx, max_dy)

    # skio.imshow(max_cropped_im1 * max_cropped_im2)
    # skio.show()

    return max_dx, max_dy


dxr, dyr = 0, 0
dxg, dyg = align(er, eg, 0, 0)
dxb, dyb = align(er, eb, 0, 0)
_, _ = align(eg, eb, 0, 0)

w, h = r.shape
crop_x_min, crop_x_max = max([dxr, dxg, dxb]), w + min([dxr, dxg, dxb])
crop_y_min, crop_y_max = max([dyr, dyg, dyb]), h + min([dyr, dyg, dyb])


print(r.shape)

ar = r[crop_x_min - dxr:crop_x_max - dxr, crop_y_min - dyr:crop_y_max - dyr]
ag = g[crop_x_min - dxg:crop_x_max - dxg, crop_y_min - dyg:crop_y_max - dyg]
ab = b[crop_x_min - dxb:crop_x_max - dxb, crop_y_min - dyb:crop_y_max - dyb]

M = np.vstack([ar.flatten(), ag.flatten(), ab.flatten()]).T
cr, cg, cb = np.linalg.pinv(M) @ np.ones(len(M))
skio.imshow(cr * ar + cg * ag + cb * ab)
skio.show()

im_out = np.floor(256 * np.dstack([ar, ag, ab])).astype('u1')
print(im_out.shape)

# save the image
fname = '../output/' + imname
skio.imsave(fname, im_out)

# display the image
# skio.imshow(im_out)
# skio.show()
