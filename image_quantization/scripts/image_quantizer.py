import numpy as np
import matplotlib.pyplot as plt
import numba


class ImageQuantization(object):
    def __init__(self):
        pass

    def uniform_quantization(self, img, N):
        return uniform_quantization(img, N)

    def logarithmic_quantization(self, img, N):
        print(f"logarithmic quantization function is incomplete!")
        return logarithmic_quantization(img, N)
    
    def blockwise_quantization(self, img, N):
        return blockwise_quantization(img, N)

    def error_diffusion_quantization(self, img, N, h=None, method="jjn"):
        if method == "visual model":
            return halftone_visual_model(img, N)
        else:
            return error_diffusion_quantization(img, h, method, N)


# uniform quantizer (Nbit)
def uniform_quantization(img, N):
    level = pow(2, N)
    d = 255 / (level-1)
    quantized_img = np.round(img / d) * d
    return quantized_img.astype(np.uint8)

# logarithmic quantizer (Nbit)
def logarithmic_quantization(img, N):
    level = pow(2, N)

    max_val = np.max(img)
    min_val = np.min(img)
    # Compute the range of the image
    image_range = max_val - min_val
    # Calculate the logarithmic base
    base = np.exp(np.log(image_range) / level)
    # Calculate the quantization levels
    quantization_levels = [min_val + (base ** i) for i in range(levels)]
    
    # Quantize the image
    quantized_image = np.zeros_like(img)
    for i in range(level - 1):
        mask = np.logical_and(img >= quantization_levels[i], img <= quantization_levels[i + 1])
        quantized_image = np.where(mask, quantization_levels[i], quantized_image)
    
    # Set the maximum value for the last interval
    mask_last = img >= quantization_levels[level - 1]
    quantized_image = np.where(mask_last, quantization_levels[level - 1], quantized_image)
    
    return quantized_image.astype(np.uint8)

# Block-wise quantization
@numba.njit
def blockwise_quantization(img, N=4):
    # Generally, N = {4, 8, 16}
    height, width = img.shape
    quantized_img = np.copy(img)

    for i in range(0, height, N):
        for j in range(0, width, N):
            i_end, j_end = i + N, j + N
            if i + N > height: i_end = height
            if j + N > width:  j_end = width
                
            block = img[i:i_end, j:j_end]
            max_value = np.max(block)
            min_value = np.min(block)
            thresh = (max_value + min_value) / 2

            for x in range(N):
                for y in range(N):
                    if block[x, y] >= thresh:
                        quantized_img[i+x, j+y] = thresh
                    else:
                        quantized_img[i+x, j+y] = min_value

    return quantized_img.astype(np.uint8)


# Visual Model 用パラメータ
A0 = 0.21
A1 = 0.32
A2 = 0.31
A3 = 0.13
B  = 0.04
C  = 1.00

AQ0 = A0
AQ1 = A1
AQ2 = A2
AQ3 = A3
BQ  = B
CQ0 = -AQ0 / B
CQ1 = -AQ1 / B
CQ2 = -AQ2 / B
CQ3 = -AQ3 / B

# Halftone with ODQ (Nbit)
@numba.njit
def halftone_visual_model(img, N):
    level = pow(2, N)
    d = 255 / (level-1)

    xi = np.zeros((img.shape[1]+2, img.shape[0]+1))
    vd = np.zeros((img.shape[1], img.shape[0]), dtype=np.int64)
    ud = img.T

    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            vd[i,j] = np.round(((np.array([[CQ0,CQ1,CQ2,CQ3,1]]) @ np.array([[xi[i,j],xi[i,j+1],xi[i+1,j], xi[i+2,j],ud[i,j]]]).T)[0, 0] / d)) * d
            xi[i+1, j+1] = (np.array([[AQ0,AQ1,AQ2,AQ3,BQ]]) @ np.array([[xi[i,j],xi[i,j+1],xi[i+1,j],xi[i+2,j],(vd[i,j] - ud[i,j])]]).T)[0,0]

    return vd.T.astype(np.uint8)


# halftone with error diffusion
@numba.njit
def error_diffusion_quantization(img, h=None, method="custom", N=1):

    # Processing
    if method == "jjn":
        ## Jarvis, Judice & Ninke
        h = np.array([[0, 0, 0, 7, 5],
                      [3, 5, 7, 5, 3],
                      [1, 3, 5, 3, 1]]) / 48
        FSTEPJ = 2
        BSTEPJ = -2
        FSTEPI = 2 
    if method == "fs":
        ## floyd-Steinburg
        h = np.array([[0, 0, 7],
                      [3, 5, 1]]) /16
        FSTEPJ = 1
        BSTEPJ = -1
        FSTEPI = 1
    if method == "custom":
        if h.shape[0]*h.shape[1] == 6:
            FSTEPJ = 1
            BSTEPJ = -1
            FSTEPI = 1
        if h.shape[0]*h.shape[1] == 15:
            FSTEPJ = 2
            BSTEPJ = -2
            FSTEPI = 2
    
    level = pow(2, N)
    d = 255 / (level-1)

    tmp1 = np.concatenate((np.ones((img.shape[0], -BSTEPJ)), img, np.ones((img.shape[0], FSTEPJ))), axis=1)
    tmp2 = np.concatenate((np.ones((FSTEPI, -BSTEPJ)), np.ones((FSTEPI, img.shape[1])), np.ones((FSTEPI, FSTEPJ))), axis=1)
    tmp = np.concatenate((tmp1, tmp2), axis=0)

    newImage = np.zeros((img.shape[0]+FSTEPI, img.shape[1]+FSTEPJ - BSTEPJ), dtype=np.int64)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]+FSTEPJ):
            if (i > img.shape[0] or j > img.shape[1] - BSTEPJ or j <= -BSTEPJ):
                newImage[i,j] = np.round(newImage[i,j] / d) * d
            else:
                tmp_ceil  = np.ceil(tmp[i,j] / d) * d
                tmp_floor = np.floor(tmp[i,j] / d) * d 

                err1 = tmp[i,j] - tmp_floor
                err2 = tmp[i,j] - tmp_ceil

                if (err1 * err1 < err2 * err2):
                    newImage[i,j] = tmp_floor
                    err3 = err1
                else:
                    newImage[i,j] = tmp_ceil
                    err3 = err2

                for k in range(1, FSTEPI + 2):
                    for l in range(BSTEPJ, FSTEPJ+1):
                        tmp[i+k-1, j+l] += err3 * h[k-1, -1*BSTEPJ+l]

    return newImage[:-FSTEPI, FSTEPJ:BSTEPJ].astype(np.uint8)
    

if __name__ == "__main__":

    # Time Test
    img_path = "/Users/tsy/Vscode/Projects/halftone_slam/datasets/samples/imgs/freiburg.png"
    