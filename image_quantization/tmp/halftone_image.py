import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numba
from tqdm import tqdm



class Halftone(object):
    def __init__(self):
        pass

    def uniform_quantizer_1bit(self, img):
        return uniform_quantizer_1bit(img)
    
    def uniform_quantizer_Nbit(self, img, N):
        return uniform_quantizer_Nbit(img, N)
    
    def block_wise_quantizer(self, img, N):
        return block_wise_quantizer(img, N)
    
    def logarithmic_quantizer_Nbit(self, img, N):
        return logarithmic_quantizer_Nbit(img, N)
    
    def halftone_visual_model_1bit(self, img):
        return halftone_visual_model_1bit(img) 
    
    def halftone_visual_model_Nbit(self, img, N):
        return halftone_visual_model_Nbit(img, N)

    def halftone_error_diffusion_1bit(self, img, h, method="jjn"):
        return halftone_error_diffusion_1bit(img, h, method)


# uniform quantizer (Nbit)
def uniform_quantizer_1bit(img):
    img[img >= 128] = 255
    img[img < 128]  = 0
    return img.astype(np.uint8)

# uniform quantizer (Nbit)
def uniform_quantizer_Nbit(img, N):
    level = pow(2, N)
    d = 255 / (level-1)
    quantized_img = np.round(img / d) * d
    return quantized_img.astype(np.uint8)

# logarithmic quantizer (Nbit)
def logarithmic_quantizer_Nbit(img, N):
    level = pow(2, N)
    scale_factor = (level - 1) / np.log2(np.max(img)+1)
    quantized_img = np.round(scale_factor*np.log2(img+1))
    
    return quantized_img.astype(np.uint8)

# Block-wise quantization
@numba.njit
def block_wise_quantizer(img, N=4):
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

# Halftone with ODQ (1bit)
@numba.njit
def halftone_visual_model_1bit(img):
    threshold = 128

    xi = np.zeros((img.shape[1]+2, img.shape[0]+1))
    vd = np.zeros((img.shape[1], img.shape[0]), dtype=np.int64)
    ud = img.T

    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            tmp = np.array([[CQ0,CQ1,CQ2,CQ3,1]]) @ np.array([[xi[i,j],xi[i,j+1],xi[i+1,j], xi[i+2,j],ud[i,j]]]).T
            if tmp > threshold:
                vd[i,j] = 255
            A1 = np.array([[AQ0,AQ1,AQ2,AQ3,BQ]])
            A2 = np.array([[xi[i,j],xi[i,j+1],xi[i+1,j],xi[i+2,j],(vd[i,j] - ud[i,j])]]).T
            xi[i+1, j+1] = (A1 @ A2)[0,0]

    return vd.T.astype(np.uint8)

# Halftone with ODQ (Nbit)
@numba.njit
def halftone_visual_model_Nbit(img, N):
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
def halftone_error_diffusion_1bit(img, h=None, method=None):

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
    

    tmp1 = np.concatenate((np.ones((img.shape[0], -BSTEPJ)), img, np.ones((img.shape[0], FSTEPJ))), axis=1)
    tmp2 = np.concatenate((np.ones((FSTEPI, -BSTEPJ)), np.ones((FSTEPI, img.shape[1])), np.ones((FSTEPI, FSTEPJ))), axis=1)
    tmp = np.concatenate((tmp1, tmp2), axis=0)

    newImage = np.zeros((img.shape[0]+FSTEPI, img.shape[1]+FSTEPJ - BSTEPJ), dtype=np.int64)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]+FSTEPJ):
            if (i > img.shape[0] or j > img.shape[1] - BSTEPJ or j <= -BSTEPJ):
                    newImage[i,j] = 255 if tmp[i,j] > 128 else 0
            else:
                err1 = tmp[i,j] - 0
                err2 = tmp[i,j] - 255

                if (err1 * err1 < err2 * err2):
                    newImage[i,j] = 0
                    err3 = err1
                else:
                    newImage[i,j] = 255
                    err3 = err2

                for k in range(1, FSTEPI + 2):
                    for l in range(BSTEPJ, FSTEPJ+1):
                        tmp[i+k-1, j+l] += err3 * h[k-1, -1*BSTEPJ+l]

    return newImage[:-FSTEPI, :-(FSTEPJ-BSTEPJ)]


def test_time(img_path, N=1):
    
    h = Halftone()
    f = [h.uniform_quantizer_1bit, 
         h.uniform_quantizer_Nbit,
         h.halftone_visual_model_1bit, 
         h.halftone_visual_model_Nbit,
         h.halftone_error_diffusion_1bit,
         ]

    test_img = cv2.imread(img_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    for i in range(len(f)):
        if "1bit" in f[i].__name__: 
            t1 = time.monotonic()
            f[i](test_img)
            t2 = time.monotonic()
        if "Nbit" in f[i].__name__: 
            t1 = time.monotonic()
            f[i](test_img, N)
            t2 = time.monotonic()

        print("-----------------------------------------------------")
        delta_t = t2 - t1
        if (delta_t <= 1e-3):
            print(f"Time for {f[i].__name__}: {round(delta_t*pow(10, 6), 4)}[μs]")
        elif (delta_t <= 1):
            print(f"Time for {f[i].__name__}: {round(delta_t*pow(10, 3), 4)}[ms]")
        else:
            print(f"Time for {f[i].__name__}: {round(delta_t, 4)}[s]")
    print("-----------------------------------------------------")


def test_halftone(img_path, N=1):
    h = Halftone()
    f = [    
        #  h.halftone_visual_model_1bit, 
        #  h.halftone_visual_model_Nbit,
        #  h.halftone_error_diffusion_1bit,
        #  h.uniform_quantizer_1bit, 
         h.uniform_quantizer_Nbit, 
        #  h.logarithmic_quantizer_Nbit, 
    ]
    names = [
        #  "halftone_visual_model_1bit", 
        #  "halftone_visual_model_Nbit",
        #  "halftone_error_diffusion_1bit_Floyd_Steinberg",
        #  "halftone_error_diffusion_1bit_Jarvis",
        #  "uniform_quantization_1bit", 
         "uniform_quantization_Nbit",
        #  "logarithmic_quantizer_Nbit",
    ]

    test_img = cv2.imread(img_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    res = []
    for i in tqdm(range(len(f))):
        if "error_diffusion" in f[i].__name__:
            res.append(f[i](test_img, "fs"))
            res.append(f[i](test_img, "jjn"))
        elif "1bit" in f[i].__name__: 
            res.append(f[i](test_img))
        elif "Nbit" in f[i].__name__: 
            res.append(f[i](test_img, N))
    
    for i in range(len(names)):
        plt.title(f"{names[i]}", fontsize=8)
        plt.imshow(res[i], cmap="gray")
        plt.show()
    


if __name__ == "__main__":

    # Time Test
    img_path = "/Users/tsy/Vscode/Projects/halftone_slam/datasets/samples/imgs/freiburg.png"

    test_halftone(img_path=img_path, N=8)

    # test_time(img_path, 5)
    # for i in range(8):
    #     test_halftone(img_path=img_path, N=i+1)
    