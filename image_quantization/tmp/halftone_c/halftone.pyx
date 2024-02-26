import numpy as np
cimport numpy as np
cimport cython
from cython import boundscheck, wraparound


cpdef inline np.ndarray[unsigned char, ndim=2] uniform_quantizer_1bit_cy(np.ndarray[unsigned char, ndim=2] img):
    img[img >= 128] = 255
    img[img < 128]  = 0
    
    return img


cpdef inline np.ndarray[unsigned char, ndim=2] uniform_quantizer_Nbit_cy(np.ndarray[unsigned char, ndim=2] img, int N):
    cdef int level = pow(2, N)
    cdef double d = 255 / (level-1)
    
    return np.round(np.round(img / d) * d)


# Halftone with ODQ
cpdef inline np.ndarray[unsigned char, ndim=2] halftone_visual_model_Nbit_cy(np.ndarray[unsigned char, ndim=2] img, int N):
    # parameters
    cdef double A0 = 0.21
    cdef double A1 = 0.32
    cdef double A2 = 0.31
    cdef double A3 = 0.13
    cdef double B  = 0.04
    cdef double C  = 1.00
    
    cdef int level = pow(2, N)
    cdef double d = 255 / (level-1)

    # processing
    cdef double AQ0 = A0
    cdef double AQ1 = A1
    cdef double AQ2 = A2
    cdef double AQ3 = A3
    cdef double BQ  = B
    cdef double CQ0 = -AQ0 / B
    cdef double CQ1 = -AQ1 / B
    cdef double CQ2 = -AQ2 / B
    cdef double CQ3 = -AQ3 / B
    
    cdef np.ndarray[double, ndim=2] xi = np.zeros((img.shape[1]+2, img.shape[0]+1), dtype=np.double)
    cdef np.ndarray[double, ndim=2] vd = np.zeros((img.shape[1], img.shape[0]), dtype=np.double)
    cdef np.ndarray[unsigned char, ndim=2] ud = img.T
    
    cdef int i, j
    with boundscheck(False), wraparound(False):
        for j in range(img.shape[0]):
            for i in range(img.shape[1]):
                vd[i,j] = np.round((np.array([[CQ0,CQ1,CQ2,CQ3,1]]) @ np.array([[xi[i,j],xi[i,j+1],xi[i+1,j], xi[i+2,j],ud[i,j]]]).T / d)) * d
                xi[i+1, j+1] = np.array([[AQ0,AQ1,AQ2,AQ3,BQ]]) @ np.array([[xi[i,j],xi[i,j+1],xi[i+1,j],xi[i+2,j],(vd[i,j] - ud[i,j])]]).T

        vd = vd.T
    
    return vd

# Halftone with ODQ(1bit)
cpdef inline np.ndarray[unsigned char, ndim=2] halftone_visual_model_1bit_cy(np.ndarray[unsigned char, ndim=2] img):
    # parameters
    cdef double A0 = 0.21
    cdef double A1 = 0.32
    cdef double A2 = 0.31
    cdef double A3 = 0.13
    cdef double B  = 0.04
    cdef double C  = 1.00
    
    cdef int threshold = 128

    # processing
    cdef double AQ0 = A0
    cdef double AQ1 = A1
    cdef double AQ2 = A2
    cdef double AQ3 = A3
    cdef double BQ  = B
    cdef double CQ0 = -AQ0 / B
    cdef double CQ1 = -AQ1 / B
    cdef double CQ2 = -AQ2 / B
    cdef double CQ3 = -AQ3 / B
    
    cdef np.ndarray[double, ndim=2] xi = np.zeros((img.shape[1]+2, img.shape[0]+1), dtype=np.double)
    cdef np.ndarray[double, ndim=2] vd = np.zeros((img.shape[1], img.shape[0]), dtype=np.double)
    cdef np.ndarray[unsigned char, ndim=2] ud = img.T
    
    cdef int i, j
    cdef np.ndarray[int, ndim=2] tmp
    with boundscheck(False), wraparound(False):
        for j in range(img.shape[0]):
            for i in range(img.shape[1]):
                if np.array([[CQ0,CQ1,CQ2,CQ3,1]]) @ np.array([[xi[i,j],xi[i,j+1],xi[i+1,j], xi[i+2,j],ud[i,j]]]).T > threshold:
                    vd[i,j] = 255
                xi[i+1, j+1] = np.array([[AQ0,AQ1,AQ2,AQ3,BQ]]) @ np.array([[xi[i,j],xi[i,j+1],xi[i+1,j],xi[i+2,j],(vd[i,j] - ud[i,j])]]).T

    return vd.T
    

# halftone with error diffusion
cpdef np.ndarray[unsigned char, ndim=2] halftone_error_diffusion_1bit_cy(np.ndarray[unsigned char, ndim=2] img, str method):
    # parameters
    cdef unsigned char thresh = 128
    cdef np.ndarray[double, ndim=2] error = np.zeros((img.shape[0], img.shape[1]))

    cdef np.ndarray[double, ndim=2] h
    cdef int FSTEPJ, BSTEPJ, FSTEPI
    
    # Processing
    if method == "jjn":   # Jarvis, Judice & Ninke
        h = np.array([[0, 0, 0, 7, 5],[3, 5, 7, 5, 3],[1, 3, 5, 3, 1]]) / 48 ;
        FSTEPJ = 2
        BSTEPJ = -2
        FSTEPI = 2
    if method == "fs":    # Floyd-Steinberg
        h = np.array([[0, 0, 7],[3, 5, 1]]) / 16
        FSTEPJ = 1
        BSTEPJ = -1
        FSTEPI = 1

    cdef np.ndarray[double, ndim=2] tmp1 = np.concatenate((np.ones((img.shape[0], -BSTEPJ), dtype=np.double), img, np.ones((img.shape[0], FSTEPJ), dtype=np.double)), axis=1)
    cdef np.ndarray[double, ndim=2] tmp2 = np.concatenate((np.ones((FSTEPI, -BSTEPJ), dtype=np.double), np.ones((FSTEPI, img.shape[1]), dtype=np.double), np.ones((FSTEPI, FSTEPJ), dtype=np.double)), axis=1)
    cdef np.ndarray[double, ndim=2] tmp = np.concatenate((tmp1, tmp2), axis=0)

    cdef np.ndarray[unsigned char, ndim=2] newImage = np.zeros((img.shape[0]+FSTEPI, img.shape[1]+FSTEPJ-BSTEPJ), dtype=np.uint8)
    
    cdef double err1, err2, err3
    cdef int i, j
    with boundscheck(False), wraparound(False):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]+FSTEPJ):
                if (i > img.shape[0] or j > img.shape[1] - BSTEPJ or j <= -BSTEPJ):
                    if tmp[i,j] > thresh:
                        newImage[i,j] = 255
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
                        for l in range(BSTEPJ, FSTEPJ + 1):
                            tmp[i+k-1, j+l] += err3 * h[k-1, -1*BSTEPJ+l]
    
    return newImage[:-FSTEPI, :-(FSTEPJ-BSTEPJ)]