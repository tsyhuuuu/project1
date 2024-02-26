import os
import pathlib
from tqdm import tqdm
from glob import glob

import cv2
import halftone_image
from halftone_image import Halftone



class HalftoneVideo(object):
    def __init__(self, RAW_IMAGE_DIR=None, HALFTONE_IMAGE_DIR=None):
        super().__init__()

        self.raw_image_list       = None
        self.raw_image_dir        = RAW_IMAGE_DIR
        self.halftone_image_names = None
        self.halftone_image_dir   = HALFTONE_IMAGE_DIR

        self.h = Halftone()


    def create_halftone_video(self, bit=1, option="uniform", w=None, err_diff="jjn", rename=False, suffix="png"):

        if option == "uniform":
            halftone = self.h.uniform_quantizer_1bit if bit == 1 else self.h.uniform_quantizer_Nbit
        elif option == "block_wise":
            halftone = self.h.block_wise_quantizer
        elif option == "visual_model":
            halftone  = self.h.halftone_visual_model_1bit if bit == 1 else self.h.halftone_visual_model_Nbit
        elif option == "error_diffusion":
            halftone = self.h.halftone_error_diffusion_1bit    
        else:
            print(f"quantization method {option} does not exist!")
            return

        self.raw_image_list = glob(f"{self.raw_image_dir}/*")
        if rename is True:
            self.halftone_image_names =  [f"halftone_img{idx}.{suffix}" for idx in range(len(self.raw_image_list))]
        else:
            self.halftone_image_names = [os.path.basename(filepath) for filepath in self.raw_image_list]

        if self.halftone_image_dir is None:
            try:
                self.halftone_image_dir = pathlib.PurePath(self.raw_image_dir, f"halftone_{option}_{bit}bit")
                os.mkdir(pathlib.PurePath(self.halftone_image_dir))
            except Exception as e:
                pass

        for idx in tqdm(range(len(self.raw_image_list))):
            img = cv2.imread(self.raw_image_list[idx])
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if option == "error_diffusion":
                if err_diff == "custom":
                    img_halftone = halftone(img_gray, w, err_diff)
                else:
                    img_halftone = halftone(img_gray, err_diff)
            elif bit == 1:
                img_halftone = halftone(img_gray)
            else:
                img_halftone = halftone(img_gray, bit)
            
            cv2.imwrite(str(pathlib.PurePath(self.halftone_image_dir, self.halftone_image_names[idx])), img_halftone)
        
        print(" ================================")
        print("=== HALFTONE COMPLETE b ^_^ d ===")
        print(" ================================")
        print(f"[SAVED DIRECTORY] {str(self.halftone_image_dir)}")
    

def test():
    RAW_DIR = "/Users/tsy/Vscode/Projects/halftone_slam/datasets/samples/imgs"
    # GOAL_DIR = "/Users/tsy/Vscode/Projects/halftone_slam/datasets/halftone_imgs"
    h_video = HalftoneVideo(RAW_IMAGE_DIR=RAW_DIR)
    h_video.create_halftone_video(option="visual_model", bit=1)

        

if __name__ == "__main__":
    test()