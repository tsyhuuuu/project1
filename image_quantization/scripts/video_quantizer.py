import os
import time
import pathlib
from tqdm import tqdm
from glob import glob

import cv2
from multiprocessing import Process
from image_quantizer import ImageQuantization



class VideoQuantization(object):
    def __init__(self, RAW_IMAGE_DIR=None, QUANTIZED_IMAGE_DIR=None):
        super().__init__()

        self.raw_image_list       = None
        self.raw_image_dir        = RAW_IMAGE_DIR
        self.quantized_image_names = None
        self.quantized_image_dir   = QUANTIZED_IMAGE_DIR

        self.img_q = ImageQuantization()


    def create_quantized_video(self, q_method="uniform", bit=None, w=None, err_diff="jjn", rename=False, suffix="png"):

        if q_method == "uniform":
            quantizer = self.img_q.uniform_quantization
        elif q_method == "blockwise":
            quantizer = self.img_q.blockwise_quantization
        elif q_method == "error_diffusion":
            quantizer = self.img_q.error_diffusion_quantization
        else:
            print(f"quantization method {q_method} does not exist!")
            return

        self.raw_image_list = glob(f"{self.raw_image_dir}/*")
        if rename is True:
            self.quantized_image_names =  [f"quantized_img{idx}.{suffix}" for idx in range(len(self.raw_image_list))]
        else:
            self.quantized_image_names = [os.path.basename(filepath) for filepath in self.raw_image_list]

        if self.quantized_image_dir is None:
            try:
                self.quantized_image_dir = pathlib.PurePath(self.raw_image_dir, f"quantized_{q_method}_{bit}bit")
                os.mkdir(pathlib.PurePath(self.halftone_image_dir))
            except Exception as e:
                pass

        for idx in tqdm(range(len(self.raw_image_list))):
            img = cv2.imread(self.raw_image_list[idx])
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if q_method == "error_diffusion":    
                if err_diff == "custom":
                    quantized_image = quantizer(img=img_gray, h=w, method=err_diff, N=bit)
                elif err_diff == "visual model":
                    quantized_image = quantizer(img=img_gray, N=bit)
                else:
                    quantized_image = quantizer(img=img_gray, method=err_diff, N=bit)
            else:
                quantized_image = quantizer(img_gray, bit)
            
            cv2.imwrite(str(pathlib.PurePath(self.quantized_image_dir, self.quantized_image_names[idx])), quantized_image)
        
        print(f" ===================================")
        print(f"=== QUANTIZATION COMPLETE b ^_^ d ===")
        print(f" ===================================")
        print(f"[SAVED DIRECTORY] {str(self.quantized_image_dir)}")


class MultiProcessVideoQuantization(object):

    def __init__(self, num_process=1):
        super().__init__()
        self.num_process = num_process

    def quantization_process(self, raw_video_dir, goal_video_dir, q_method=None, bit=None, w=None, err_diff=None):
        q_video = VideoQuantization(RAW_IMAGE_DIR=raw_video_dir, QUANTIZED_IMAGE_DIR=goal_video_dir)
        q_video.create_quantized_video(q_method=q_method, bit=bit, w=w, err_diff=err_diff)

    def multi_video_quantization(self, raw_video_dirs, goal_video_dirs, q_params):
        if len(raw_video_dirs) != len(goal_video_dirs):
            print("len(raw_video_dirs) not equal to len(goal_video_dirs)!")
            return 
        
        PROCESSES = []
        while not raw_video_dirs == []:
            if len(PROCESSES) < self.num_processes:
                q_param = q_params.pop(0)
                p = Process(target=self.quantization_process, args=[raw_video_dirs.pop(0), goal_video_dirs.pop(0), q_param[0], q_param[1], q_param[2], q_param[3]])
                PROCESSES.append(p)
                p.start()
                time.sleep(1.0)

            if len(PROCESSES) == self.num_processes or raw_video_dirs == []:
                for p in PROCESSES:
                    p.join()
                    PROCESSES.remove(p)


def test():
    RAW_DIR = "/Users/tsy/Vscode/Projects/halftone_slam/datasets/samples/imgs"
    # GOAL_DIR = "/Users/tsy/Vscode/Projects/halftone_slam/datasets/halftone_imgs"
    h_video = VideoQuantization(RAW_IMAGE_DIR=RAW_DIR)
    h_video.create_halftone_video(q_method="visual_model", bit=1)

        

if __name__ == "__main__":
    test()