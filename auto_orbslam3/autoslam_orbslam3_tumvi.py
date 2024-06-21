"""
RGBD to be continue
"""

# coding: utf-8

import os, time, subprocess
from glob import glob
import psutil as ps
import multiprocessing as mp


SLAM_NAME = "orbslam3"
CAMERA_TYPEs = ["Monocular", "Stereo", "Monocular-VI", "Stereo-VI"]

"""
 [tumvi] raw datasets and quantization datasets 
    - uniform quantization (1~8bit)
    - blockwise quantization
        - Block4
        - Block8
        - Block16
    - error diffusion
        - visual model
        - jarvis
        - floyd
"""
DATASET_TYPEs = ["outdoors4", "outdoors5"]
ALL_DATASET_LIST = [f"{i}_{j}_{k}bit" for i in DATASET_TYPEs for j in ["uniform", "visual", "jarvis", "floyd"] for k in range(1, 8)] + [f"{i}_BW_N{k}" for i in DATASET_TYPEs for k in [4, 8, 16]] + [f"{i}_8bit" for i in DATASET_TYPEs]

TRAJECTORY_SAVE_FOLDER = "/home/ubuntu/Desktop/Results/tumvi/trajectoryFolder/"
MEMORY_SAVE_FOLDER     = "/home/ubuntu/Desktop/Results/tumvi/memoryFolder/"
MPKFS_SAVE_FOLDER      = "/home/ubuntu/Desktop/Results/tumvi/mpkfsFolder/"
RESULTS_SAVE_FILE      = "/home/ubuntu/Desktop/Results/tumvi/results.txt"


def write_to_file(file_path, data):
    with open(file_path, 'a') as file:
        file.write(data)

def touch_file(target_file):
    target_name = os.path.basename(target_file)
    if not os.path.isfile(target_file):
        _ = subprocess.run(["touch", target_file])
        print(f"{target_name} has been created!")
    else:
        print(f"{target_name} has already been existed.")


def slam_and_evaluation(dataset_name, dataset_type, camera_type=0):

    ORBSLAM3_CMDs = [
        f"to be continue",
        f"to be continue",
        f"./Examples/Monocular-Inertial/mono_inertial_tum_vi ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/TUM-VI.yaml /home/ubuntu/Desktop/Datasets/tumvi/{dataset_name}/mav0/cam0/data ./Examples/Monocular-Inertial/TUM_TimeStamps/dataset-{dataset_name[:9]}_512.txt ./Examples/Monocular-Inertial/TUM_IMU/dataset-{dataset_name[:9]}_512.txt {dataset_name}_{camera_type}",
        f"to be continue"
    ]

    EVALUATION_CMDs = [
        f"to be continue",
        f"to be continue",
        f"python3 evaluation/evaluate_ate_scale.py /home/ubuntu/Desktop/Datasets/tumvi/{dataset_name}/mav0/mocap0/data.csv kf_{dataset_name}_{camera_type}.txt",
        f"to be continue",
    ]

    MOVE_FILE_CMD1 = f"mv f_{dataset_name}_{camera_type}.txt kf_{dataset_name}_{camera_type}.txt {TRAJECTORY_SAVE_FOLDER}"

    # MOVE_FILE_CMD2 = f"mv mpkfs_{dataset_name}_{camera_type}.txt {MPKFS_SAVE_FOLDER}"

    print(f"[{camera_type}] Processing {dataset_name} ")

    try:
        CMD = ORBSLAM3_CMDs[camera_type].split(" ")
        obj1 = ps.Popen(CMD, 
                        stdout=subprocess.PIPE,  # 输出结果
                        stderr=subprocess.PIPE,  # 输出 err
                        )
        
        memory_usage_history = []
        while not os.path.isfile(f"f_{dataset_name}_{camera_type}.txt"):
            mem = round(obj1.memory_info().rss / 1024 / 1024, 3)
            memory_usage_history.append(mem)
            time.sleep(3)

        time.sleep(5)
        CMD = EVALUATION_CMDs[camera_type].split(" ")
        obj2 = subprocess.run(CMD, 
                            stdout=subprocess.PIPE, # 输出结果
                            stderr=subprocess.PIPE,  # 输出 err
                            check=True)
        
        # [orbslam3-monocular] MH01_8bit: mem_max mem_ave ate scale ate_gt std_ate
        result = f"[{SLAM_NAME}-{CAMERA_TYPEs[camera_type]}] {dataset_name}: {max(memory_usage_history)} {round(sum(memory_usage_history)/len(memory_usage_history), 3)} {obj2.stdout.decode('utf-8')} \n"
        write_to_file(RESULTS_SAVE_FILE, result)
        print(f"saving memory usage, ate, scale, std_ate in {RESULTS_SAVE_FILE} ...")
 
        CMD = MOVE_FILE_CMD1.split(" ")
        _ = subprocess.run(CMD, 
                            stdout=subprocess.PIPE,  # 输出结果
                            stderr=subprocess.PIPE,  # 输出 err
                            check=True)
        print(f"saving 2 trajectory in {TRAJECTORY_SAVE_FOLDER} ...")

        # CMD = MOVE_FILE_CMD2.split(" ")
        # _ = subprocess.run(CMD, 
        #                     stdout=subprocess.PIPE,  # 输出结果
        #                     stderr=subprocess.PIPE,  # 输出 err
        #                     check=True)
        # print(f"saving mpkfs file in {MPKFS_SAVE_FOLDER} ...")

        touch_file(f"{MEMORY_SAVE_FOLDER}memory_{dataset_name}_{camera_type}.txt")
        memory_usage_history_str = ",".join([str(i) for i in memory_usage_history])
        write_to_file(f"{MEMORY_SAVE_FOLDER}memory_{dataset_name}_{camera_type}.txt", memory_usage_history_str)

        obj1.terminate()



    except Exception as e:
        result = f"[{SLAM_NAME}-{CAMERA_TYPEs[camera_type]}] {dataset_name}: incomplete \n"
        write_to_file(RESULTS_SAVE_FILE, result)
        print(e)



def auto_slam(num_processes=3, camera_type=[0, 1, 2, 3]):
    
    touch_file(RESULTS_SAVE_FILE)

    CURRENT_DATASET_LIST = []
    for dataset_name in ALL_DATASET_LIST:
        dataset_path = f"/home/ubuntu/Desktop/Datasets/tumvi/{dataset_name}"
        if dataset_path in glob("/home/ubuntu/Desktop/Datasets/tumvi/*"):
            CURRENT_DATASET_LIST.append(dataset_name)
    
    CURRENT_DATASET_LIST_BACKUP = CURRENT_DATASET_LIST.copy()
    PROCESSES = []
    for c_type in camera_type:    # [0, 1, 2] --- 0: Monocular; 1: Stereo; 2: RGBD
        while len(CURRENT_DATASET_LIST) != 0:
            if len(PROCESSES) < num_processes:
                dataset_name = CURRENT_DATASET_LIST.pop(0)
                p = mp.Process(target=slam_and_evaluation, args=[dataset_name, dataset_name[:9], c_type])
                PROCESSES.append(p)
                print(f"====== Processing {dataset_name} now... ======")
                p.start()
                time.sleep(3.0)
            if len(PROCESSES) == num_processes:
                for p in PROCESSES:
                    p.join()
                    PROCESSES.remove(p)

        CURRENT_DATASET_LIST = CURRENT_DATASET_LIST_BACKUP.copy()



def main():
    auto_slam(1, [2,2,2])



if __name__ == "__main__":
    main()