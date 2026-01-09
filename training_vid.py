import os
import shutil
import imageio
from tqdm import tqdm

## makes mp4 video of in-training output images

# directory = "checkpoints/2025-01-06_14-23-18_traffic_8hidden/lora_sequential_2025-01-06_17-28-40_rank8"
directory = "checkpoints/2025-01-18_19-03-07_big_buck_bunny/lora_2025-01-19_17-11-25_rank16"
max_num_frames = 200
image_ext = "png"
fps = 25
retrieve_trained_model_output_from_single_dir = False  #if true, frames are named trained_model_output_ecochx
save_name = "training" if retrieve_trained_model_output_from_single_dir else "lora_output"


def listdir_by_time(path):
    """List files in a directory sorted by modification time."""
    files = os.listdir(path)
    files_with_mtime = [(f, os.path.getmtime(os.path.join(path, f))) for f in files]
    files_with_mtime.sort(key=lambda x: x[1])  # Sort by modification time (oldest first)
    return [f[0] for f in files_with_mtime]

def frames_to_mp4(frames, output_path, fps=10):
    """Converts a list of image frames to an MP4 video."""
    with imageio.get_writer(output_path, format='mp4', mode='I', fps=fps, quality=9) as writer:
        for frame in tqdm(frames):
            writer.append_data(frame)
            

if retrieve_trained_model_output_from_single_dir:
    # frames are named trained_model_output_ecochx
    filenames = listdir_by_time(directory)
    filenames = [f"{directory}/{training_png}" for training_png in filenames if training_png.startswith("trained_model_output")]
    
else:
    # get frames from lora_frame{i} subdirs in given directory (use best frame of trained model in each subdir)
    filenames = listdir_by_time(directory)
    filenames = [f"{directory}/{lora_frame_dir}/trained_model_output.{image_ext}" for lora_frame_dir in filenames if lora_frame_dir.startswith("lora_frame")]

filenames = filenames[:max_num_frames]
print("num frames: ", len(filenames))
print(filenames)

### temp: collect all pngs into single dir
# frame_seq_savedir = "../collected_frames_highway_finetune" 
# os.makedirs(frame_seq_savedir, exist_ok=True)
# for png_filepath in filenames:
#     frame_filename = os.path.basename(os.path.dirname(png_filepath)) # "frame_xxxxx"
#     print(frame_filename)
#     shutil.copy2(png_filepath, f"{frame_seq_savedir}/{frame_filename}.{image_ext}") 
# exit()
###

frames = [imageio.v2.imread(filename) for filename in filenames]  
frames_to_mp4(frames, f"{directory}/{save_name}.mp4", fps=fps)

