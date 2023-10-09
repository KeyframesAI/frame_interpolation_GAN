import os
import cv2

from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim

IN_PATH = 'kinetics'
OUT_PATH = 'frames/kinetics'
#DIMS = 64
DIMS = 256
NAME_START_ID = 0
IS_ANIME = False
SKIP = 1 if IS_ANIME else 3
CHECK_SIMILARITY = True
MAX_SIMILARITY = 0.99 if IS_ANIME else 0.999

def video_to_rgb(n, video_filename, out_dir, resize_shape):
    file_template = 'frame_{0:09d}.jpg'
    reader = cv2.VideoCapture(video_filename)
    success, frame1 = reader.read()

    count = 0
    while success:
        frame1 = cv2.resize(frame1, resize_shape)
        
        success, frame2 = reader.read()
        if not success:
            break
        frame2 = cv2.resize(frame2, resize_shape)
        
        similarity = 0.0
        if count % SKIP != 0:
            similarity = 2.0
        elif CHECK_SIMILARITY:
            similarity = ssim(frame1, frame2, win_size=DIMS-1, channel_axis=2)

        if count % SKIP == 0 and similarity < MAX_SIMILARITY:
            #out_filepath = os.path.join(out_dir, file_template.format(count))
            out_filepath = os.path.join(out_dir, file_template.format(n))
            cv2.imwrite(out_filepath, frame2)
            n += 1
        
        frame1 = frame2
        count += 1
    return n

def process_videofile(n, video_filename, video_path, rgb_out_path, file_extension: str ='.mp4'):
    filepath = os.path.join(video_path, video_filename)
    video_filename = video_filename.replace(file_extension, '')
    OUT_HEIGHT_WIDTH = (DIMS, DIMS)

    out_dir = rgb_out_path
    if (not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    return video_to_rgb(n, filepath, out_dir, resize_shape=OUT_HEIGHT_WIDTH)


if __name__ == '__main__':
    # the path to the folder which contains all video files (mp4, webm, or other)
    video_path = IN_PATH
    # the root output path where RGB frame folders should be created
    rgb_out_path = OUT_PATH
    # the file extension that the videos have
    file_extension = '.mp4'
    # hight and width to resize RGB frames to
    
    if (not os.path.isdir(rgb_out_path)):
        os.mkdir(rgb_out_path)
    

    video_filenames = os.listdir(video_path)

    print('This can take an hour or two depending on dataset size')
    
    n = NAME_START_ID
    for video_filename in video_filenames:
        n = process_videofile(n, video_filename, video_path, rgb_out_path, file_extension)

    print('all done')
