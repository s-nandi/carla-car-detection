import cv2
import os

def video_to_image(video_path, image_folder):
    success, image = vidcap.read()
    frame = 0
    while success:
        image_path = os.path.join(image_folder, f"frame_{frame}.png")
        cv2.imwrite(image_path, image)     # save frame as JPEG file      
        success, image = vidcap.read()
        frame += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video_path', required=True, help='Path to the output video')
    parser.add_argument('--image_folder', required=True, help='Location of the image folder')
    video_to_image(args.video_path, args.image_folder)

    
