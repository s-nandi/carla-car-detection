import cv2
import os

def image_to_video(image_folder, video_path, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    if '.' not in video_path:
        video_path += '.avi'
    video = cv2.VideoWriter(video_path, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--image_folder', required=True, help='Location of the image folder')
    parser.add_argument('--video_path', required=True, help='Path to the output video')
    parser.add_argument('--fps', type=int, default=30)

    args = parser.parse_args()
    # Name of the directory containing the object detection module we're using
    image_to_video(args.image_folder, args.video_path, args.fps)
