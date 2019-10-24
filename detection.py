import sys
from pathlib import Path, PurePath
sys.path.append("./models/research/object_detection/")
sys.path.append("./models/research/")

import os
import cv2
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from image_to_video_converter import images_to_video
from PIL import Image

class detector:
    def __init__(self, model_directory):
        model_path = os.path.join(model_directory, 'frozen_inference_graph.pb')
        labelmap_path = os.path.join(model_directory, 'labelmap.pbtxt')
        self.num_classes = 5
        self.label_map = label_map_util.load_labelmap(labelmap_path)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                    max_num_classes=self.num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                self.serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
            tf.import_graph_def(self.od_graph_def, name='')
        
            self.sess = tf.Session(graph=self.detection_graph)
        # Define input and output tensors (i.e. data) for the object detection classifier
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def draw_boxes_for_image(self, frame, min_score_threshold):
        frame_expanded = np.expand_dims(frame, axis=0)   
        (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: frame_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=min_score_threshold)
        
        """
        print("Self cateogry index")
        print(self.category_index)
        print("Score/Classes")
        for box, score, cls in zip(np.squeeze(boxes), np.squeeze(scores),np.squeeze(classes).astype(np.int32)):
            print(score, cls, self.category_index[cls])
        """
            
        good_boxes = [box
                      for box, score, cls in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))
                      if score >= min_score_threshold and 'traffic' not in self.category_index[cls]['name']]
        return frame, good_boxes

    @staticmethod
    def denormalize(box, width, height):
        # Order taken from: https://www.tensorflow.org/api_docs/python/tf/image/draw_bounding_boxes
        y_min, x_min, y_max, x_max = box[0], box[1], box[2], box[3]
        x_min *= width
        x_max *= width
        y_min *= height
        y_max *= height
        return [x_min, x_max, y_min, y_max]

    @staticmethod
    def log_boxes(frame_number, boxes, ofile, width, height):
        for box in boxes:
            box = detector.denormalize(box, width, height)
            # Cast float coordinates to integers
            box = map(int, box)
            box = [frame_number] + list(box)
            line = "|".join(map(str, box))
            print(line, file=ofile)
            
    def process_image(self, video_name, frame_number, image_path,
                      min_score_threshold, output_path, save_images):
        image = cv2.imread(image_path)
        image_name = Path(image_path).stem
        # Set up logging file
        log_name = os.path.join(output_path, f"{video_name}_log.txt")
        with open(log_name, 'a') as log_file:
            print("At Frame:", frame_number)
            frame = np.array(image)
            # Draw boxes
            frame, boxes = self.draw_boxes_for_image(frame, min_score_threshold)
            height, width, layers = frame.shape
            # Log boxes
            detector.log_boxes(frame_number, boxes, log_file, width, height)
            # Save frame with boxes
            if save_images:
                frame_path = os.path.join(output_path, f"{video_name}_frame_{image_name}.png")
                print("Saving image at", frame_path)
                vis_util.save_image_array_as_png(frame, frame_path)

    def process_image_folder(self, folder_path, min_score_threshold, output_path, save_images):
        folder_name = Path(folder_path).stem
        frame_number = 0;
        for f in os.listdir(folder_path):
            image_path = os.path.join(folder_path, f)
            if os.path.isfile(image_path):
                self.process_image(folder_name, frame_number, image_path,
                                   min_score_threshold, output_path, save_images)
                frame_number += 1
                
    def process_video(self, video_path, min_score_threshold, output_path, save_images):
        video_name = Path(video_path).stem
        # Open video file
        video = cv2.VideoCapture(video_path)
        # Set up logging file
        log_name = os.path.join(output_path, f"{video_name}_log.txt")
        with open(log_name, 'a') as log_file:
            frames = []
            while(video.isOpened()):
                ret, frame = video.read()
                if not ret:
                    break
                frame_number = len(frames)
                print("At Frame:", frame_number)
                # Draw boxes
                frame, boxes = self.draw_boxes_for_image(frame, min_score_threshold)
                height, width, layers = frame.shape
                # Log boxes
                detector.log_boxes(frame_number, boxes, log_file, width, height)
                # Save frame with boxes
                if save_images:
                    frame_path = os.path.join(output_path, f"{video_name}_frame_{frame_number}.png")
                    print("Saving image at", frame_path)
                    vis_util.save_image_array_as_png(frame, frame_path)
                frames.append(frame)    
            # Save as video
            if save_images:
                out_video_path = os.path.join(output_path, f"{video_name}.avi")
                print("Saving video at", out_video_path)
                images_to_video(frames, out_video_path, 30)
        # Clean up
        video.release()
        cv2.destroyAllWindows()
        
def default_detector():
    det = detector("./trained_model/detectors/")
    return det

def default_inference():
    det = default_detector()
    det.process_video("./data/SignaledJunctionRightTurn_1.avi", 0.70, "./output/temp/", False)
    return det

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to the frozen inference graph and labelmap files',
                        required=True)
    parser.add_argument('--video_path', help='Path to the video', required=True)
    parser.add_argument('--min_threshold', type=float, help='Minimum score threshold for a bounding box to be drawn', default=0.7)
    parser.add_argument('--output_path', help='Path for storing output images and/or logs', required=True)
    parser.add_argument('--save_images', action='store_true')

    args = parser.parse_args()

    det = detector(args.model_path)
    det.process_video(args.video_path, args.min_threshold, args.output_path, args.save_images)
