# carla-car-detection

## Dependencies ##
* pip install opencv-python absl-py matplotlib numpy
* pip install tensorflow-gpu==1.14
(this might be incomplete)

## Getting the bounding box log file from a video ##
Run the following from the root directory
```
python detection.py --model_path=/trained_model/detectors \
                     --video_path=/path/to/video/file \
                     --min_threshold=0.70 
                     --output_path=/path/to/output/folder
```

Threshold determines the level of certainty required for a bounding box to be reported (higher values result in more false positives)

## Getting screenshots and videos ##
Modify the above command as follows
```
python detection.py --model_path=/trained_model/detectors \
                     --video_path=/path/to/video/file \
                     --min_threshold=0.70 
                     --output_path=/path/to/output/folder
                     --save_images
```
Each frame will be saved as a `.png` file, and the concatenated video will be saved as a `.avi` file

## Converting a folder of images to a video ##
If object detection is needed for a set of images in a folder, first use 
```
python image_to_video_converter.py --image_folder=/path/to/image/folder --video_path=/path/to/output/video --fps=desired_integer_fps
```
Then use the outputted video for either of the previous two object detection scripts

## Example Output ##
[![Right Turn](https://img.youtube.com/vi/yQ0sntd1y8k/0.jpg)](https://www.youtube.com/watch?v=yQ0sntd1y8k)
