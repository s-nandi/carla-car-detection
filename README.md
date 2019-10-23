# carla-car-detection

## Dependencies ##
* pip install opencv-python absl-py matplotlib numpy pillow
* pip install tensorflow-gpu==1.14
(this might be incomplete)

## Getting the bounding box log file from a video ##
Run the following from the root directory
```
python detection.py --model_path=/full_trained_model/detectors-iteration# --video_path=/path/to/video/file --min_threshold=0.70 --output_path=/path/to/output/folder
```

Threshold determines the level of certainty required for a bounding box to be reported (higher values result in more false positives) \
To use a different model, you can change `model_path` accordingly (ex. `--model_path=/trained_model/detectors-9614`)

## Getting screenshots and videos ##
Use the following command (it just adds a `--save_images` argument to the script)
```
python detection.py --model_path=/full_trained_model/detectors-iteration# --video_path=/path/to/video/file --min_threshold=0.70 --output_path=/path/to/output/folder --save_images
```
Each frame will be saved as a `.png` file, and the concatenated video will be saved as a `.avi` file

## Converting a folder of images to a video ##
If object detection is needed for a set of images in a folder, first use 
```
python image_to_video_converter.py --image_folder=/path/to/image/folder --video_path=/path/to/output/video --fps=desired_integer_fps
```
Then use the outputted video for either of the previous two object detection scripts

## Evaluation Results ##
The log file storing the mAP scores (and some other metrics) is located at: [Evaluation Metrics](evaluations/full_evaluation/log.txt)

For a cleaner representation, consider exporting the `csv` from tensorboard as described below.

## Evaluation Visualizations ##
If you can use tensorboard on your machine, use the following from the root directory:
```
tensorboard --logdir evaluations/full_evaluation
```
You can scroll to the bottom of the rendered page and either save the mAP plot image as a `.svg` or export the mAP values as a `.csv`

You can turn smoothing to 0 to get an exact plot

## Training Visualization ##
Similarly to visualizing the mAP metric, you can use the following from the root directory to visualize training loss metrics:
```
tensorboard --logdir full_trained_model/checkpoints
```

## Example Output ##
[![Right Turn](https://img.youtube.com/vi/yQ0sntd1y8k/0.jpg)](https://www.youtube.com/watch?v=yQ0sntd1y8k)
