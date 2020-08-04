# carla-car-detection

Given an input video, you can use this to output a log file of bounding boxes around traffic signs/lights, cars, bicycles, and motorcycles. Each line of the log file is in the format: 

```
frame # | x_min | x_max | y_min | y_max
```

You can also use a folder of images as input, in case there are issues with converting the list of images into a video.

## Dependencies ##
Python 3 is required; the repository has been tested with Python 3.6 running on Ubuntu 16.04 and 18.04.

Some of the files in this repository are stored using [Git Large File Storage](https://git-lfs.github.com/).  It is easiest to set up Git LFS before cloning this repository.  Otherwise, use `git lfs pull` after cloning and initializing Git LFS to ensure you have the full trained models.

If using pip to manage dependencies:
* pip install opencv-python absl-py matplotlib numpy pillow
* pip install tensorflow-gpu==1.14

## Getting the bounding box log file from a video ##
Run the following from the root directory
```
python detection.py --model_path=full_trained_model/detectors-40264 --video_path=/path/to/video/file --min_threshold=0.70 --output_path=/path/to/output/folder
```

Threshold determines the level of certainty required for a bounding box to be reported (higher values result in more false positives). \
To use a different model, you can change `model_path` accordingly (ex. `--model_path=/trained_model/detectors-9614`).

The detection log is saved as `{video_name}_log.txt` (ex. `scenario1_log.txt` for `scenario1.avi`).

## Getting the bounding box log file from a folder of images ##
Run the following from the root directory
```
python image_detection.py --model_path=full_trained_model/detectors-40264 --images_path=/path/to/images/folder --min_threshold=0.70 --output_path=/path/to/output/folder
```

Threshold and `model_path` are the same parameters as described above.

The detection log is saved as `{folder_name}_log.txt` (ex. `rgb_log.txt` for `scenario1/rgb/`).

## Getting screenshots and videos ##
Add a `--save_images` argument to either of the prior scripts to output the frames (`.png`) and a video concatenating the frames (`.avi`). If running the script on a folder of images, the frames will be concatenated in the same order as the sorted file names.

## Converting a folder of images to a video ##
If you want to convert a folder of images to a video instead, you can run:
```
python image_to_video_converter.py --image_folder=/path/to/image/folder --video_path=/path/to/output/video --fps=desired_integer_fps
```

## Evaluation Results ##
The log file storing the mAP scores (and some other metrics) is located at: [Evaluation Metrics](evaluations/full_evaluation/log.txt). The results are in ascending order of the number of iterations for the models present in [trained models](full_trained_model) 

For a cleaner representation, consider exporting the `csv` from tensorboard as described below.

## Evaluation Visualizations ##
If you can view tensorboard results on your machine, use the following from the root directory:
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

## Acknowledgements ##
The model was trained and evaluated with the data provided in: https://github.com/DanielHfnr/Carla-Object-Detection-Dataset
