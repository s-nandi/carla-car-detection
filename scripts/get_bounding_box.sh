#!/bin/bash

root=~/Projects/carla-car-detection/

for iter in 35684; #1457 5230 10667 15323 20518 25793 30303 35684;
do
	for scenario in 3 #1 2 3 4;
	do
		echo "Processing scenario "$scenario" using model trained " \
			$iter" iterations"
		result_dir=results/scenario$scenario
		mkdir -p $result_dir
		python3 detection.py \
			--model_path=$root/full_trained_model/detectors-$iter \
			--video_path=$root/data/scenario$scenario"_rgb.avi" \
			--min_threshold=0.70 \
			--output_path=$root$result_dir
	done
done
