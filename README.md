# Shot Detection


## Installation

This assignment used Anaconda as the Python environment.

To install this environment, please enter the following command in the root directory of this project in your terminal:

`conda env create`. Then activat the environment using the command: `conda activate video_search_assignment_1`.

This should install the dependencies needed for this assignment.

Note: OpenCV version `4.5.1.48` was used for this project, but was not specified in the `.yml` file and all OpenCV versions >= `4.2 should probably work just fine.

## Content

The first assignment was to apply shot detection for the provided video `"everest.mp4"`.
This was done by calculating a 64-bin histogram for each extracted frame and calculating the difference between the neighbouring frames.

In this version, the selected key frame is simply the frame in the middle of a shot range, 
and it is planned to replace this with a clustering method in future versions.