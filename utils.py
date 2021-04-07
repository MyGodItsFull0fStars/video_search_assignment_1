import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import wraps
import time
import os
import shutil


def manhattan_distance(hist_left: np.ndarray, hist_right: np.ndarray) -> int:
    assert len(hist_left) == len(hist_right), 'Length of histograms does not match!'
    temp_sum: int = 0
    for i in range(len(hist_left)):
        temp_sum += abs(hist_left[i] - hist_right[i])
    return temp_sum


def euclidian_distance(hist_left: np.ndarray, hist_right: np.ndarray) -> float:
    temp_sum: float = 0
    for i in range(len(hist_left)):
        temp_sum += np.sqrt(np.square(hist_left[i] - hist_right[i]))
    return temp_sum


def plot_image_list(image_list: list, cols: int = 5):
    rows: int = len(image_list) // cols
    if rows % 1 != 0:
        rows += 1

    plt.axis('off')

    image_idx: int = 0
    for i in range(rows):
        for k in range(cols):
            plt.subplot((i + 1), cols, (k + 1))
            current_image = image_list[image_idx]
            plt.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
            image_idx += 1

    plt.show()


def write_images_to_disk(image_list: list) -> None:
    directory_path: str = os.getcwd()
    path = os.path.join(directory_path, 'detected_shots')
    if os.path.exists('detected_shots'):
        delete_directory_and_contents('detected_shots')

    os.mkdir(path)
    os.chdir(path)
    for idx, image in enumerate(image_list):
        cv2.imwrite('image_{}.jpg'.format(idx), image)
    os.chdir(directory_path)


def delete_directory_and_contents(directory_path: str) -> None:
    try:
        shutil.rmtree(directory_path)
    except OSError as err:
        print('Error: {} : {}'.format(directory_path, err.strerror))


def time_decorator(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
        t_start = time.time()
        output = my_func(*args, **kw)
        t_end = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (t_end - t_start) * 1000))
        return output

    return timed
