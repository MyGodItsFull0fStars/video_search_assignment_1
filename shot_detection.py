import cv2
import numpy as np
from copy import deepcopy


def _get_histogram_bin(frame) -> list:
    return np.histogram(frame.ravel(), 64, [0, 256])[0]


class FrameData:

    def __init__(self, frame=None):
        self.frame = deepcopy(frame)
        self.histogram_bin = _get_histogram_bin(self.frame)


FrameDataList = [FrameData]


class ShotDetection:

    def __init__(self):
        # Capture frame-by-frame
        self.detected_shots: list = []

    def capture_video(self, video_path: str = None) -> None:
        print('[Capture Video]')
        if video_path is None or video_path == '':
            raise Exception('Video path must be provided!')

        video_frame_list = self.transform_video_to_frame_data_list(video_path)
        self.keyframe_detection(video_frame_list)
        # self.detected_shots = [elem.frame for elem in video_frame_list]

    def transform_video_to_frame_data_list(self, video_path: str = None) -> FrameDataList:
        print('[Transform Video To Frama Data List]')
        if video_path is None or video_path == '':
            raise Exception('Video path must be provided!')
        frames: FrameDataList = []

        video_capture = cv2.VideoCapture(video_path)

        while video_capture.isOpened():
            valid_frame, frame = video_capture.read()
            if valid_frame:
                frames.append(FrameData(frame))
            else:
                break

        video_capture.release()

        return frames

    def keyframe_detection(self, video_frame_list: FrameDataList):
        print('[Detecting Keyframes]')
        T_D: int = 150000
        T_H: int = 700000

        first_frame: int = 0
        last_frame: int = 0

        cumulative_threshold = 0
        for left_idx in range(len(video_frame_list) - 1):
            right_idx = left_idx + 1
            left_histogram = video_frame_list[left_idx].histogram_bin
            right_histogram = video_frame_list[right_idx].histogram_bin
            last_frame = right_idx

            md = self.manhattan_distance(left_histogram, right_histogram)
            cumulative_threshold += md

            if md > T_D:
                detected_shot_idx = (first_frame + last_frame) // 2
                print('Adding shot {} with T_D'.format(detected_shot_idx))
                self.detected_shots.append(video_frame_list[detected_shot_idx])
                first_frame = right_idx
                cumulative_threshold = 0
                continue

            if cumulative_threshold > T_H:
                detected_shot_idx = (first_frame + last_frame) // 2
                print('Adding shot {} with T_H'.format(detected_shot_idx))
                self.detected_shots.append(video_frame_list[detected_shot_idx])
                first_frame = right_idx
                cumulative_threshold = 0
                continue

            # print('Manhattan Distance of frames ({}/{}) is = {}'.format(left_idx, right_idx, md))

    def manhattan_distance(self, hist_left: np.ndarray, hist_right: np.ndarray) -> int:

        sum: int = 0
        for i in range(len(hist_left)):
            sum += abs(hist_left[i] - hist_right[i])
        return sum

    def euclidian_distance(self, hist_left: np.ndarray, hist_right: np.ndarray) -> float:
        sum: float = 0
        for i in range(len(hist_left)):
            sum += np.sqrt(np.square(hist_left[i] - hist_right[i]))
        return sum

    def __delete__(self, instance):
        cv2.destroyAllWindows()
