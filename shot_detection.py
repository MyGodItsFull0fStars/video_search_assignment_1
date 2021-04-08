import cv2
from utils import manhattan_distance, time_decorator, get_opencv_histogram_bin


class FrameData:

    def __init__(self, frame=None):
        self.frame = frame
        self.histogram_bin = get_opencv_histogram_bin(self.frame)


FrameDataList = [FrameData]


class ShotDetection:

    def __init__(self):
        # Capture frame-by-frame
        self._detected_shots: list = []

    def get_keyframes(self, video_path: str = None) -> list:
        print('[Capture Video]')
        if video_path is None or video_path == '':
            raise Exception('Video path must be provided!')

        video_frame_list = self._transform_video_to_frame_data_list(video_path)
        self._keyframe_detection(video_frame_list)

        return self._detected_shots

    @time_decorator
    def _transform_video_to_frame_data_list(self, video_path: str = None) -> FrameDataList:
        print('[Transform Video To Frama Data List]')
        if video_path is None or video_path == '':
            raise Exception('Video path must be provided!')
        frames: FrameDataList = []

        video_capture = cv2.VideoCapture(video_path)

        count = 0

        while video_capture.isOpened():
            # while video_capture.isOpened() and count < 1000:
            valid_frame, frame = video_capture.read()
            if valid_frame:
                frames.append(FrameData(frame))
            else:
                break
            count += 1

        video_capture.release()

        return frames

    def _keyframe_detection(self, video_frame_list: FrameDataList, threshold_d: int = 200000, threshold_h: int = 80000):
        print('[Detecting Keyframes]')
        T_D: int = threshold_d
        T_H: int = threshold_h

        first_frame: int = 0

        cumulative_threshold = 0
        for left_idx in range(len(video_frame_list) - 1):
            right_idx = left_idx + 1
            left_histogram = video_frame_list[left_idx].histogram_bin
            right_histogram = video_frame_list[right_idx].histogram_bin

            md = manhattan_distance(left_histogram, right_histogram)
            cumulative_threshold += md

            if md > T_D or cumulative_threshold > T_H:
                detected_shot_idx = first_frame + (right_idx - first_frame) // 2
                self._detected_shots.append(video_frame_list[detected_shot_idx].frame)
                first_frame = right_idx
                cumulative_threshold = 0
                continue

    def __delete__(self, instance):
        cv2.destroyAllWindows()
