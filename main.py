import shot_detection
import cv2

if __name__ == '__main__':
    print('start')
    st = shot_detection.ShotDetection()
    st.capture_video('everest.mp4')

    print(len(st.detected_shots))

    print('end')
