import shot_detection
import utils

if __name__ == '__main__':
    print('start')
    st = shot_detection.ShotDetection()

    detected_keyframes = st.get_keyframes('everest.mp4')
    utils.write_images_to_disk(detected_keyframes)

    print('end')
