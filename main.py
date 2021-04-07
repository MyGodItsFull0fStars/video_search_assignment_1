import shot_detection
import cv2
import matplotlib.pyplot as plot
import utils
import numpy as np

if __name__ == '__main__':
    print('start')
    st = shot_detection.ShotDetection()
    st.capture_video('everest.mp4')

    print(len(st.detected_shots))

    # utils.plot_image_list(st.detected_shots)

    utils.write_images_to_disk(st.detected_shots)
    # plot.imshow(cv2.cvtColor(st.detected_shots[0], cv2.COLOR_BGR2RGB))
    # plot.show()

    print('end')
