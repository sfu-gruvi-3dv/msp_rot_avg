import numpy as np
import cv2, os

""" Configure
"""
file_path = '/Users/corsy/Downloads/IMG_1217.MOV'

output_dir = '/Users/corsy/Downloads/IMG_1217'

output_file_prefix = 'f_'

output_file_extention = 'jpg'

skip_frames = 10

show = True

""" Script
"""
if not os.path.exists(file_path):
    print('File %s not exist' % file_path)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

cap = cv2.VideoCapture(file_path)
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    if frame is not None:

        if show:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if i % skip_frames == 0:
            cv2.imwrite(os.path.join(output_dir, '%s%05d.%s' % (output_file_prefix, i, output_file_extention)), frame)
    else:
        break

    i += 1

cap.release()
cv2.destroyAllWindows()
