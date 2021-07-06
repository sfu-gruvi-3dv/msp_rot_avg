import sys, os

if len(sys.argv) != 3:
    print('Usage: python gen_calibration.py <image_list.txt> <output_calib.txt>')

if not os.path.exists(sys.argv[1]):
    print('File not exist: %s' % sys.argv[1])

input_img_list_file = open(sys.argv[1], 'r')
output_calib_file = open(sys.argv[2], 'w')
lines = input_img_list_file.readlines()

# camera intrinsic
f = 418.71093*4
cx = 240.0*4
cy = 135.0*4
w = 1920
h = 1080

for l in lines:
    output_calib_file.write('%f %f %f %d %d\n' % (f, cx, cy, w, h))

output_calib_file.close()