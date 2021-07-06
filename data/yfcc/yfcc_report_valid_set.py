import os, glob

yfcc_dataset_dir = '/media/lihengl/t2000/Dataset/yfcc100/set'
frame_list_thres = 4800

valid_scene = []
yfcc_scenes = glob.glob(os.path.join(yfcc_dataset_dir, '*'))
for yfcc_scene in yfcc_scenes:
    if not os.path.isdir(yfcc_scene):
        continue

    imgs = glob.glob(os.path.join(yfcc_scene, '*.jpg'))
    num_imgs = len(imgs)
    bundle_file_path = os.path.join(yfcc_scene, 'bundle.out')
    scene_name = yfcc_scene.split('/')[-1].strip()
    edge_cache_file_path = os.path.join(yfcc_scene, 'edge_feat_pos_cache.bin')

    if frame_list_thres > num_imgs and os.path.exists(bundle_file_path) and os.path.exists(edge_cache_file_path):
        print("{'name': '%s',    'bundle_prefix': 'yfcc', 'valid_num': %d}," % (scene_name, num_imgs))
        # print(scene_name, 'bundle', num_imgs)
        valid_scene.append(scene_name)
    # elif not os.path.exists(bundle_file_path) and os.path.exists(edge_cache_file_path):
        # print("{'name': '%s',    'bundle_prefix': 'yfcc', 'valid_num': %d}," % (scene_name, num_imgs))

print('Valid scenes: %d' % len(valid_scene))