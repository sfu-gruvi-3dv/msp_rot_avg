import os, pickle, sys
from tqdm import tqdm
import glob

def extract_meta_info(edge_feat_pos_cache_path, edge_meta_out_path):
    print('Process on %s' % edge_feat_pos_cache_path)

    if not os.path.exists(edge_feat_pos_cache_path):
        print('File not found: %s' % edge_feat_pos_cache_path)

    with open(edge_feat_pos_cache_path, 'rb') as f:
        edge_feat_pos_cache = pickle.load(f)

        new_edge_meta_cache = dict()
        keys = list(edge_feat_pos_cache.keys())
        for key in tqdm(keys):
            new_block = dict()
            edge_cache = edge_feat_pos_cache[key]
            new_block['Rt'] = edge_cache['Rt'].copy()
            new_block['type'] = edge_cache['type']
            new_block['rel_err'] = edge_cache['rel_err']
            new_edge_meta_cache[key] = new_block

    with open(edge_meta_out_path, 'wb') as f:
        pickle.dump(new_edge_meta_cache, f)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python preprocess_extract_edge_meta.py <dataset_dir>')

    raw_edge_cache_name = 'edge_feat_pos_cache.bin'
    output_edge_meta_name = 'edge_rt_cache.bin'

    scene_dir = sys.argv[1]
    scene_dirs = glob.glob(os.path.join(scene_dir, '*'))

    for dir in scene_dirs:
        edge_cache_path = os.path.join(dir, raw_edge_cache_name)

        if 'NYC' not in edge_cache_path:
            continue

        if os.path.isdir(dir) and os.path.exists(edge_cache_path):
            out_edge_meta_path = os.path.join(dir, output_edge_meta_name)
            if os.path.exists(out_edge_meta_path):
                print('Remove :%s' % out_edge_meta_path)
                if os.path.exists(out_edge_meta_path.split('.')[0]):
                    print('Remove :%s' % out_edge_meta_path.split('.')[0])
                    os.remove(out_edge_meta_path.split('.')[0])
                os.remove(out_edge_meta_path)
                # continue

            extract_meta_info(edge_cache_path, out_edge_meta_path)
