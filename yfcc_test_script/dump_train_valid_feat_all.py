import os, sys, pickle, json, argparse,shutil
sys.path.append("/mnt/Tango/pg/pg_akt_rot_avg")

def get_args(DEBUG=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/Exp_5/yfcc_init_opt_cache/train_json_config")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    oswalk = os.walk(args.root)
    for root, dirs, files in oswalk:
        for file in files:
            if os.path.splitext(file)[-1] == ".json":
                command = "python /mnt/Tango/pg/pg_akt_rot_avg/data/cache_yfcc_test_node_edge_feat.py "
                command += " --config_file {:s}".format(os.path.join(root, file))
                command += " --dev 1"
                os.system(command)