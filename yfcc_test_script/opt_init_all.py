import os, sys, pickle, json, argparse,shutil
sys.path.append("/mnt/Tango/pg/pg_akt_rot_avg")

def get_args(DEBUG=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/Exp_5/yfcc_init_opt_cache/train_json_config")
    parser.add_argument("--output", default="/mnt/Exp_5/yfcc_init_opt_cache/cache_dir")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    oswalk = os.walk(args.root)
    for root, dirs, files in oswalk:
        for file in files:
            if os.path.splitext(file)[-1] == ".json":
                output_file_name = os.path.splitext(file)[0]+ ".bin"
                if os.path.exists(os.path.join(args.output, output_file_name)):
                    continue
                with open(os.path.join(root, file), "rb") as fin:
                    config_dict = json.load(fin)
                datasets = config_dict["valid"]
                for dataset, data_dict in datasets.items():
                    ds_name = data_dict["dataset_list"][0]
                command = "python /mnt/Tango/pg/pg_akt_rot_avg/dbg/dbg_spt_supergraph_mpropagate_24s_yfcc.py "
                command += " --config {:s}".format(os.path.join(root, file))
                command += " --dev 0 "
                command += " --dump_init "
                command += " --output_file {:s} ".format(os.path.join(args.output, output_file_name))
                command += " --name {:s} ".format(ds_name)
                print(command)
                os.system(command)