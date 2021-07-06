import os, sys, pickle, json, argparse,shutil
sys.path.append("/mnt/Tango/pg/pg_akt_rot_avg")

def get_args(DEBUG):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="/mnt/Tango/pg/pg_akt_rot_avg/train_config/yfcc_2_80nodes_dump_feat.json")
    parser.add_argument("--dev", default=0, type=int)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--clean_cache", action="store_true")
    args = None
    if DEBUG:
        args = parser.parse_args(["--json","/mnt/Tango/pg/pg_akt_rot_avg/train_config/yfcc_2_80nodes_dump_feat.json",
        "--dev","0","--output_path","/mnt/Exp_5/yfcc_init_opt_cache"]) 
    else:
        args = parser.parse_args()
    return args

def make_json_config_dict(dataset:str, data_name:str, args):
    json_config_dict = dict()
    json_config_dict["clean_cache"] = args.clean_cache
    json_config_dict["valid"] = dict()
    json_config_dict["valid"][dataset] = dict()
    json_config_dict["valid"][dataset]["dataset_list"] = [data_name]
    json_config_dict["valid"][dataset]["node_edge_lmdb_path"] = "{:s}_{:s}_lmdb.bin".format(dataset, data_name)
    json_config_dict["valid"][dataset]["node_edge_lmdb_name"] = "{:s}_{:s}_lmdb.lmdb".format(dataset, data_name)
    json_config_dict["valid"][dataset]["img_max_dim"] = 400
    json_config_dict["valid"][dataset]["sampling_num_range"] = [80,100]
    json_config_dict["valid"][dataset]["sub_graph_nodes"] = 80
    json_config_dict["valid"][dataset]["sample_res_cache"] = "{:s}_{:s}_sampling_cache.bin".format(dataset,data_name)
    json_config_dict["valid"][dataset]["sampling_undefine_edge"] = False
    json_config_dict["valid"][dataset]["load_img"] = True
    json_config_dict["valid"][dataset]["load_keypt_match"] = True
    json_config_dict["valid"][dataset]["load_node_edge_feat"] = False
    json_config_dict["valid"][dataset]["subgraph_edge_cover_ratio"] = 0.2
    return json_config_dict

if __name__ == "__main__":
    DEBUG=True
    args = get_args(DEBUG)
    
    json_dict = dict()
    with open(args.json,"r") as fin:
        json_dict = json.load(fin)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    """ Generating json file for train datasets
    """
    if "train" in json_dict:
        print("Generating train datasets json file")
        training_json_output_path = os.path.join(args.output_path,"train_json_config")
        if os.path.exists(training_json_output_path):
            shutil.rmtree(training_json_output_path)
        if not os.path.exists(training_json_output_path):
            os.makedirs(training_json_output_path)
        datasets = json_dict["train"]
        for dataset, dataset_dict in datasets.items():
            for data in dataset_dict["dataset_list"]:
                json_file_name = "{:s}_{:s}.json".format(dataset, data)
                print("----Generating {:s}".format(json_file_name))
                json_config_dict = make_json_config_dict(dataset, data, args)
                with open(os.path.join(training_json_output_path, json_file_name), "w") as fout:
                    json.dump(json_config_dict, fout)
    else:
        print("No train datasets")

    """ Generating json file for valid datasets
    """
    if "valid" in json_dict:
        print("Generating valid datasets json file")
        training_json_output_path = os.path.join(args.output_path,"valid_json_config")
        if os.path.exists(training_json_output_path):
            shutil.rmtree(training_json_output_path)
        if not os.path.exists(training_json_output_path):
            os.makedirs(training_json_output_path)
        datasets = json_dict["valid"]
        for dataset, dataset_dict in datasets.items():
            for data in dataset_dict["dataset_list"]:
                json_file_name = "{:s}_{:s}.json".format(dataset, data)
                print("----Generating {:s}".format(json_file_name))
                json_config_dict = make_json_config_dict(dataset, data, args)
                with open(os.path.join(training_json_output_path, json_file_name), "w") as fout:
                    json.dump(json_config_dict, fout)
    else:
        print("No valid datasets")
    