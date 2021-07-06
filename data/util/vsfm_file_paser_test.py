import numpy as np

def parse_vsfm_feat_match(file_path):

    file = open(file_path, 'r')
    line_count = 0

    #read_pair_idx_state = False
    #read_num_matches_state = False
    #read_num_matches = True
    
    read_file_name = True
    read_pair_idx_state = False
    read_match_state = 0

    cur_pair_idx = [None, None]
    cur_num_matches = 0
    cur_matches = [None, None]

    matches = []

    # Read line-by-line to parsing attribute files
    while 1:
        lines = file.readlines(1000000)
        if not lines:
            break
        for line in lines:

            if not len(line.strip()) or line.startswith('#'):
                line_count += 1
                continue

            tokens = line.split(' ')
            if read_file_name:
                file_name = tokens[-1].strip()
                if cur_pair_idx[0] == None:
                    cur_pair_idx[0] = file_name
                elif cur_pair_idx[1] == None:
                    cur_pair_idx[1] = file_name
                    
                    read_match_number_state = True
                    read_file_name = False
                    
                line_count += 1
                continue
            if read_match_number_state:
                tokens = int(line.strip())
                cur_match_number = tokens
                line_count += 1
                
                read_match_number_state = False
                read_match_state = 0
                
                continue
            if read_match_state < 2:
                cur_match_list = [int(x) for x in line.strip().split()]
                if read_match_state == 0:
                    cur_matches[0] = cur_match_list
                else:
                    cur_matches[1] = cur_match_list
                   
                    cur_match_dict = dict()
                    cur_match_dict['ids_name'] = cur_pair_idx
                    cur_match_dict['match_number'] = tokens
                    cur_match_dict['n1_feat_id'] = cur_matches[0]
                    cur_match_dict['n2_feat_id'] = cur_matches[1]
                    
                    matches.append(cur_match_dict)
                    
                    read_match_state = 0
                    read_file_name = True
                line_count += 1
                continue

    # Close  file
    file.close()

    return matches

if __name__ == '__main__':
    m = parse_vsfm_matches('/Users/luwei/Desktop/f_mat_1_20.txt')
    print(len(m))