import numpy as np
import os
<<<<<<< HEAD
def parse_vsfm_matches(file_path):
=======

>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877

def parse_vsfm_matches(file_path):
    file = open(file_path, 'r')
    line_count = 0

    read_pair_idx_state = False
    read_num_matches_state = False
    read_num_matches = True

    cur_pair_idx = [None, None]
    cur_num_matches = 0
    cur_matches = None
    cur_match_itr = 0

    matches = []

    # Read line-by-line to parsing attribute files
    while 1:
        lines = file.readlines(100000)
        if not lines:
            break
        for line in lines:

            if line.startswith('#'):
                line_count += 1
                continue

            tokens = line.split(' ')
            if tokens[0].strip().isdigit() and read_num_matches is True and read_pair_idx_state is False:
                file_name = tokens[-1].strip()
<<<<<<< HEAD
                file_name = file_name.replace("\\","/")
=======
                file_name = file_name.replace("\\", "/")
>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877
                file_name = os.path.splitext(os.path.split(file_name)[-1])[0]
                if cur_pair_idx[0] == None:
                    cur_pair_idx[0] = (int(tokens[0]), file_name)
                elif cur_pair_idx[1] == None:
                    cur_pair_idx[1] = (int(tokens[0]), file_name)
                    read_pair_idx_state = True
                    line_count += 1
                    continue
            if read_pair_idx_state is True and read_num_matches_state is False:
                if tokens[0].strip().isdigit():
                    cur_num_matches = int(tokens[0].strip())
                    read_num_matches_state = True
                    cur_pair_dict = dict()
                    cur_pair_dict['ids'] = (cur_pair_idx[0][0], cur_pair_idx[1][0])
                    cur_pair_dict['ids_name'] = (cur_pair_idx[0][1], cur_pair_idx[1][1])

                    cur_pair_dict['n1_feat_id'] = list()
                    cur_pair_dict['n2_feat_id'] = list()
                    cur_pair_dict['n1_feat_pos'] = list()
                    cur_pair_dict['n2_feat_pos'] = list()

                    matches.append(cur_pair_dict)

                    line_count += 1
                    continue

            if read_num_matches_state is True and read_num_matches_state is True:
                n1_id = int(tokens[0].strip())
                n1_x = float(tokens[1].strip())
                n1_y = float(tokens[2].strip())
                n2_id = int(tokens[3].strip())
                n2_x = float(tokens[4].strip())
                n2_y = float(tokens[5].strip())

                cur_pair_dict['n1_feat_id'].append(n1_id)
                cur_pair_dict['n2_feat_id'].append(n2_id)
                cur_pair_dict['n1_feat_pos'].append((n1_x, n1_y))
                cur_pair_dict['n2_feat_pos'].append((n2_x, n2_y))

                cur_match_itr += 1

                if cur_match_itr >= cur_num_matches:
                    cur_pair_idx = [None, None]
                    cur_num_matches = 0
                    cur_match_itr = 0

                    read_num_matches_state = False
                    read_pair_idx_state = False
                    read_num_matches = True

            line_count += 1

    # Close  file
    file.close()

    return matches


def parse_vsfm_feat_match(file_path):
<<<<<<< HEAD

    file = open(file_path, 'r')
    line_count = 0

    #read_pair_idx_state = False
    #read_num_matches_state = False
    #read_num_matches = True
    
=======
    file = open(file_path, 'r')
    line_count = 0

    # read_pair_idx_state = False
    # read_num_matches_state = False
    # read_num_matches = True

>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877
    read_file_name = True
    read_pair_idx_state = False
    read_match_state = 0

    cur_pair_idx = [None, None]
    cur_num_matches = 0
    cur_matches = [None, None]

    matches = []

    # Read line-by-line to parsing attribute files
    while 1:
        lines = file.readlines()
        if not lines:
            break
        for line in lines:
            if not len(line.strip()) or line.startswith('#'):
                line_count += 1
                continue
            tokens = line
            if read_file_name:
                file_name = tokens.strip()
<<<<<<< HEAD
                file_name = file_name.replace("\\","/")
=======
                file_name = file_name.replace("\\", "/")
>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877
                file_name = os.path.splitext(os.path.split(file_name)[-1])[0]
                if cur_pair_idx[0] == None:
                    cur_pair_idx[0] = file_name
                elif cur_pair_idx[1] == None:
                    cur_pair_idx[1] = file_name
<<<<<<< HEAD
                    
                    read_match_number_state = True
                    read_file_name = False
                    
=======

                    read_match_number_state = True
                    read_file_name = False

>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877
                line_count += 1
                continue
            if read_match_number_state:
                tokens = int(line.strip())
                cur_match_number = tokens
                line_count += 1
<<<<<<< HEAD
                
                read_match_number_state = False
                read_match_state = 0
                
                continue
            if read_match_state <=1:
=======

                read_match_number_state = False
                read_match_state = 0

                continue
            if read_match_state <= 1:
>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877
                cur_match_list = [int(x) for x in line.strip().split()]
                if read_match_state == 0:
                    cur_matches[0] = cur_match_list
                    read_match_state += 1
                else:
                    cur_matches[1] = cur_match_list
<<<<<<< HEAD
                   
=======

>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877
                    cur_match_dict = dict()
                    cur_match_dict['ids_name'] = cur_pair_idx
                    cur_match_dict['match_number'] = cur_match_number
                    cur_match_dict['n1_feat_id'] = cur_matches[0]
                    cur_match_dict['n2_feat_id'] = cur_matches[1]
<<<<<<< HEAD
                    
                    cur_pair_idx = [None, None]
                    
                    matches.append(cur_match_dict)
                    
=======

                    cur_pair_idx = [None, None]

                    matches.append(cur_match_dict)

>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877
                    read_match_state = 0
                    read_file_name = True
                line_count += 1
                continue

    # Close  file
    file.close()

    return matches

<<<<<<< HEAD
=======

>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877
if __name__ == '__main__':
    m = parse_vsfm_feat_match('/mnt/Tango/pg/ICCV15_alledges/Gendarmenmarkt/featmatch_1_20.txt')
    for match in m[:10]:
        print(match['ids_name'])
        print(match['n1_feat_id'])
<<<<<<< HEAD
    print(len(m))
=======
    print(len(m))
>>>>>>> b30758e8d434b84660f29bd98ed4594c6e131877
