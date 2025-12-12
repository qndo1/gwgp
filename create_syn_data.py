"""
Build databases
"""

from preprocess.DataIO import build_dict_mimic3, build_dict_mc3, build_dict_ppi, build_dict_syn
import dev.util as util
import pickle

# make file directory data\train if not exist
util.makedirs(util.DATA_TRAIN_DIR)

# syn
nl = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
for n_trail in range(30):
    for num in [10, 25, 50, 100]:
        for i in range(len(nl)):
            data_syn = build_dict_syn(num_node=num,
                                      noise=nl[i])
            filename = '{}/syn_{}_{}_{}_database.pickle'.format(util.DATA_TRAIN_DIR, n_trail, num, i)
            with open(filename, 'wb') as f:
                pickle.dump(data_syn, f)
            print(len(data_syn['src_interactions']))
            print(len(data_syn['tar_interactions']))