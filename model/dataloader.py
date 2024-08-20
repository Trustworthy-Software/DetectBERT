import os
import pickle

from torch.utils.data import Dataset


class ApkEmbDataset(Dataset):
    def __init__(self, root_dir, samp_list):
        self.emb_dir = root_dir
        self.hash_list = []
        for line in open(samp_list, 'r').readlines():
            hash = line.strip().split('.')[0]
            label = 1 if hash.startswith('malware') else 0
            emb_path = os.path.join(self.emb_dir, hash+'.pkl')
            emb_path = emb_path.replace('/goodware/', '/goodware_1_1/')
            if not os.path.exists(emb_path):
                emb_path = emb_path.replace('/goodware_1_1/', '/goodware_1_2/')
            if not os.path.exists(emb_path):
                emb_path = emb_path.replace('/goodware_1_2/', '/goodware_2_1/')
            if not os.path.exists(emb_path):
                emb_path = emb_path.replace('/goodware_2_1/', '/goodware_2_2/')
            if os.path.exists(emb_path):
                self.hash_list.append([emb_path, label])
    
    def __len__(self):
        return len(self.hash_list)
    
    def __getitem__(self, index):
        emb_path, label = self.hash_list[index]
        apk_emb = pickle.load(open(emb_path, 'rb'))

        return apk_emb, label