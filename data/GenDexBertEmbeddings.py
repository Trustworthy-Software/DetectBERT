import os
import os.path as osp
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import DownloadApk, Disassemble, get_device
from SmaliPreprocess import Smalis2Txt

import tokenization
from models import DexBERT, Config
from dataloader import PreprocessEmbedding


class SmaliSeqDataset(Dataset):
    def __init__(self, file, tokenize, max_len, pipeline=[]):
        super().__init__()
        self.file = open(file, "r", encoding='utf-8', errors='ignore') 
        self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.pipeline = pipeline
        self.current_class_id = 0
        
        self.instance_list = self.instance_generator()

    def read_tokens(self, f, length, discard_last_and_restart=False, keep_method_name=True):
        """ Read tokens from file pointer with limited length """
        tokens   = []
        ClassEnd = False
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None, ClassEnd
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens, ClassEnd # return last tokens in the document
            if line.strip().startswith('ClassName:'):
                continue  # skip the smali class name
            if line.strip().startswith('MethodName:') and not keep_method_name:
                continue # skip the smali method name
            if line.strip().startswith('ClassEnd'):
                ClassEnd = True
                return tokens, ClassEnd
            tokens.extend(self.tokenize(line.strip()))
        return tokens, ClassEnd

    def instance_generator(self): # iterator to load data
        instance_list = []
        close_file = False
        while True and not close_file:
            len_tokens = self.max_len

            tokens, ClassEnd = self.read_tokens(self.file, len_tokens, discard_last_and_restart=False, keep_method_name=True)
            
            if ClassEnd:  # end of current class -> end of current batch
                self.current_class_id += 1
            
            if tokens is None:  # end of file
                self.file.close()
                close_file = True
                break
            if len(tokens) == 0:
                continue 

            class_id = self.current_class_id
            instance = (tokens, class_id)
            for proc in self.pipeline:
                instance = proc(instance)
            
            instance_list.append(instance)
        return instance_list
    
    def __len__(self):
        return len(self.instance_list)
    
    def __getitem__(self, index):
        input_ids, segment_ids, input_mask, class_id = self.instance_list[index]
        return np.array(input_ids), np.array(segment_ids), np.array(input_mask), np.array(class_id)


def BertInfer(BertAEmodel, dataloader, device):
    print(f"Starting inference on {device}...")
    class_vector_list = []
    last_class_id  = 0

    seq_iter_bar = tqdm(dataloader, desc="Processing batches")
    with torch.no_grad():
        for batch_idx, batch in enumerate(seq_iter_bar):
            try:
                # Move batch to device
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, class_id = batch

                # Process batch
                r2 = BertAEmodel(input_ids, segment_ids, input_mask)
                
                # Move results to CPU for numpy operations
                if device.type == "cuda":
                    r2 = r2.cpu()
                    class_id = class_id.cpu()
                
                batch_vec = r2.detach().numpy()
                
                # Update progress
                seq_iter_bar.set_postfix({
                    "Batch": f"{batch_idx+1}/{len(dataloader)}",
                    "GPU Mem": f"{torch.cuda.max_memory_allocated()//1024//1024}MB" if torch.cuda.is_available() else "N/A"
                })
                
                for i, emb in enumerate(batch_vec):
                    if len(class_vector_list) == 0:
                        class_vector_list.append(np.expand_dims(emb, axis=0))
                        continue
                    if int(class_id[i]) == last_class_id:
                        class_vector_list[-1] = np.concatenate([class_vector_list[-1], np.expand_dims(emb, axis=0)])
                        continue
                    class_vector_list.append(np.expand_dims(emb, axis=0))
                    last_class_id = int(class_id[i])

                # Clear CUDA cache periodically
                if device.type == "cuda" and batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA out of memory in batch {batch_idx}. Clearing cache and trying again...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"\nError processing batch {batch_idx}: {str(e)}")
                    continue
            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {str(e)}")
                continue
    
    print(f"Inference completed. Processed {len(class_vector_list)} classes.")
    return class_vector_list

def Hash2ApkEmb(hash, tmp_dir, save_dir, BertAE, batch_size, pipeline):
    print(f"Starting Hash2ApkEmb with hash={hash}")
    if not hash.endswith('.apk'):
        print("Hash doesn't end with .apk - downloading APK")
        apk_path = osp.join(tmp_dir, hash.upper()+'.apk')
        DownloadApk(apk_path)
        smali_dir = osp.join(tmp_dir, hash)
    else:
        print("Hash ends with .apk - copying from save_dir")
        apk_path = osp.join(tmp_dir, hash)
        os.system('cp {} {}'.format(osp.join(save_dir, hash), apk_path))
        smali_dir = osp.join(tmp_dir, hash.split('.')[0])
    print(f"Disassembling APK at {apk_path} to {smali_dir}")
    Disassemble(apk_path, smali_dir)
    print("Converting smali to txt")
    Smalis2Txt(tmp_dir, smali_dir, only_keep_func_name=False)
    ApkName = smali_dir.split('/')[-1] if smali_dir.split('/')[-1] else smali_dir.split('/')[-2]
    txt_file = osp.join(tmp_dir, ApkName+'.txt')
    print(f"Created txt file at {txt_file}")

    print("Creating dataset and dataloader")
    dataset = SmaliSeqDataset(txt_file, tokenize, Bert_model_cfg.max_len, pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("Running BertInfer")
    class_vec_list = BertInfer(BertAE, dataloader, device)
    class_vec_list = np.vstack(class_vec_list)

    print("Saving results")
    with open(osp.join(save_dir, ApkName+'.pkl'), 'wb') as f:
        pickle.dump(class_vec_list, f)
    print("Cleaning up temporary files")
    os.system('rm -r {}'.format(osp.join(tmp_dir, '*')))
    # import ipdb; ipdb.set_trace();

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'  # Use GPU 0
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        # Set CUDA memory usage
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    root_dir = './DATA/Africa_APKs'
    # src_data_list = [['goodware_hashes.txt', 'data/goodware']]
    src_data_list = [['Infinix_apk.txt', 'Infinix_apk'], ['Tecno_apk.txt', 'Tecno_apk'], ['itel_apk.txt', 'itel_apk']]
    # src_data_list = [['malware_hashes.txt', 'data/malware']]
    # src_data_list = [['src/goodware_hashes_1_1.txt', 'data/goodware_1_1']]
    # src_data_list = [['src/goodware_hashes_1_2.txt', 'data/goodware_1_2']]
    # src_data_list = [['src/goodware_hashes_2_1.txt', 'data/goodware_2_1']]
    # src_data_list = [['src/goodware_hashes_2_2.txt', 'data/goodware_2_2']]
    Bert_model_cfg = './bert_base.json'
    vocab = './vocab.txt'
    DexBERT_file = './model_steps_604364.pt'

    # model initialization
    batch_size = 32
    Bert_model_cfg = Config.from_json(Bert_model_cfg)
    BertAE = DexBERT(Bert_model_cfg)
    print(f"Loading model on {device}...")
    BertAE.load_state_dict(torch.load(DexBERT_file, map_location=device), strict=False)
    BertAE.to(device)
    BertAE.eval()
    
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))  

    pipeline = [PreprocessEmbedding(tokenizer.convert_tokens_to_ids)]

    for pair in src_data_list:
        src_path, data_dir = pair[0], pair[1]
        hash_list = open(osp.join(root_dir, src_path), 'r').readlines()
        save_dir = osp.join(root_dir, data_dir)
        tmp_dir = osp.join(save_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        
        for hash in tqdm(hash_list): 
            hash = hash.strip()
            print(f"\nProcessing hash: {hash}")
            if os.path.exists(os.path.join(save_dir, hash.upper()+'.pkl')):
                print(f"Skipping {hash} - already processed")
                continue
            try:
                print(f"Starting Hash2ApkEmb for {hash}")
                Hash2ApkEmb(hash, tmp_dir, save_dir, BertAE, batch_size, pipeline)
            except Exception as e:
                print(f"Error processing {hash}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        # import ipdb; ipdb.set_trace();
    os.system('rm -r {}'.format(tmp_dir))
            