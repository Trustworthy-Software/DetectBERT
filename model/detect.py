import os
import torch
from tqdm import tqdm
import pickle
import os.path as osp
from datetime import datetime

from DetectBERT import DetectBERT
from utils import read_yaml, get_device

def detect_apks(model_weights, apk_list_dict, root_dir, output_file):
    # Load configuration
    cfg = read_yaml('./config.yaml')
    
    # Set up device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(cfg.Train.device)
    device = get_device()
    
    # Initialize model
    print("Initializing DetectBERT model...")
    classifier = DetectBERT(cfg=cfg, n_classes=cfg.Model.catg_num, 
                          input_size=cfg.Model.input_len, 
                          hidden_size=cfg.Model.hidden_len)
    classifier.load_state_dict(torch.load(model_weights, map_location=device), strict=True)
    classifier = classifier.to(device)
    classifier.eval()
    
    print("Starting detection...")
    results = []
    stats = {'total': 0, 'malware': 0, 'benign': 0}
    folder_statistics = {}  # Store statistics for each folder
    
    with torch.no_grad():
        # Process each folder's APKs
        for folder_name, apks in apk_list_dict.items():
            print(f"\nProcessing APKs from {folder_name}")
            folder_stats = {'total': 0, 'malware': 0, 'benign': 0, 'errors': 0}
            
            for apk in tqdm(apks):
                apk = apk.strip()
                if not apk:  # Skip empty lines
                    continue
                    
                if apk.endswith('.apk'):
                    apk_name = apk.split('.')[0]
                else:
                    apk_name = apk
                
                # Load embedding
                emb_path = osp.join(root_dir, folder_name, apk_name + '.pkl')
                if not os.path.exists(emb_path):
                    print(f"Warning: Embedding not found for {apk} in {folder_name}")
                    folder_stats['errors'] += 1
                    continue
                    
                try:
                    with open(emb_path, 'rb') as f:
                        apk_emb = pickle.load(f)
                    
                    # Prepare input
                    input_embs = torch.from_numpy(apk_emb).unsqueeze(0).to(device)
                    
                    # Get prediction
                    output = classifier(data=input_embs)
                    pred = output['Y_hat'].item()
                    prob = output['Y_prob'].cpu().numpy()[0]
                    
                    # Update statistics
                    folder_stats['total'] += 1
                    stats['total'] += 1
                    if pred == 1:
                        folder_stats['malware'] += 1
                        stats['malware'] += 1
                    else:
                        folder_stats['benign'] += 1
                        stats['benign'] += 1
                    
                    # Format result with confidence score and folder name
                    confidence = prob[pred] * 100
                    classification = 'malware' if pred == 1 else 'benign'
                    result = {
                        'apk': apk,
                        'folder': folder_name,
                        'classification': classification,
                        'confidence': confidence,
                        'prob_malware': prob[1] * 100,
                        'prob_benign': prob[0] * 100
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing {apk}: {str(e)}")
                    folder_stats['errors'] += 1
                    continue
            
            # Store folder statistics
            folder_statistics[folder_name] = folder_stats
            
            # Print folder statistics
            print(f"\nFolder {folder_name} Statistics:")
            print(f"Total APKs processed: {folder_stats['total']}")
            if folder_stats['total'] > 0:
                print(f"Malware detected: {folder_stats['malware']} ({folder_stats['malware']/folder_stats['total']*100:.1f}%)")
                print(f"Benign apps: {folder_stats['benign']} ({folder_stats['benign']/folder_stats['total']*100:.1f}%)")
            else:
                print("No APKs were successfully processed in this folder")
            if folder_stats['errors'] > 0:
                print(f"Failed to process: {folder_stats['errors']} APKs")
    
    # Save results
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        # Write header with model info
        f.write("="*100 + "\n")
        f.write("DetectBERT Malware Detection Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_weights}\n")
        f.write("="*100 + "\n\n")
        
        # Write per-folder statistics
        f.write("Per-Folder Statistics:\n")
        f.write("-"*50 + "\n")
        for folder_name, folder_stats in folder_statistics.items():
            f.write(f"\n{folder_name}:\n")
            f.write(f"  Total APKs processed: {folder_stats['total']}\n")
            if folder_stats['total'] > 0:
                f.write(f"  Malware detected: {folder_stats['malware']} ({folder_stats['malware']/folder_stats['total']*100:.1f}%)\n")
                f.write(f"  Benign apps: {folder_stats['benign']} ({folder_stats['benign']/folder_stats['total']*100:.1f}%)\n")
            if folder_stats['errors'] > 0:
                f.write(f"  Failed to process: {folder_stats['errors']} APKs\n")
        f.write("\n" + "="*100 + "\n\n")
        
        # Write overall statistics
        f.write("Overall Statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total APKs processed: {stats['total']}\n")
        if stats['total'] > 0:
            f.write(f"Malware detected: {stats['malware']} ({stats['malware']/stats['total']*100:.1f}%)\n")
            f.write(f"Benign apps: {stats['benign']} ({stats['benign']/stats['total']*100:.1f}%)\n")
        f.write("\n" + "="*100 + "\n\n")
        
        # Write detailed results
        f.write("Detailed Results:\n")
        f.write("-"*120 + "\n")
        f.write(f"{'APK Name':<50} {'Folder':<15} {'Classification':<12} {'Confidence':<10} {'Malware Prob':<12} {'Benign Prob':<12}\n")
        f.write("-"*120 + "\n")
        
        if results:
            for result in sorted(results, key=lambda x: (x['folder'], x['classification'], -x['confidence'])):
                f.write(f"{result['apk']:<50} {result['folder']:<15} {result['classification']:<12} "
                       f"{result['confidence']:>6.1f}%   {result['prob_malware']:>8.1f}%   {result['prob_benign']:>8.1f}%\n")
        else:
            f.write("No APKs were successfully processed.\n")
    
    print("\nDetection completed! Summary:")
    print(f"Total APKs processed: {stats['total']}")
    if stats['total'] > 0:
        print(f"Malware detected: {stats['malware']} ({stats['malware']/stats['total']*100:.1f}%)")
        print(f"Benign apps: {stats['benign']} ({stats['benign']/stats['total']*100:.1f}%)")
    print(f"Results have been saved to: {output_file}")

if __name__ == "__main__":
    # Model weights path
    model_weights = "DetectBERT/save/split_1/model_steps_2500001.pt"
    
    # APK list from previous DexBERT code
    root_dir = 'DATA/Africa_APKs'
    src_data_list = [['Infinix_apk.txt', 'Infinix_apk'], 
                     ['Tecno_apk.txt', 'Tecno_apk'], 
                     ['itel_apk.txt', 'itel_apk']]
    
    # Get all APKs organized by folder
    apk_list_dict = {}
    for src_path, folder_name in src_data_list:
        with open(os.path.join(root_dir, src_path), 'r') as f:
            apk_list_dict[folder_name] = f.readlines()
    
    # Output file
    output_file = "detection_results.txt"
    
    # Run detection
    detect_apks(model_weights, apk_list_dict, root_dir, output_file) 