import os
import os.path as osp
import sys
import torch

API_key = ""  # please obtain your personal key on https://androzoo.uni.lu/

def DownloadApk(ApkFile):
    '''
    To download ApkFile that doesn't exist.

    :param String ApkFile: absolute path of the ApkFile
    '''

    if osp.exists(ApkFile):
        pass
    else:
        SaveDir, ApkName = osp.dirname(ApkFile), osp.basename(ApkFile)
        Hash = ApkName.split('.')[0]
        os.system("cd {} && curl -O --remote-header-name -G -d apikey={} -d sha256={} https://androzoo.uni.lu/api/download > /dev/null".format(
            SaveDir, API_key, Hash))

def Disassemble(ApkPath, OutDir):
    '''
    To disassemble Dex bytecode in a given Apk file into smali code.
    Java version: "11.0.11" 2021-04-20
    The baksmali tool baksmali-2.5.2.jar was downloaded on: https://bitbucket.org/JesusFreke/smali/downloads/
    '''
    os.system("java -jar {} disassemble {} -o {}".format(osp.join(sys.path[0], 'baksmali-2.5.2.jar'), ApkPath, OutDir)) 

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def truncate_tokens(tokens, max_len):
    while True:
        if len(tokens) <= max_len:
            break
        else:
            tokens.pop()