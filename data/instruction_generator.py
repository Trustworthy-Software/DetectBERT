import os
import os.path as osp
import multiprocessing as mp

manager = mp.Manager
ClassDictionary = manager().dict()

class Method(object):
    def __init__(self):
        self.name = ''
        self.ClassName = ''
        self.instructions = []
    
    def add_instruction(self, raw_strig: str):
        self.instructions.append(raw_strig.strip())


class SmaliClass(object):
    def __init__(self):
        self.name = ''
        self.methods = []
        self.api_names = []
    
    def add_method(self, method: Method):
        self.methods.append(method)

    def add_api_name(self, api_name: str):
        self.api_names.append(api_name)


def FunctionGenerator(SmaliFile):

    MethodFlag = False

    for line in open(SmaliFile, 'r').readlines():
        if line.startswith('.class'):
            ClassName = line.strip().split(' ')[-1][1:-1]
            if ClassName in ClassDictionary:
                ClassDictionary[ClassName] += 1
                break
            else:
                ClassDictionary[ClassName] = 1
        if line.startswith('.method'):
            MethodFlag = True
            method = Method()
            method.name = line.split(' ')[-1][:-1]
            method.ClassName = ClassName
            continue
        if line.startswith('.end method'):
            MethodFlag = False
            yield method
        if MethodFlag and len(line.strip()) > 0 and not line.strip().startswith('.'):
            method.add_instruction(line)


def SmaliInstructionGenerator(SmaliRootDir, flag='method'):
    '''
    SmaliRootDir: is the disassembled directory from dex files in an APK.
    flag: can only be 'method' or 'class' indicating the generator to yield method or class.
    '''

    assert flag in {'method', 'class'}
    
    SmaliFileList = []
    for root, _, files in os.walk(SmaliRootDir, topdown=False):
        SmaliFileList = SmaliFileList + [osp.join(root, x) for x in files if x.endswith('.smali')]
    
    for SmaliFile in SmaliFileList:
        if flag == 'class':
            Class = SmaliClass() 
        for method in FunctionGenerator(SmaliFile):
            if flag == 'method':
                yield method
            else:
                Class.add_method(method)
        if flag == 'class' and len(Class.methods):
            Class.name = Class.methods[0].ClassName
            yield Class
                

if __name__ == "__main__":

    from numpy.core.fromnumeric import mean
    import numpy as np

    k1_3_cnt = 0
    num_list = []
    method_num_list = []
    root_dir = './dataset/debug/raw_smalis'
    app_list = os.listdir(root_dir)
    for app_hash in app_list:
        clas_cnt = 0
        for cls in SmaliInstructionGenerator(osp.join(root_dir, app_hash), 'class'):
            clas_cnt += 1
            method_num_list.append(len(cls.methods))
        num_list.append(clas_cnt)
        if clas_cnt <= 2000 and clas_cnt >= 1000:
            k1_3_cnt += 1
        # print('class number: ', clas_cnt)

    print("#"*10, 'Class number in each APK', "#"*10)
    print("max number: ", max(num_list))
    print("min number: ", min(num_list))
    print("mean number: ", int(mean(num_list)))
    print("median number: ", int(np.median(np.array(num_list))))
    print('number in (1k, 3k): ', k1_3_cnt)
    print("#"*10, 'Method number in each Class', "#"*10)
    print("max number: ", max(method_num_list))
    print("min number: ", min(method_num_list))
    print("mean number: ", int(mean(method_num_list)))
    print("median number: ", int(np.median(np.array(method_num_list))))