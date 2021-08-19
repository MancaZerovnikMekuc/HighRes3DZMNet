import niftynet
import sys
def start_inference(base_dir, gpu_num, cuda_devices, model_dir, config_file, split_file=None, inference_iter=None):
    model_dir = base_dir + 'models\\' +  model_dir
    config_file_path = '.\\configFiles\\' + config_file
    sys.argv = ['', 'inference',
                '-a', 'net_segment',
                '--conf', config_file_path,
                '--model_dir', model_dir,
                '--num_gpus', str(gpu_num),
                '--cuda_devices', str(cuda_devices)]
    if (split_file):
        sys.argv = sys.argv + ['--dataset_split_file', split_file]
    if (inference_iter):
        sys.argv = sys.argv + ['--inference_iter', inference_iter]
    niftynet.main()
