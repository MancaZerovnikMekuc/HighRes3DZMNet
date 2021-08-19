import niftynet
import sys
def start_training(base_dir, gpu_num, cuda_devices, model_dir, config_file):
    model_dir = base_dir + 'models\\' +  model_dir
    config_file_path = '.\\configFiles\\' + config_file
    sys.argv = ['', 'train',
                '-a', 'net_segment',
                '--conf', config_file_path,
                '--model_dir', model_dir,
                '--num_gpus', str(gpu_num),
                '--cuda_devices', str(cuda_devices)]
    niftynet.main()

def start_training_lr_drop(base_dir, gpu_num, cuda_devices, model_dir, config_file, starting_lr, drop, steps, starting_iter=0, split_file=None):
    model_dir = base_dir + 'models\\' +  model_dir
    config_file_path = '.\\configFiles\\' + config_file
    start_iter  = starting_iter
    for i in range(0,len(steps)):
        sys.argv = ['', 'train',
                    '-a', 'net_segment',
                    '--conf', config_file_path,
                    '--model_dir', model_dir,
                    '--num_gpus', str(gpu_num),
                    '--cuda_devices', str(cuda_devices),
                    '--lr', str(starting_lr),
                    '--max_iter', str(steps[i])
                    ]
        if (start_iter != 0):
            sys.argv = sys.argv  + ['--starting_iter', str(start_iter)]
        if (split_file):
            sys.argv = sys.argv + ['--dataset_split_file', split_file]
        niftynet.main()
        starting_lr = starting_lr * drop
        start_iter = steps[i]

    return 0