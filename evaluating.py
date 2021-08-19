import os
import sys
import niftynet
import configparser

def start_evaluating(renameFibLGM, renameFibLGMFV, renameFibLGMVal, renameAll,  model_dir, config_file):
    output_dir = '.\\models\\' + model_dir + '\\output\\'

    config_file_path = '.\\configFiles\\EVAL\\' + config_file
    if(renameFibLGM):
        #first remove existing renamed volumes
        filelist = os.listdir(output_dir)
        files = [i for i in filelist if i.startswith("fib1-")]
        for f in files:
            os.remove(os.path.join(output_dir, f))
        #rename output
        old_file = os.path.join(output_dir, "fib1000_niftynet_out.nii.gz")
        new_file = os.path.join(output_dir, "fib1-0-0-0.nii.gz")
        os.rename(old_file, new_file)
        old_file = os.path.join(output_dir, "fib1321_niftynet_out.nii.gz")
        new_file = os.path.join(output_dir, "fib1-3-2-1.nii.gz")
        os.rename(old_file, new_file)
    if(renameAll):
        #first remove existing renamed volumes
        filelist = os.listdir(output_dir)
        files = [i for i in filelist if i.startswith("fib1-")]
        for f in files:
            os.remove(os.path.join(output_dir, f))
        #rename output
        old_file = os.path.join(output_dir, "fib1000_niftynet_out.nii.gz")
        new_file = os.path.join(output_dir, "fib1-0-0-0.nii.gz")
        os.rename(old_file, new_file)

        old_file = os.path.join(output_dir, "fib1321_niftynet_out.nii.gz")
        new_file = os.path.join(output_dir, "fib1-3-2-1.nii.gz")
        os.rename(old_file, new_file)

        old_file = os.path.join(output_dir, "fib1103_niftynet_out.nii.gz")
        new_file = os.path.join(output_dir, "fib1-1-0-3.nii.gz")
        os.rename(old_file, new_file)

        old_file = os.path.join(output_dir, "fib1430_niftynet_out.nii.gz")
        new_file = os.path.join(output_dir, "fib1-4-3-0.nii.gz")
        os.rename(old_file, new_file)

        old_file = os.path.join(output_dir, "fib1330_niftynet_out.nii.gz")
        new_file = os.path.join(output_dir, "fib1-3-3-0.nii.gz")
        os.rename(old_file, new_file)
    if(renameFibLGMFV):
        #first remove existing renamed volumes
        filelist = os.listdir(output_dir)
        files = [i for i in filelist if i.startswith("fib1-")]
        for f in files:
            os.remove(os.path.join(output_dir, f))
        #rename output
        old_file = os.path.join(output_dir, "fib1321_niftynet_out.nii.gz")
        new_file = os.path.join(output_dir, "fib1-3-2-1.nii.gz")
        os.rename(old_file, new_file)
    if(renameFibLGMVal):
        #first remove existing renamed volumes
        filelist = os.listdir(output_dir)
        files = [i for i in filelist if i.startswith("fib1-")]
        for f in files:
            os.remove(os.path.join(output_dir, f))
        #rename output
        old_file = os.path.join(output_dir, "fib1430_niftynet_out.nii.gz")
        new_file = os.path.join(output_dir, "fib1-4-3-0.nii.gz")
        os.rename(old_file, new_file)

    parser = configparser.SafeConfigParser()
    parser.read(config_file_path)
    parser.set('seg', 'path_to_search', './models/' + model_dir + '/output' )
    with open(config_file_path, 'w') as configfile:
        parser.write(configfile)
    model_dir = '.\\models\\' + model_dir
    sys.argv=['', 'evaluation','-a','net_segment',
              '--conf',config_file_path,
              '--model_dir', model_dir]
    niftynet.main()
