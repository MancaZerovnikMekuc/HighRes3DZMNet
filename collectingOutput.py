import os
import shutil
def collect(base_dir, final_dir, model_dir):
    model_dir = base_dir + 'models\\' +  model_dir + '\\output\\'
    final_dir = base_dir + 'models\\' + final_dir + '\\'
    try:
        os.mkdir(final_dir)
    except:
        print("directory already exists.....writing in existing one")
    final_dir = final_dir + "output\\"
    try:
        os.mkdir(final_dir)
    except:
        print("output directory also exists....writing in existing one")
    for file in os.listdir(model_dir):
        if file.endswith(".nii.gz"):
            print(os.path.join("/mydir", file))
            os.rename(model_dir + file, final_dir + file)
    return 0
