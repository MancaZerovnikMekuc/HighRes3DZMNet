from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import shutil

def connect_to_drive():
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("credentials.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("credentials.txt")
    drive = GoogleDrive(gauth)
    return drive
def print_root_files_ids(drive):
    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file1 in file_list:
        print('title: %s, id: %s' % (file1['title'], file1['id']))
def print_subfolder_files_id(drive, fid):
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false" % fid}).GetList()
    for file1 in file_list:
        print('title: %s, id: %s' % (file1['title'], file1['id']))
def upload_to_folder(fid, file_path, file_name, drive):
    f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}], 'title': file_name})
    f.SetContentFile(file_path)
    f.Upload()
    print('Created file %s with mimeType %s' % (f['title'], f['mimeType']))
def create_subfolder(drive, parent_folder_id, name):
    child_folder = drive.CreateFile({'title': name, 'parents': [{'id': parent_folder_id}], 'mimeType' : 'application/vnd.google-apps.folder'})
    child_folder.Upload()
    fid = child_folder['id']
    return fid
def upload_and_clear_models(drive, fid, selected_model, clear_models, model_dir):
    model_fid = create_subfolder(drive, fid, "Model")
    file1_name = "model.ckpt-" + str(selected_model) + ".index"
    file1_path = model_dir + "\\models\\" + file1_name
    file2_name = "model.ckpt-" + str(selected_model) + ".meta"
    file2_path = model_dir + "\\models\\" + file2_name
    file3_name = "model.ckpt-" + str(selected_model) + ".data-00000-of-00001"
    file3_path = model_dir + "\\models\\" + file3_name
    upload_to_folder(model_fid, file1_path, file1_name, drive)
    upload_to_folder(model_fid, file2_path, file2_name, drive)
    upload_to_folder(model_fid, file3_path, file3_name, drive)
    if (clear_models):
        models_dir = model_dir + "\\models"
        filelist = os.listdir(models_dir)
        for f in filelist:
            os.remove(os.path.join(models_dir, f))
    return
def upload_and_clear_eval(drive, fid, clear_eval, model_dir):
    eval_fid = create_subfolder(drive, fid, "Eval")
    eval_dir = model_dir + "\\evaluation"
    upload_to_folder(eval_fid, eval_dir + "\\eval_label.csv", "eval_label.csv", drive)
    upload_to_folder(eval_fid, eval_dir + "\\eval_subject_id_label.csv", "eval_subject_id_label.csv", drive)
    if(clear_eval):
        os.remove(eval_dir + "\\eval_subject_id_label.csv")
        os.remove(eval_dir + "\\eval_label.csv")
    return
def upload_and_clear_log(drive, fid, name, clear_logs, model_dir):
    current_fid = create_subfolder(drive, fid, name)
    fid_training =  create_subfolder(drive, current_fid, "Training")
    fid_validation = create_subfolder(drive, current_fid, "Validation")
    #get the newest log folder in logs dir
    result = []
    b = model_dir + "\\logs"
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    latest_subdir = max(result, key=os.path.getmtime)
    #get training and validation files from here
    #training
    latest_subdir_training = latest_subdir + "\\training"
    files = os.listdir(latest_subdir_training)
    log_files = [i for i in files if i.startswith('events')]
    training_file = log_files[0]
    training_file_path = latest_subdir_training + "\\" + training_file
    #validation
    latest_subdir_validation = latest_subdir + "\\validation"
    files = os.listdir(latest_subdir_validation)
    log_files = [i for i in files if i.startswith('events')]
    validation_file = log_files[0]
    validation_file_path = latest_subdir_validation + "\\" + validation_file
    #upload both files
    upload_to_folder(fid_training, training_file_path, validation_file, drive)
    upload_to_folder(fid_validation, validation_file_path, training_file, drive)
    if(clear_logs):
        shutil.rmtree(latest_subdir, ignore_errors=True, onerror=None)
    return
def upload_and_clear_output(drive, fid, clear_output, model_dir):
    current_fid = create_subfolder(drive, fid, "Output")
    output_path = model_dir + "\\output"
    files = os.listdir(output_path)
    output_files = [i for i in files if i.endswith("nii.gz")]
    for output in output_files:
        current_path = output_path + "\\" + output
        upload_to_folder(current_fid, current_path, output, drive)
        if(clear_output):
            os.remove(os.path.join(output_path, output))
    return
