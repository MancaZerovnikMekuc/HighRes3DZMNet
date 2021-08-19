from clear_and_save_util import *

###### CHECK AND UPDATE THOSE EVERYTIME ########################################
###### BEFORE RUN:
# - infere
# - evaluate


#CHECK THIS NUMBER!!!
selected_model = 30870
config_file = "hrn4mle.ini"
name_of_experiment = "EX1hrn"
model_dir = "C:\\Users\\manca.zerovnik\\Documents\\hrnMLE"
#############################################
# ###################################

###### CHECK EVERYTIME #########################################################
upload_eval = True
upload_models = True
upload_logs = True
upload_output = True
upload_config = True
# clear will be performed only if upload is enabled
clear_eval = True
clear_models = True
clear_logs = True
clear_output = True
#################################################################################

# DO NOT CHANGE
if(upload_eval or upload_models or upload_logs or upload_output or upload_config):
    config_file_path = ".\\configFiles\\" + config_file
    MEL_seg_ID = "1jYEdh2A_r5f6khw_rHAcs6tdwjuuqdZn"
    logs_ID = "18oNqSuL_VA5TXOyRsuiG2W1rD6L0VtXf"
    my_drive = connect_to_drive()
    current_fid = create_subfolder(my_drive, MEL_seg_ID, name_of_experiment)
    if(upload_logs):
        upload_and_clear_log(my_drive, logs_ID, name_of_experiment, clear_output, model_dir)
    if(upload_eval):
        upload_and_clear_eval(my_drive, current_fid, clear_eval, model_dir)
    if(upload_models):
        upload_and_clear_models(my_drive, current_fid, selected_model, clear_models, model_dir)
    if(upload_output):
        upload_and_clear_output(my_drive, current_fid, clear_output, model_dir)
    if(upload_config):
        upload_to_folder(current_fid, config_file_path, config_file, my_drive)
