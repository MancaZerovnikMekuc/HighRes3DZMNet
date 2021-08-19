#import tensorflow as tf
# Creates a graph.
#a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
#print(sess.run(c))

import tensorflow as tf
# ckpt_path: full path to checkpoint file (e.g.: /path/to/ckpt/model.ckpt-###)
# output_file: name of output file (e.g.: /path/to/file/net_vars.txt)
def get_ckpt_vars(ckpt_path, output_file):
    file = open(output_file, 'w+')
    for var in tf.train.list_variables(ckpt_path):
        file.write(str(var) + '\n')
    file.close()

get_ckpt_vars('C:/Users/manca.zerovnik/Documents/EX1hrn2classBH/models/model.ckpt-35000', \
              'C:/Users/manca.zerovnik/Documents/net_vars.txt')