import tensorflow as tf
from models.CGNet import CGNet
from models.ESNet import ESNet
from models.Enet import Enet
# from models.ESNet import ESNet
from models.Deeplabv3 import deeplabv3_plus
#from models.ddrnet_23_slim import ddrnet_23_slim
#from models.Enet import Enet
#from models.CGNet import CGNet
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph


from models.ddrnet_23_slim import ddrnet_23_slim

#code reference : https://github.com/tensorflow/tensorflow/issues/32809
model = CGNet()
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        # if write_path:
        #     opts['output'] = 'file:outfile={}'.format(write_path)  # suppress output
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops

print("The FLOPs is:{}".format(get_flops(model)) ,flush=True )