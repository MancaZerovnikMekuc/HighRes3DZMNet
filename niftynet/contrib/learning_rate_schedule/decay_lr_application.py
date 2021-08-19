import tensorflow as tf

from niftynet.application.segmentation_application import \
    SegmentationApplication
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.layer.loss_segmentation import LossFunction

SUPPORTED_INPUT = set(['image', 'label', 'weight'])


class DecayLearningRateApplication(SegmentationApplication):
    REQUIRED_CONFIG_SECTION = "SEGMENTATION"

    def __init__(self, net_param, action_param, is_training):
        SegmentationApplication.__init__(
            self, net_param, action_param, is_training)
        tf.logging.info('starting decay learning segmentation application')
        self.learning_rate = None
        self.current_lr = action_param.lr
        #if self.action_param.validation_every_n > 0:
        #    raise NotImplementedError("validation process is not implemented "
        #                              "in this demo.")

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()

        if self.is_training:
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(for_training=True),
                                    lambda: switch_sampler(for_training=False))
            else:
                data_dict = switch_sampler(for_training=True)
            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training,
                        'keep_prob': self.net_param.keep_prob}
            net_out = self.net(image, **net_args)

            with tf.name_scope('Optimiser'):
                self.learning_rate = tf.placeholder(tf.float32, shape=[])
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.learning_rate)
            loss_func = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type,
                softmax=self.segmentation_param.softmax)
            data_loss = loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None),
                weight_map=data_dict.get('weight', None))
            loss = data_loss
            loss_func_m = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type='DiceClassOne',
                softmax=self.segmentation_param.softmax
            )
            metricM = loss_func_m(
                prediction=net_out,
                ground_truth=data_dict.get('label', None),
                weight_map=data_dict.get('weight', None))


            loss_func_el = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type='DiceClassTwo',
                softmax=self.segmentation_param.softmax
            )
            #metricEL = loss_func_el(
            #    prediction=net_out,
            #    ground_truth=data_dict.get('label', None),
            #    weight_map=data_dict.get('weight', None))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = data_loss + reg_loss
            else:
                loss = data_loss
            # Get all vars
            to_optimise = tf.trainable_variables()
            vars_to_freeze = \
                self.action_param.vars_to_freeze or \
                self.action_param.vars_to_restore
            if vars_to_freeze:
                import re
                var_regex = re.compile(vars_to_freeze)
                # Only optimise vars that are not frozen
                to_optimise = \
                    [v for v in to_optimise if not var_regex.search(v.name)]
                tf.logging.info(
                    "Optimizing %d out of %d trainable variables, "
                    "the other variables fixed (--vars_to_freeze %s)",
                    len(to_optimise),
                    len(tf.trainable_variables()),
                    vars_to_freeze)

            grads = self.optimiser.compute_gradients(
                loss, var_list=to_optimise, colocate_gradients_with_ops=True)

            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=data_loss, name='loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss, name='loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=self.learning_rate, name='lr',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.learning_rate, name='lr',
                average_over_devices=False, collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=metricM, name='diceM',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            #outputs_collector.add_to_collection(
            #    var=metricEL, name='diceEL',
            #    average_over_devices=True, summary_type='scalar',
            #    collection=TF_SUMMARIES)
        else:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            SegmentationApplication.connect_data_and_network(
                self, outputs_collector, gradients_collector)

    def set_iteration_update(self, iteration_message):
        """
        This function will be called by the application engine at each
        iteration.
        """
        current_iter = iteration_message.current_iter
        if iteration_message.is_training:
            if current_iter > 0 and current_iter % 100 == 0:
                self.current_lr = self.current_lr / 2.0
            iteration_message.data_feed_dict[self.is_validation] = False
        elif iteration_message.is_validation:
            iteration_message.data_feed_dict[self.is_validation] = True
        iteration_message.data_feed_dict[self.learning_rate] = self.current_lr
