import os

import hypertune
import tensorflow as tf

class HypertuneHook(tf.train.SessionRunHook):
    def __init__(self):
        self.hypertune = hypertune.HyperTune()
        self.hp_metric_tag = os.environ.get('CLOUD_ML_HP_METRIC_TAG', '')
        self.trial_id = os.environ.get('CLOUD_ML_TRIAL_ID', 0)

    def end(self, session):
        step_variable = session.graph.get_collection('global_step')
        global_step = session.run(step_variable)[0]

        tf.logging.info('DEBUG: HypertuneHook called, tag: {}, trial_id: {}, global_step: {}'.format(self.hp_metric_tag, self.trial_id, global_step))

        # The name of the tensor is given in metric_fn in resnet_main_hypertune.py.
        metric_tensor = session.graph.get_tensor_by_name('top_5_accuracy/value:0')

        metric_value = session.run(metric_tensor)

        self.hypertune.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.hp_metric_tag,
            metric_value=metric_value,
            global_step=global_step)

