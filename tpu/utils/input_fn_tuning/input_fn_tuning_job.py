import argparse
from functools import partial
import json
import os

import oyaml as yaml
import shlex
import signal
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from subprocess import Popen, PIPE, call
import tensorflow as tf
import time

from tpu_utils import *


# The search space needs to map exactly to input_fn_params in trainer.py
space = [
    Integer(1, 256, name='tfrecord_dataset_buffer_size'),
    Integer(1, 256, name='tfrecord_dataset_num_parallel_reads'),
    Integer(1, 96, name='parallel_interleave_cycle_length'),
    Integer(1, 2, name='parallel_interleave_block_length'),
    Integer(1, 64, name='parallel_interleave_buffer_output_elements'),
    Integer(1, 64, name='parallel_interleave_prefetch_input_elements'),
    Integer(1, 256, name='map_and_batch_num_parallel_calls'),
    Integer(1, 256, name='transpose_num_parallel_calls'),
    # prefetch_buffer_size == -1 corresponds to tf.contrib.data.AUTOTUNE.
    Integer(-1, 1024, name='prefetch_buffer_size')
]

# Commands to be run in subprocesses
submit_script_name = 'input_fn_tuning_submit.sh'
trace_script_name = 'input_fn_tuning_trace.sh'

submit_command = shlex.split('bash {}'.format(submit_script_name))
trace_command = shlex.split('bash {}'.format(trace_script_name))
copy_cmd = 'gsutil -m cp -r {} {}/'

def tpu_utils_env(env):
    project_id = env['PROJECT_ID']
    location = env['LOCATION']
    tpu_name = env['TPU_NAME']

    return [project_id, location, tpu_name]


def create_tpu_and_wait(env, sleep_time=30):
    env = tpu_utils_env(env)

    print('>>>>> creating tpu')
    create_tpu(*env)

    time.sleep(sleep_time)

    node = get_tpu(*env)
    while node['state'] != 'READY':
        print('>>>>> waiting for tpu to be ready')
        time.sleep(sleep_time)
        node = get_tpu(*env)


def delete_tpu_and_wait(env, sleep_time=10):
    env = tpu_utils_env(env)

    print('>>>>> deleting tpu')
    delete_tpu(*env)

    tpu_name = env.pop(-1)

    nodes = list_tpus(*env)
    while nodes and 'nodes' in nodes and tpu_name in [n['name'].split('/')[-1] for n in nodes['nodes']]:

            print('>>>>> waiting for tpu to be deleted')
            time.sleep(sleep_time)

            nodes = list_tpus(*env)


def build_submit_script(input_fn_params):
    print('>>>>> creating submit script with parameters: {}'.format(input_fn_params))

    with open('input_fn_tuning_submit_base.sh', 'r') as f:
        submit_sh = f.read()

    input_fn_params_str = ' '.join(['--{}={}'.format(k, v) for k, v in input_fn_params.items()])
    submit_sh = submit_sh.format(input_fn_params_str=input_fn_params_str)
    
    with open(submit_script_name, 'w') as f:
        f.write(submit_sh)


def build_trace_script():
    print('>>>>> creating trace script with parameters')

    with open('input_fn_tuning_trace_base.sh', 'r') as f:
        trace_sh = f.read()
    
    with open(trace_script_name, 'w') as f:
        f.write(trace_sh)


def kill_process(process):
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except:
        pass


def make_profile_tpu(subprocess_env):
    def profile_tpu(**input_fn_params):
        # location to save the accumulated results
        output_name = subprocess_env['OUTPUT_NAME']

        # trial-specific output in output_dir
        timestamp = str(int(time.time()))
        output_uri = os.path.join(subprocess_env['OUTPUT_DIR'], 'trials', timestamp)

        # model_dir is used only within a trial.
        # Its content is backed up in output_uri at the end of each trial.
        tpu_name = subprocess_env['TPU_NAME']
        model_dir = subprocess_env['MODEL_DIR']
        if tf.gfile.Exists(model_dir):
            tf.gfile.DeleteRecursively(model_dir)

        # create new TPU each time
        create_tpu_and_wait(subprocess_env)

        # create the scripts
        build_submit_script(input_fn_params)
        build_trace_script()

        # run task
        print('>>>>> running training task')
        p_submit = Popen(submit_command, stdout=PIPE, preexec_fn=os.setsid, env=subprocess_env)

        # wait until the job starts before starting to collect trace
        time.sleep(120)

        # run profiler
        p_submit.poll()
        returncode = p_submit.returncode

        n_scores = 3
        n_attempts = 10

        scores = []
        counter = 0
        checked_filenames = set()

        while returncode is None and len(scores) < n_scores and counter < n_attempts:
            print('>>>>> running profiler')
            p_trace = Popen(trace_command, stdout=PIPE, preexec_fn=os.setsid, env=subprocess_env)
            counter += 1

            time.sleep(45)
            kill_process(p_trace)

            print('>>>>> checking trace files')

            trace_filenames = tf.gfile.Glob('{}/plugins/profile/**/input_pipeline.json'.format(model_dir))

            if trace_filenames:
                early_stop = False
                for trace_filename in trace_filenames:
                    if trace_filename in checked_filenames:
                        continue

                    print('>>>>> reading: {}'.format(trace_filename))
                    with tf.gfile.GFile(trace_filename, 'r') as f:
                        json_str = f.read()

                    checked_filenames.add(trace_filename)
                    input_pipeline = json.loads(json_str)

                    # some trace files might not have a valid score
                    try:
                        infeed_percent_average = float(input_pipeline[0]['p']['infeed_percent_average'])

                        if infeed_percent_average > 0.0:
                            scores.append(infeed_percent_average)
                            print('>>>>> current scores: {}'.format(scores))
                    except:
                        pass

                    # This happens when each training step takes too long.
                    if 'No step time measured' in json_str:
                        early_stop = True

                if early_stop:
                    print('>>>>> early stopping')
                    break

            p_submit.poll()
            returncode = p_submit.returncode

        print('>>>>> training process finished with returncode: {}, number of attempts: {}, number of scores: {}'.format(returncode, counter, len(scores)))

        # kill processes, just in case
        print('>>>>> killing training process')
        kill_process(p_submit)

        # calculate average score
        print('>>>>> calculating score')
        if scores:
            score = sum(scores) / len(scores)
        else:
            # Give the worst possible score when no valid scores collected.
            score = 100.0

        print('>>>>> scores: {}, average score: {}'.format(scores, score))

        # write artifacts to output_uri:
        # the generated submit script, the whole model_dir, and the scores
        print('>>>>> writing trial outputs')
        tf.gfile.Copy(submit_script_name, os.path.join(output_uri, submit_script_name))

        copy_command = copy_cmd.format(model_dir, output_uri)
        copy_command = shlex.split(copy_command)
        call(copy_command)

        with tf.gfile.GFile(os.path.join(output_uri, 'scores.txt'), 'w') as f:
            f.write(str(scores))

        # Add new results to the accumulated results
        params_scores = {
            'input_fn_params': {k:int('{}'.format(v)) for k, v in input_fn_params.items()},
            'scores': scores,
            'score': score
        }
        entry = {timestamp: params_scores}

        with tf.gfile.GFile(output_name, 'a') as f:
            yaml.dump(entry, f, default_flow_style=False)

        # clean up artifacts
        print('>>>>> removing artifacts')
        os.remove(submit_script_name)
        os.remove(trace_script_name)

        # delete TPU
        delete_tpu_and_wait(subprocess_env)

        return score

    return profile_tpu


def load_previous_trials(output_name):
    print('>>>>> Loading previous results')
    with tf.gfile.GFile(output_name, 'r') as f:
        yaml_str = f.read()

    results_dict = yaml.load(yaml_str)
    x0 = []
    y0 = []

    if results_dict:
        for timestamp, scores_dict in results_dict.items():
            score = scores_dict['score']
            params_dict = scores_dict['input_fn_params']
            params = [params_dict[d.name] for d in space]

            x0.append(params)
            y0.append(score)
    else:
        x0 = None
        y0 = None

    return x0, y0


def main(args):
    # The aggregated output from previous trials.
    output_name = os.path.join(args.output_dir, 'params_scores.yaml')

    if tf.gfile.Exists(output_name):
        x0, y0 = load_previous_trials(output_name)
    else:
        # No previous trial, create a file to record scores.
        x0 = None
        y0 = None
        with tf.gfile.GFile(output_name, 'w') as f:
            f.write('')

    subprocess_env = {
        'PROJECT_ID': args.project_id,
        'LOCATION': args.location,
        'TPU_NAME': args.tpu_name,
        # OUTPUT_DIR holds results from the whole optimation job (`args.n_calls` trials).
        'OUTPUT_DIR': args.output_dir,
        'OUTPUT_NAME': output_name,
        # MODEL_DIR is cleared at the start of each trial.
        'MODEL_DIR': os.path.join(args.output_dir, 'model_dir')
    }

    # create the objective function with runtime arguments
    profile_tpu = make_profile_tpu(subprocess_env)
    profile_tpu = use_named_args(space)(profile_tpu)

    gp_minimize(profile_tpu, space, n_calls=args.n_calls, n_random_starts=5, x0=x0, y0=y0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--project-id',
        type=str,
        default='')
    parser.add_argument(
        '--location',
        type=str,
        default='europe-west4-a')
    parser.add_argument(
        '--tpu-name',
        type=str,
        default='input_fn_tuning_tpu')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='gs://your-gcs-bucket/path')

    parser.add_argument(
        '--n-calls',
        type=int,
        default=10)

    args, _ = parser.parse_known_args()

    main(args)


