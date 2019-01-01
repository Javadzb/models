from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf
import json

from object_detection import model_hparams
from object_detection import model_lib

from object_detection import selection_funcs as sel
from object_detection.save_subset_imagenetvid_tf_record import save_tf_record
from object_detection.utils import config_util
from object_detection import inputs

import pdb
import os
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

#======================================================================================

"""
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)
#tf.logging.set_verbosity(tf.logging.WARN)
flags = tf.app.flags


flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '/home/abel/DATA/faster_rcnn/resnet101_coco/checkpoints/',
                    'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string('perf_dir', '/home/abel/DATA/faster_rcnn/resnet101_coco/performances/',
                    'Directory to save performance json files.')
flags.DEFINE_string('data_dir', '/home/abel/DATA/ILSVRC/',
                    'Directory that contains data.')
flags.DEFINE_string('pipeline_config_path',
                    '/home/abel/DATA/faster_rcnn/resnet101_coco/configs/faster_rcnn_resnet101_imagenetvid-active_learning-fR5.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('name', 'Rnd-FullDVideoExt',
                    'Name of method to run')
flags.DEFINE_string('cycles','20',
                    'Number of cycles')
flags.DEFINE_string('epochs','10',
                    'Number of epochs')
flags.DEFINE_string('restart_from_cycle','0',
                    'Cycle from which we want to restart training, if any')
flags.DEFINE_string('run','10',
                    'Number of current run')
flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS
"""
#======================================================================================
flags = tf.app.flags

flags.DEFINE_string('name', 'Rnd-FullDVideoExt',
                    'Name of method to run')
flags.DEFINE_string('data_dir', '/home/abel/DATA/ILSVRC/',
                    'Directory that contains data.')
#flags.DEFINE_string('train_dir', '/home/abel/DATA/faster_rcnn/resnet101_coco/checkpoints/',
#                    'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string('perf_dir', '/home/abel/DATA/faster_rcnn/resnet101_coco/performances/',
                    'Directory to save performance json files.')
flags.DEFINE_string('run','10',
                    'Number of current run')
flags.DEFINE_string('model_dir', None, 'Path to output model directory '
		    'where event and checkpoint files will be written.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_string('cycles','20',
                    'Number of cycles')
flags.DEFINE_string('epochs','1',
                    'Number of epochs')
flags.DEFINE_string('restart_from_cycle','0',
                    'Cycle from which we want to restart training, if any')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string('pipeline_config_path',
                    '/home/abel/DATA/faster_rcnn/resnet101_coco/configs/faster_rcnn_resnet101_imagenetvid-active_learning-fR5.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
    'represented as a string containing comma-separated '
    'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
    'run_once', True, 'If running in eval-only mode, whether to run just '
    'one round of eval vs running continuously (default).'
)
FLAGS = flags.FLAGS
#======================================================================================


# This should be a custom name per method once we can overwrite fields in
# pipeline_file

#pdb.set_trace()

data_info = {'data_dir': FLAGS.data_dir,
          'annotations_dir':'Annotations',
          'label_map_path': './data/imagenetvid_label_map.pbtxt',
          'set': 'train_75K_clean_short'}
          #'set': 'train_150K_clean'}
          #'set':'train_shrinked'}

# Harcoded keys to retrieve metrics
keyBike = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n03790512'
keyCar = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n02958343'
keyMotorbike = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n02834778'
keyAll = 'PascalBoxes_Precision/mAP@0.5IOU'


def get_dataset(data_info):
    """ Gathers information about the dataset given and stores it in a
    structure at the frame level.
    Args:
        data_info: dictionary with information about the dataset
    Returns:
        dataset: structure in form of list, each element corresponds to a
            frame and its a dictionary with multiple keys
        videos: list of videos
    """
    dataset = []
    path_file = os.path.join(data_info['data_dir'],'AL', data_info['set'] + '.txt')
    with open(path_file,'r') as pF:
        idx = 0
        for line in pF:
            # Separate frame path and clean annotation flag
            split_line = line.split(' ')
            # Remove trailing \n
            verified = True if split_line[1][:-1] == '1' else False
            path = split_line[0]
            split_path = path.split('/')
            filename = split_path[-1]
            video = split_path[-3]+'/'+split_path[-2]
            dataset.append({'idx':idx,'filename':filename,'video':video,'verified':verified})
            idx+=1
    videos = set([d['video'] for d in dataset])
    return dataset,videos


#def nms_detections(boxes,scores,labels,thresh_nms = 0.8):
    #boxlist = np_box_list.BoxList(boxes)
    #boxlist.add_field('scores',scores)

def visualize_detections(dataset, unlabeled_set, detections, groundtruths):
    detected_boxes = detections['boxes']
    detected_scores = detections['scores']
    detected_labels = detections['labels']

    gt_boxes = groundtruths['boxes']

    for f in dataset:
        if f['idx'] in unlabeled_set:
            im_path = os.path.join(FLAGS.data_dir,'Data','VID','train',f['video'],f['filename'])
            curr_im = Image.open(im_path)
            im_w,im_h = curr_im.size
            gt_im = gt_boxes[unlabeled_set.index(f['idx'])]
            vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_im,im_w,im_h))
            det_im = detected_boxes[unlabeled_set.index(f['idx'])]
            vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(det_im[:2,],im_w,im_h),color='green')
            curr_im.show()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(unused_argv):


  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  #config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)
  

  # Get info about full dataset
  dataset,videos = get_dataset(data_info)


  # Get experiment information from FLAGS
  name = FLAGS.name
  num_cycles = int(FLAGS.cycles)
  run_num = int(FLAGS.run)
  #num_steps = str(train_config.num_steps)
  epochs = int(FLAGS.epochs)
  restart_cycle = int(FLAGS.restart_from_cycle)


  if FLAGS.checkpoint_dir:
    """
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # The first eval input will be evaluated.
      input_fn = eval_input_fns[0]
    if FLAGS.run_once:
      estimator.evaluate(input_fn,
                         num_eval_steps=None,
                         checkpoint_path=tf.train.latest_checkpoint(
                             FLAGS.checkpoint_dir))
    else:
      model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                train_steps, name)
    """
  else:

    # Load active set from cycle 0 and point to right model

    if restart_cycle==0:
        model_dir = FLAGS.model_dir + 'R' + str(run_num) + 'cycle0/'
        #train_config.fine_tune_checkpoint = model_dir + 'model.ckpt'
    else:
        model_dir = FLAGS.model_dir + name + 'R' + str(run_num) + 'cycle' + str(restart_cycle) + '/'
        # Get actual checkpoint model
        #with open(model_dir+'checkpoint','r') as cfile:
        #    line = cfile.readlines()
        #    train_config.fine_tune_checkpoint = line[0].split(' ')[1][1:-2]


    active_set = []
    unlabeled_set=[]

    with open(model_dir + 'active_set.txt', 'r') as f:
        for line in f:
            active_set.append(int(line))

    for cycle in range(restart_cycle+1,num_cycles+1):


        #### Evaluation of trained model on unlabeled set to obtain data for selection

        if 'Rnd' not in name and cycle < num_cycles:

            eval_train_dir = model_dir + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + 'eval_train/'

            if os.path.exists(eval_train_dir + 'detections.dat'):
                with open(eval_train_dir + 'detections.dat','rb') as infile:
                ###### pdb remove latinq
                    detected_boxes = pickle.load(infile)
                    #detected_boxes = pickle.load(infile,encoding='latin1')
            else:

                # Get unlabeled set
                data_info['output_path'] = FLAGS.data_dir + 'AL/tfrecords/' + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + '_unlabeled.record'

                # Do not evaluate labeled samples, their neighbors or unverified frames
                aug_active_set =  sel.augment_active_set(dataset,videos,active_set,num_neighbors=5)

                unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]


                # For TCFP, we need to get detections for pretty much every frame,
                # as not candidates can may be used to support candidates
                if ('TCFP' in name):
                    unlabeled_set = [i for i in range(len(dataset))]

                print('Unlabeled frames in the dataset: {}'.format(len(unlabeled_set)))


		save_tf_record(data_info,unlabeled_set)


		"""

	        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path=FLAGS.pipeline_config_path,
                                           config_override=None)                
	        eval_input_configs = configs['eval_input_configs']
		eval_config = configs['eval_config']
		model_config = configs['model']
                eval_input_configs = configs['eval_input_configs']

		MODEL_BUILD_UTIL_MAP = {'create_eval_input_fn': inputs.create_eval_input_fn}
  		create_eval_input_fn = MODEL_BUILD_UTIL_MAP['create_eval_input_fn']

                eval_input_fns = [create_eval_input_fn(
                       eval_config=eval_config,
                       eval_input_config=eval_input_config,
                       model_config=model_config) for eval_input_config in eval_input_configs]
		"""


                # Set number of eval images to number of unlabeled samples and point to tfrecord
                #eval_input_config.tf_record_input_reader.input_path[0] = data_info['output_path']
                #eval_config.num_examples = len(unlabeled_set)

	        model_dir = FLAGS.model_dir + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + '/'
        	config = tf.estimator.RunConfig(model_dir=model_dir)

                train_and_eval_dict = model_lib.create_estimator_and_inputs(
 			Unlabeled_set_length=len(unlabeled_set),
	    		Active_set_length=len(active_set),
            		epochs=epochs,
            		data_info=data_info,
            		FLAGS=FLAGS,
	    		restart_cycle=restart_cycle,
            		run_config=config,
            		hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
            		pipeline_config_path=FLAGS.pipeline_config_path,
            		train_steps=FLAGS.num_train_steps,
            		sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
           	        sample_1_of_n_eval_on_train_examples=(FLAGS.sample_1_of_n_eval_on_train_examples))
       
	        estimator = train_and_eval_dict['estimator']
        	train_input_fn = train_and_eval_dict['train_input_fn']
        	eval_input_fns = train_and_eval_dict['eval_input_fns']
        	eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
        	predict_input_fn = train_and_eval_dict['predict_input_fn']
        	train_steps = train_and_eval_dict['train_steps']

		"""
	        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        	    train_input_fn,
        	    eval_input_fns,
        	    eval_on_train_input_fn,
        	    predict_input_fn,
           	    train_steps,
	            eval_on_train_data=False)

                def get_next_eval_train(config):
                   return dataset_builder.make_initializable_iterator(
                        dataset_builder.build(config)).get_next()

                # Initialize input dict again (necessary?)
                #create_eval_train_input_dict_fn = functools.partial(get_next_eval_train, eval_input_config)

                graph_rewriter_fn = None
                if 'graph_rewriter_config' in configs:
                    graph_rewriter_fn = graph_rewriter_builder.build(
                        configs['graph_rewriter_config'], is_training=False)

                # Need to reset graph for evaluation
                tf.reset_default_graph()


                #if FLAGS.eval_training_data:
                #name = 'evaluation_of_training_data'
                #input_fn = eval_on_train_input_fn
                #else:
                #   name = 'validation_data'
                #   # The first eval input will be evaluated.
		"""

                input_fn = eval_input_fns[0]

                if FLAGS.run_once:
                   predictions=estimator.evaluate(input_fn,
                         checkpoint_path=tf.train.latest_checkpoint(eval_train_dir))
                else:
                   pdb.set_trace()
                   model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                train_steps, name)

		pdb.set_trace()

                #visualize_detections(dataset, unlabeled_set, detected_boxes, groundtruth_boxes)
                with open(eval_train_dir + 'detections.dat','wb') as outfile:
                    pickle.dump(detected_boxes,outfile, protocol=pickle.HIGHEST_PROTOCOL)

                print('Done computing detections in training set')


                # Remove tfrecord used for training
                if os.path.exists(data_info['output_path']):
                    os.remove(data_info['output_path'])


        #### Training of current cycle
        model_dir = FLAGS.model_dir + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + '/'
        config = tf.estimator.RunConfig(model_dir=model_dir)


        # Budget for each cycle is the number of videos (0.5% of train set)
        if ('Rnd' in name):
            #indices = select_random_video(dataset,videos,active_set)
            #indices = sel.select_random(dataset,videos,active_set,budget=num_videos)
            indices = sel.select_random_video(dataset,videos,active_set)
        else:
            if ('Ent' in name):
                indices = sel.select_entropy_video(dataset,videos,FLAGS.data_dir,active_set,detected_boxes)
            elif ('Lst' in name):
                indices = sel.select_least_confident_video(dataset,videos,active_set,detected_boxes)
            elif ('TCFP' in name):
                indices = sel.select_TCFP_per_video(dataset,videos,FLAGS.data_dir,active_set,detected_boxes)
            elif ('FP_gt' in name):
	        indices = sel.selectFpPerVideo(dataset,videos,active_set,detected_boxes,groundtruth_boxes,cycle)
            elif ('FN_gt' in name):
	        indices = sel.selectFnPerVideo(dataset,videos,active_set,detected_boxes,groundtruth_boxes,cycle)
            elif ('FPN' in name):
	        indices = sel.select_FPN_PerVideo(dataset,videos,active_set,detected_boxes,groundtruth_boxes,cycle)

        active_set.extend(indices)

        data_info['output_path'] = FLAGS.data_dir + 'AL/tfrecords/' + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + '.record'
        save_tf_record(data_info,active_set)

        pdb.set_trace()

        train_and_eval_dict = model_lib.create_estimator_and_inputs(
            Unlabeled_set_length=len(unlabeled_set),
	    Active_set_length=len(active_set),
            epochs=epochs,
            data_info=data_info,
            FLAGS=FLAGS,
	    restart_cycle=restart_cycle,
            run_config=config,
            hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
            pipeline_config_path=FLAGS.pipeline_config_path,
            train_steps=FLAGS.num_train_steps,
            sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=(FLAGS.sample_1_of_n_eval_on_train_examples))
       
        estimator = train_and_eval_dict['estimator']
        train_input_fn = train_and_eval_dict['train_input_fn']
        eval_input_fns = train_and_eval_dict['eval_input_fns']
        eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
        predict_input_fn = train_and_eval_dict['predict_input_fn']
        train_steps = train_and_eval_dict['train_steps']

        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False)


        print('-----------------train and evaluation-------------------------')

        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


        #Save active_set in train dir in case we want to restart training
        with open(model_dir + 'active_set.txt', 'w') as f:
            for item in active_set:
                f.write('{}\n'.format(item))

        # Remove tfrecord used for training
        if os.path.exists(data_info['output_path']):
            os.remove(data_info['output_path'])

            # Update initial model, add latest cycle
            #train_config.fine_tune_checkpoint = train_dir + 'model.ckpt-' + num_steps


if __name__ == '__main__':
  tf.app.run()



