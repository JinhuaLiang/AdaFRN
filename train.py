# coding: utf-8
# Time-Fequency Resolution Attention Net(T-FRANet) created by J. Liang in TJU
 
import os
import sys
import dcase_util
import argparse

import numpy as np
import pandas as pd 
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import multi_gpu_model

from utils import callbacks_operation, data_processor, file_processor
from utils import generator, config

sys.path.insert(1, os.path.join(sys.path[0], 'models'))
from AdaFRN import AdaFRN

tf.set_random_seed(1234)

work_space = config.WORKSPACE
data_path = config.DATASET_DIR


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-schedule', '--lr_schedule', 
                        type=str, 
                        default='Adpt', 
                        help="Swich whether use learning rate schedule")
    parser.add_argument('-board', '--tensorboard_storage_path', 
                        type=str, 
                        default=None, 
                        help='set tensorboard path in current dir')
    parser.add_argument('-ckpt', '--checkpoint_storage_path', 
                        type=str, 
                        default=None, 
                        help='Set checkpoint path in current dir')
    parser.add_argument('-res', '--results_storage_path', 
                        type=str, 
                        default=None, 
                        help='Set results path in current dir')
    parser.add_argument('-batch', '--batch_size', 
                        type=int, 
                        default=32, 
                        help='Set the batch size in each epoch')
    parser.add_argument('-lr', '--learning_rate', 
                        type=float, 
                        default=0.001, 
                        help='Set the oringin learning rate ')
    parser.add_argument('-a','--alpha', 
                        type=float, 
                        default=0.2,
                        help='Adjust the hyperparameters in Mixup')
    parser.add_argument('-r', 
                        type=float, 
                        default=8,
                        help='Adjust the hyperparameters in FS')
    parser.add_argument('-epochs', 
                        type=int, 
                        default=300,
                        help='Set the trainning epochs')
    parser.add_argument('-model',
                        type=str,
                        default='adafrn',
                        help='Select the model to be trained')
    parser.add_argument('-mixup',
                        default=False,
                        action='store_true',
                        help='Use mixup or not')
    parser.add_argument('-fine_tuning',
                        default=False,
                        action='store_true',
                        help='Load pre-train model')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    return FLAGS, unparsed


def main(_):
    dcase_util.utils.setup_logging()
    log = dcase_util.ui.FancyLogger()
    log.title('Train for AdaFRN')

    # Initialize the datasets
    db = dcase_util.datasets.TAUUrbanAcousticScenes_2019_DevelopmentSet(
        storage_name ='TAU-urban-acoustic-scenes-2019-development',
        data_path = data_path
    )
    db.initialize()
    db.show()

    log.section_header('Feature Extraction')
    
    # Set feature storage path
    feature_storage_path = os.path.join('development', 'feature', '128x512')
    dcase_util.utils.Path().create(feature_storage_path)
    # Extract feature
    extractor = dcase_util.features.MelExtractor(
        fs=44100,
        win_length_samples=1724,
        hop_length_samples=862,
        n_mels=128
    )
          
    # Loop over all audio files in the dataset with Container and extract features for them
    for audio_filename in db.audio_files:
        # Show some progress
        log.line(os.path.split(audio_filename)[1], indent=2)
        # Get filename for feature data from audio filename
        feature_filename = os.path.join(
            feature_storage_path,
            os.path.split(audio_filename)[1].replace('.wav', '.cpickle')
        )
        # Load audio data
        audio = dcase_util.containers.AudioContainer().load(
            filename=audio_filename,
            mono=True,
            fs=extractor.fs
        )
        # Extract features and store them into FeatureContainer, and save it to the disk
        features = dcase_util.containers.FeatureContainer(
            filename=feature_filename,
            data=extractor.extract(audio.data),
            time_resolution=extractor.hop_length_seconds
        ).save()
        
    log.foot()
    log.section_header('Feature Normalization')
    # Set normalization data storage path
    normalization_storage_path = os.path.join('development', 'normalization')
    dcase_util.utils.Path().create(normalization_storage_path)
    # Loop over all cross-validation folds and calculate mean and std for the training data
    for fold in db.folds():
        # Show some progress
        log.line('Fold {fold:d}'.format(fold=fold), indent=2)
        # Get filename for the normalization factors
        fold_stats_filename = os.path.join(
            normalization_storage_path,
            'norm_fold_{fold:d}.cpickle'.format(fold=fold)
        )
        # Normalizer
        normalizer = dcase_util.data.Normalizer(filename=fold_stats_filename)
        # Loop through all training data
        for item in db.train(fold=fold):
            # Get feature filename
            feature_filename = os.path.join(
                feature_storage_path,
                os.path.split(item.filename)[1].replace('.wav', '.cpickle')
            )
            # Load feature matrix
            features = dcase_util.containers.FeatureContainer().load(
                filename=feature_filename
            )
            # Accumulate statistics
            normalizer.accumulate(features.data)
        # Finalize and save
        normalizer.finalize().save()
    log.foot()

    log.section_header('Data Processing')
    
    # Concatenate the samples in development dataset into train/eval tensors
    train_data = []
    train_label = []
    for item in db.train(fold=1):
        # Get feature filename
        feature_filename = os.path.join(
            feature_storage_path,
            os.path.split(item.filename)[1].replace('.wav', '.cpickle')
        )
        # Load all features.
        features = dcase_util.containers.FeatureContainer().load(
            filename=feature_filename
        )
        # Normalize features.
        # TODO: 
        features = normalizer.normalize(features)   
        # Store feature data
        train_data.append(features.data)
        train_label.append(item.scene_label)
        
    eval_data = []
    eval_label = []
    for item in db.eval(fold=1):
        # Get feature filename
        feature_filename = os.path.join(
            feature_storage_path,
            os.path.split(item.filename)[1].replace('.wav', '.cpickle')
        )
        # Load all features.
        features = dcase_util.containers.FeatureContainer().load(
            filename=feature_filename
        )
        # Normalize features.
        features = normalizer.normalize(features)
        # Store feature data
        eval_data.append(features.data)
        eval_label.append(item.scene_label)
    
    # Convert the data format to np.array
    train_data = np.array(train_data)
    eval_data = np.array(eval_data)
    # Convert the tensor to 4-D tensor
    if train_data.ndim != 4:
        train_data, eval_data = data_processor.add_dimension(train_data, eval_data)
    # Encode the lable to one-hot 
    one_hot_train_label, one_hot_eval_label = data_processor.one_hot_encoder(
        labels_list=db.scene_labels(), 
        num_label_class=db.scene_label_count(), 
        train_label=train_label, eval_label=eval_label)
        
    log.foot()

    log.section_header('train and evaluate')
    
    spectrum_size = (128, 512, 1)
    batch_size = FLAGS.batch_size
    lr = FLAGS.learning_rate
    alpha = FLAGS.alpha
    r = FLAGS.r
    
    # Basic generator
    # train_generator = generator.BasicGenerator(x_set=train_data, y_set=one_hot_train_label, batch_size=batch_size)
    # eval_generator = generator.BasicGenerator(x_set=eval_data, y_set=one_hot_eval_label, batch_size=batch_size)
    
    # # Generator with mixup
    # print("Using Mixup Generator.")
    # train_generator = generator.MixupGenerator(train_data, one_hot_train_label, batch_size=batch_size, alpha=0.4)()
    
    print("Now use {}.".format(FLAGS.model))
    if FLAGS.model == 'adafrn':
        print("r = {}".format(r))
        model = AdaFRN(spectrum_size, num_channels=256, r=r, L=8)
        
        if FLAGS.fine_tuning == True:
            weights_path = './result/adfrn_w_a=0.4/weights.51-0.708.hdf5'
            model.load_weights(filepath=weights_path)
        
    else:
        ValueError("No model named {}!".format(FLAGS.model))

    try:
        parallel_model = multi_gpu_model(model, gpus=2, cpu_relocation=True)
        print("Training with multiple GPUs..")

    except ValueError:
        parallel_model = model
        print("Training with single GPU or CPU..")
    
    parallel_model.compile(
          optimizer=keras.optimizers.Adam(lr=lr),
          loss='categorical_crossentropy',
          metrics=['acc', Precision(), Recall()]
          )

    # Conclude the callbacks option
    callbacks = callbacks_operation.callbacks_maker(
        decay=FLAGS.lr_schedule, 
        tensorboard_storage_path=FLAGS.tensorboard_storage_path, 
        checkpoint_storage_path=FLAGS.checkpoint_storage_path
        )

    if FLAGS.mixup:
        # Generator with mixup
        # print("Using Mixup Generator (alpha={}).".format(alpha))
        # train_generator = generator.MixupGenerator(train_data, 
                                                   # one_hot_train_label, 
                                                   # batch_size=batch_size, 
                                                   # alpha=alpha)()

        # history = parallel_model.fit_generator(
            # train_generator, 
            # epochs=FLAGS.epochs, 
            # verbose=1, 
            # callbacks=callbacks,
            # steps_per_epoch=train_data.shape[0] // batch_size,
            # validation_data=(eval_data, one_hot_eval_label), 
            # workers=1, # 4 
            # use_multiprocessing=False # True
            # )
            
        train_generator = generator.Mixup_threadsafe(train_data, 
                                                     one_hot_train_label, 
                                                     batch_size=batch_size, 
                                                     alpha=alpha)
        
        history = parallel_model.fit_generator(
            train_generator, 
            epochs=FLAGS.epochs, 
            verbose=1, 
            callbacks=callbacks,
            steps_per_epoch=train_data.shape[0] // batch_size,
            validation_data=(eval_data, one_hot_eval_label), 
            workers=1, 
            use_multiprocessing=False
            )
    
    else:        
        history = parallel_model.fit(
            train_data, one_hot_train_label,
            epochs=FLAGS.epochs,
            batch_size=batch_size,
            validation_data=(eval_data, one_hot_eval_label),
            callbacks=callbacks
            )
    
    log.section_header('Test')
    
    # Get eval labels list filename 
    eval_labels_filename = os.path.join('development', 'data', 
                                        'eval_labels.txt')
    # if meta not exist, create eval labels list
    if not os.path.exists(eval_labels_filename):
        # Initialize eval labels container
        eval_meta = dcase_util.containers.MetaDataContainer(
            filename=eval_labels_filename)
        # Strore eval meta
        for item in db.eval(fold=1):
            eval_meta.append(
                {
                    'filename': item.filename,
                    'scene_label': item.scene_label
                }
            )
        eval_meta.save()
    # Whether store the results
    results_storage_path = FLAGS.results_storage_path
    if results_storage_path != None:
        dcase_util.utils.Path().create(results_storage_path)
        # Prediction eval data, replace 'eval' with 'test' if nessessary
        one_hot_eval_predictions = model.predict(eval_data)
        # Get results filename
        results_filename = os.path.join(results_storage_path, 'result.txt')
        # Initialize results container
        res = dcase_util.containers.MetaDataContainer(filename=results_filename)
        eval_predictions = data_processor.one_hot_decoder(
                            labels_list=db.scene_labels(),
                            one_hot_labels=one_hot_eval_predictions)
        for index, item in enumerate(db.eval(fold=1)):
            res.append(
                {
                    'filename': item.filename,
                    'scene_label': eval_predictions[index]
                }
            )
        # Save results container
        res.save()
    log.foot()
    

if __name__ == '__main__':
    FLAGS, unparsed = get_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    main(FLAGS)