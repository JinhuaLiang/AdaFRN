# coding: utf-8
"""define the operations following the end of the terminial
""" 
import os
import sys
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def make_exists(storage_name):
    # Make sure the file name exists
    if not os.path.exists(storage_name):
        dir_name, file_name = os.path.split(os.path.abspath(sys.argv[0]))
        os.makedirs(os.path.join(dir_name, storage_name))


def linear(epoch, lr):
    # design scheduler for 500 epochs
    if epoch == 0:
        lr = lr
    elif epoch % 50 == 0:
        lr = lr * 0.2
    # elif epoch % 25 == 0:
        # lr = lr * 0.6
    print('lr: %f' % lr)
    return lr
    

# class ReduceLROnPlateauWithWarmup(keras.callbacks.ReduceLROnPlateau):
    # def __init__(self, monitor, min_delta, factor, patience, base_learning_rate, warmup_length, warmup_coeff):
        # super(ReduceLROnPlateauWithWarmup, self).__init__(
            # monitor=monitor, 
            # min_delta=min_delta,
            # factor=factor,
            # patience=patience,
            # mode="max"
        # )

        # self.warmup_length = warmup_length
        # self.warmup_coeff = warmup_coeff
        # self.base_learning_rate = base_learning_rate

    # def on_epoch_end(self, epoch, logs=None):
        # if epoch < self.warmup_length:
            # new_lr = self.warmup_coeff*self.base_learning_rate
            # keras.backend.set_value(self.model.optimizer.lr, new_lr)
        # if epoch == self.warmup_length:
            # keras.backend.set_value(self.model.optimizer.lr, self.base_learning_rate)
            # super()._reset()
            # super().on_epoch_end(epoch, logs)
        # else:
            # super().on_epoch_end(epoch, logs)


class ReduceLROnPlateauWithWarmup(keras.callbacks.ReduceLROnPlateau):
    """
    Implements ReduceLROnPlateau with gradual learning rate warmup:
        `lr = initial_lr / scale` ---> `lr = initial_lr`
    This technique was described in the paper "Accurate, Large Minibatch SGD: Training
    ImageNet in 1 Hour". See https://arxiv.org/pdf/1706.02677.pdf for details.
    Math recap:
                                                 batch
        epoch               = full_epochs + ---------------
                                            steps_per_epoch
                               lr     scale - 1
        lr'(epoch)          = ---- * (-------- * epoch + 1)
                              scale     warmup
                               lr
        lr'(epoch = 0)      = ----
                              scale
        lr'(epoch = warmup) = lr
    """

    def __init__(self, monitor, min_delta, factor, patience, warmup_epochs=5, scale=10, steps_per_epoch=None, verbose=0, **kwargs):
        super(ReduceLROnPlateauWithWarmup, self).__init__(
            monitor=monitor, 
            min_delta=min_delta,
            factor=factor,
            patience=patience,
            verbose=verbose,
            **kwargs
        )
        self.warmup_epochs = warmup_epochs
        self.scale = scale
        self.initial_lr = None
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self.current_epoch = None

    def _autodetect_steps_per_epoch(self):
        if self.params.get('steps'):
            # The number of steps is provided in the parameters.
            return self.params['steps']
        elif self.params.get('samples') and self.params.get('batch_size'):
            # Compute the number of steps per epoch using # of samples and a batch size.
            return self.params['samples'] // self.params['batch_size']
        else:
            raise ValueError('Could not autodetect the number of steps per epoch. '
                             'Please specify the steps_per_epoch parameter to the '
                             'LearningRateWarmupCallback() or upgrade to the latest '
                             'version of Keras.')

    def on_train_begin(self, logs=None):
        self.initial_lr = K.get_value(self.model.optimizer.lr)
        if not self.steps_per_epoch:
            self.steps_per_epoch = self._autodetect_steps_per_epoch()
        super()._reset()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if self.current_epoch < self.warmup_epochs:
            old_lr = K.get_value(self.model.optimizer.lr)
            epoch = self.current_epoch + float(batch) / self.steps_per_epoch
            new_lr = self.initial_lr / self.scale * \
                (epoch * (self.scale - 1) / self.warmup_epochs + 1)
            K.set_value(self.model.optimizer.lr, new_lr)
            
        elif self.current_epoch == self.warmup_epochs and batch == 0:
            old_lr = K.get_value(self.model.optimizer.lr)
            epoch = self.current_epoch + float(batch) / self.steps_per_epoch
            new_lr = self.initial_lr / self.scale * \
                (epoch * (self.scale - 1) / self.warmup_epochs + 1)
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose:
                print('Epoch %d: finished gradual learning rate warmup to %s.' %
                  (epoch + 1, new_lr))
            super()._reset()
        
        else:
            # Outside of adjustment scope.
            return


class LearningRateWarmupCallback(keras.callbacks.Callback):
    """
    Implements gradual learning rate warmup:
        `lr = initial_lr / scale` ---> `lr = initial_lr`
    This technique was described in the paper "Accurate, Large Minibatch SGD: Training
    ImageNet in 1 Hour". See https://arxiv.org/pdf/1706.02677.pdf for details.
    Math recap:
                                                 batch
        epoch               = full_epochs + ---------------
                                            steps_per_epoch
                               lr     scale - 1
        lr'(epoch)          = ---- * (-------- * epoch + 1)
                              scale     warmup
                               lr
        lr'(epoch = 0)      = ----
                              scale
        lr'(epoch = warmup) = lr
    """

    def __init__(self, warmup_epochs=5, scale=10, momentum_correction=False, steps_per_epoch=None, verbose=0):
        """
        Construct a new LearningRateWarmupCallback that will gradually warmup learning rate.
        Args:
            warmup_epochs: The number of epochs of the warmup phase. Defaults to 5.
            momentum_correction: Apply momentum correction to optimizers that have momentum. defaults to True.
            steps_per_epoch: The callback will attempt to autodetect number of batches per, epoch with Keras >= 2.0.0. Provide this value if you have an older version of Keras.
            verbose: verbosity mode, 0 or 1.
        """
        super(LearningRateWarmupCallback, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.scale = scale
        self.momentum_correction = momentum_correction
        self.initial_lr = None
        self.restore_momentum = None
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self.current_epoch = None

    def _autodetect_steps_per_epoch(self):
        if self.params.get('steps'):
            # The number of steps is provided in the parameters.
            return self.params['steps']
        elif self.params.get('samples') and self.params.get('batch_size'):
            # Compute the number of steps per epoch using # of samples and a batch size.
            return self.params['samples'] // self.params['batch_size']
        else:
            raise ValueError('Could not autodetect the number of steps per epoch. '
                             'Please specify the steps_per_epoch parameter to the '
                             'LearningRateWarmupCallback() or upgrade to the latest '
                             'version of Keras.')

    def on_train_begin(self, logs=None):
        self.initial_lr = K.get_value(self.model.optimizer.lr)
        if not self.steps_per_epoch:
            self.steps_per_epoch = self._autodetect_steps_per_epoch()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if self.current_epoch > self.warmup_epochs:
            # Outside of adjustment scope.
            return

        if self.current_epoch == self.warmup_epochs and batch > 0:
            # Outside of adjustment scope, final adjustment is done on first batch.
            return

        old_lr = K.get_value(self.model.optimizer.lr)
        epoch = self.current_epoch + float(batch) / self.steps_per_epoch
        new_lr = self.initial_lr / self.scale * \
            (epoch * (self.scale - 1) / self.warmup_epochs  + 1)
        K.set_value(self.model.optimizer.lr, new_lr)

        if self.current_epoch == self.warmup_epochs and self.verbose:
            print('Epoch %d: finished gradual learning rate warmup to %s.' %
                  (epoch + 1, new_lr))

        if hasattr(self.model.optimizer, 'momentum') and self.momentum_correction:
            # See the paper cited above for more information about momentum correction.
            self.restore_momentum = K.get_value(self.model.optimizer.momentum)
            K.set_value(self.model.optimizer.momentum,
                        self.restore_momentum * new_lr / old_lr)

    def on_batch_end(self, batch, logs=None):
        if self.restore_momentum:
            K.set_value(self.model.optimizer.momentum, self.restore_momentum)
            self.restore_momentum = None
    

def callbacks_maker(decay, tensorboard_storage_path, checkpoint_storage_path):
    """ Return a list of callbacks storage operations
        There just one predictor in the func
    """
    callbacks = []
    
    # If save checkpoint results
    if checkpoint_storage_path != None:
        # Make sure the file exists
        make_exists(checkpoint_storage_path)
        # Add checkpoint ops to callbacks
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_storage_path,
                         'weights.{epoch:02d}-{val_acc:.3f}.hdf5'),
            monitor='val_acc',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            period=0)
        callbacks.append(checkpoint)
    
    # If save the tensorboard results
    if tensorboard_storage_path != None:
        # Make sure the path exists
        make_exists(tensorboard_storage_path)
        # Add tensorboard op to callbacks
        tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_storage_path)
        callbacks.append(tensorboard)
    
    if decay == 'linear':
        scheduler = keras.callbacks.LearningRateScheduler(linear)
        callbacks.append(scheduler)
        print("Now lr will reduce linearly")
    elif decay == 'Adpt':
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                      factor=0.2,
                                                      patience=5,
                                                      verbose=1,
                                                      min_delta=0.0001,
                                                      cooldown=0, 
                                                      min_lr=0)
        callbacks.append(reduce_lr)
        print("Now lr will reduce on plateau")
    elif decay == 'purewarmup':
        warmup = LearningRateWarmupCallback(warmup_epochs=5, 
                                            scale=10, 
                                            momentum_correction=False, 
                                            steps_per_epoch=None, 
                                            verbose=1)
    elif decay == 'warmup':
        warmup = ReduceLROnPlateauWithWarmup(monitor='val_loss', 
                                             factor=0.2,
                                             patience=10,
                                             verbose=1,
                                             min_delta=0.0001,
                                             cooldown=0, 
                                             min_lr=0,
                                             warmup_epochs=5, 
                                             scale=10, 
                                             momentum_correction=False, 
                                             steps_per_epoch=None, 
                                             )
        callbacks.append(warmup)
        print("Now lr will reduce on plateau with gradually warmup")
    else:
        print("No lr will reduce automatically")
        pass
    
    return callbacks


def frcnn_callback_maker(decay, tensorboard_storage_path, checkpoint_storage_path):
    """ Return a list of callbacks storage operations for FRCNN
        There are five predictors in the list
    """
    callbacks = []
    # If save checkpoint results
    if checkpoint_storage_path != None:
        # Make sure the file exists
        make_exists(checkpoint_storage_path)
        # Add checkpoint ops to callbacks
        # checkpoint0 = keras.callbacks.ModelCheckpoint(
            # os.path.join(checkpoint_storage_path,
                         # 'weights.{epoch:02d}-\
                         # {val_prediction_0_acc:.4f}-\
                         # {val_prediction_1_acc:.4f}-\
                         # {val_prediction_2_acc:.4f}-\
                         # {val_prediction_3_acc:.4f}-\
                         # {val_prediction_4_acc:.4f}.hdf5'),
            # monitor='val_prediction_0_acc',
            # verbose=0,
            # save_best_only=True,
            # save_weights_only=False,
            # mode='max',
            # period=0)
        # callbacks.append(checkpoint0)
        checkpoint1 = keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_storage_path,
                         'weights.{epoch:02d}-\
                         {val_prediction_1_acc:.4f}-\
                         {val_prediction_2_acc:.4f}-\
                         {val_prediction_3_acc:.4f}-\
                         {val_prediction_4_acc:.4f}.hdf5'),
            monitor='val_prediction_1_acc', 
            verbose=0, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='max', 
            period=0)
        callbacks.append(checkpoint1)
        checkpoint2 = keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_storage_path, 
                         'weights.{epoch:02d}-\
                         {val_prediction_1_acc:.4f}-\
                         {val_prediction_2_acc:.4f}-\
                         {val_prediction_3_acc:.4f}-\
                         {val_prediction_4_acc:.4f}.hdf5'), 
            monitor='val_prediction_2_acc', 
            verbose=0, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='max', 
            period=0)
        callbacks.append(checkpoint2)
        checkpoint3 = keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_storage_path, 
                         'weights.{epoch:02d}-\
                         {val_prediction_1_acc:.4f}-\
                         {val_prediction_2_acc:.4f}-\
                         {val_prediction_3_acc:.4f}-\
                         {val_prediction_4_acc:.4f}.hdf5'), 
            monitor='val_prediction_3_acc', 
            verbose=0, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='max', 
            period=0)
        callbacks.append(checkpoint3)
        checkpoint4 = keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_storage_path, 
                         'weights.{epoch:02d}-\
                         {val_prediction_1_acc:.4f}-\
                         {val_prediction_2_acc:.4f}-\
                         {val_prediction_3_acc:.4f}-\
                         {val_prediction_4_acc:.4f}.hdf5'), 
            monitor='val_prediction_4_acc', 
            verbose=0, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='max', 
            period=0)
        callbacks.append(checkpoint4)
    # If save the tensorboard results
    if tensorboard_storage_path != None:
        # Make sure the path exists
        make_exists(tensorboard_storage_path)
        # Add tensorboard op to callbacks
        tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_storage_path)
        callbacks.append(tensorboard)
    if decay == 'linear':
        scheduler = keras.callbacks.LearningRateScheduler(linear)
        callbacks.append(scheduler)

    return callbacks