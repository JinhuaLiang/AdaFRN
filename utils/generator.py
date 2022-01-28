import numpy as np
from tensorflow import keras


class BasicGenerator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
    
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        
    def __len__(self):
        """ Calculate iteration num per epoch"""
        return int(np.ceil(len(self.x) / float(self.batch_size)))
        
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
        

       
class MixupGenerator():
    """ """
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        print("alpha is set to {}".format(alpha))

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)
        
        d = np.sqrt( X_l ** 2 + (1 - X_l) ** 2) #re-normalize the spectrum
        X_l /= d 

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        
        X = X1 * X_l + X2 * (1/d - X_l)
        # X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
"""
class MixupGenerator():
    """ """
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        # print("alpha is set to {}".format(alpha))

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(0.1, 0.9, self.batch_size)
        print("Now use B(0.1, 0.9)")
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)
        
        d = np.sqrt( X_l ** 2 + (1 - X_l) ** 2) #re-normalize the spectrum
        X_l /= d 

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        
        X = X1 * X_l + X2 * (1/d - X_l)
        # X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
"""
class Mixup_threadsafe(keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, X_train, y_train, batch_size=16, shuffle=True, alpha=.2, datagen=None):
            'Initialization'
            self.batch_size = batch_size
            self.X_train = X_train
            self.y_train = y_train
            self.shuffle = shuffle
            self.on_epoch_end()
            self.alpha= alpha
            self.datagen=datagen
    
        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.X_train) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
            # Generate data
            X, y = self.__data_generation(indexes)
    
            return X, y
    
        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.X_train))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
        
        def __data_generation(self, batch_ids):
            _, h, w, c = self.X_train.shape
            l = np.random.beta(self.alpha, self.alpha, self.batch_size)
            X_l = l.reshape(self.batch_size, 1, 1, 1)
            y_l = l.reshape(self.batch_size, 1)
            
            # Re-normalize the spectrum
            d = np.sqrt( X_l ** 2 + (1 - X_l) ** 2) 
            X_l /= d 
            
            batch_ids2 = np.random.permutation(batch_ids)
            X1 = self.X_train[batch_ids]
            # X2 = self.X_train[np.flip(batch_ids)] #replaced this with flip
            X2 = self.X_train[batch_ids2]
            # X = X1 * X_l + X2 * (1 - X_l)
            X = X1 * X_l + X2 * (1/d - X_l)
        
            if self.datagen:
                for i in range(self.batch_size):
                    X[i] = self.datagen.random_transform(X[i])
                    X[i] = self.datagen.standardize(X[i])
        
            y1 = self.y_train[batch_ids]
            # y2 = self.y_train[np.flip(batch_ids)]
            y2 = self.y_train[batch_ids2]
            y = y1 * y_l + y2 * (1 - y_l) #removed the list option
        
            # return X/255, y #Rex added dividing by 255 here
            return X, y