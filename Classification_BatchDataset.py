"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
from past.builtins import xrange
import numpy as np
import imageio
import pandas as pa
import re
import os
import sys

class class_dataset_reader:
    path = ""
    class_mappings = ""
    files = []
    images = []
    annotations = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, seed = 444, split = 0.2, min_nr = 2, one_hot=True):
        """
        Initialize a file reader for the DeepScores classification data
        :param records_list: path to the dataset
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        """
        print("Initializing DeepScores Classification Batch Dataset Reader...")
        self.path = records_list
        self.seed = seed

        self.class_names = pa.read_csv(self.path+"/class_names.csv", header=None)

        config = open(self.path+"/config.txt", "r")
        config_str = config.read()
        self.tile_size = re.split('\)|,|\(', config_str)[4:6]

        self.tile_size[0] = int(self.tile_size[0])
        self.tile_size[1] = int(self.tile_size[1])

        self.seed = seed
        self.min_nr = min_nr
        self.split = split
        self.one_hot = one_hot

        # show image
        # from PIL import Image
        # im = Image.fromarray(self.images[234])
        # im.show()
        # print self.annotations[234]

    def read_images(self):
        for folder in os.listdir(self.path):
            if os.path.isdir(self.path +"/"+folder) and max(self.class_names[1].isin([folder])):
                    class_index = int(self.class_names[self.class_names[1] == folder][0])
                    self.load_class(folder,class_index)
                    print(folder + " loaded")

        # cast into arrays
        self.images = np.stack(self.images)
        self.annotations = np.stack(self.annotations)

        # extract test data
        test_indices = []
        train_indices = []
        print("splitting data: " + str(1 - self.split) + "-training " + str(self.split) + "-testing")
        for cla in np.unique(self.annotations):
            if sum(self.annotations == cla) < self.min_nr:
                print(
                "Less than " + str(self.min_nr) + " occurences - removing class " + self.class_names[1][cla])
            else:
                # do split
                cla_indices = np.where(self.annotations == cla)[0]
                np.random.shuffle(cla_indices)
                train_indices.append(cla_indices[0:int(len(cla_indices) * (1 - self.split))])
                test_indices.append(cla_indices[int(len(cla_indices) * (1 - self.split)):len(cla_indices)])

        train_indices = np.concatenate(train_indices)
        test_indices = np.concatenate(test_indices)



        self.test_images = self.images[test_indices]
        self.test_annotations = self.annotations[test_indices]

        self.images = self.images[train_indices]
        self.annotations = self.annotations[train_indices]

        # Shuffle the data
        perm = np.arange(self.images.shape[0])
        np.random.seed(self.seed)
        np.random.shuffle(perm)
        self.images = self.images[perm]
        self.annotations = self.annotations[perm]

        # Reshape to fit Tensorflow
        self.images = np.expand_dims(self.images, -1)
        self.test_images = np.expand_dims(self.test_images, -1)

        if sum(np.unique(self.annotations) != np.unique(self.test_annotations)) != 0:
            print("NOT THE SAME CLASSES IN TRAIN AND TEST - EXITING")
            sys.exit(1)

        self.nr_classes = max(self.test_annotations) + 1
        if self.one_hot:
            self.annotations = np.eye(self.nr_classes, dtype=np.uint8)[self.annotations]
            self.test_annotations = np.eye(self.nr_classes, dtype=np.uint8)[self.test_annotations]


        # self.__channels = True
        # self.images = np.array([self._transform(filename['image']) for filename in self.files])
        # self.__channels = False
        # self.annotations = np.array(
        #     [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        # print (self.images.shape)
        # print (self.annotations.shape)

    def load_class(self, folder, class_index):
        # move trough images in folder
        for image in os.listdir(self.path +"/"+folder):
            self.load_image(folder, image, class_index)
        return None

    def load_image(self,folder,image, class_index):
        image = imageio.imread(self.path + "/" + folder + "/" + image)
        nr_y = image.shape[0] // self.tile_size[0]
        nr_x = image.shape[1] // self.tile_size[1]

        for x_i in xrange(0, nr_x):
            for y_i in xrange(0, nr_y):
                self.images.append(image[y_i*self.tile_size[0]:(y_i+1)*self.tile_size[0], x_i*self.tile_size[1]:(x_i+1)*self.tile_size[1]])
                self.annotations.append(class_index)
                # if self.images[len(self.images)-1].shape != (self.tile_size[0],self.tile_size[1]):
                #     print("sadf")

                # show image
                # from PIL import Image
                # im = Image.fromarray(image[y_i*self.tile_size[0]:(y_i+1)*self.tile_size[0], x_i*self.tile_size[1]:(x_i+1)*self.tile_size[1]])
                # im.show()

        return None

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def get_test_records(self):
        return self.test_images, self.test_annotations

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]



if __name__ == "__main__":
    data_reader = class_dataset_reader("../Datasets/DeepScores/classification_data")
    #data_reader = Classification_BatchDataset("../Datasets/classification_data")
