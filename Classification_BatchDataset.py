"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import pandas as pa
import re
import os

class Classification_BatchDataset:
    path = ""
    class_mappings = ""
    files = []
    images = []
    annotations = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, seed = 444):
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

        self._read_images()

        # cast into arrays
        self.images = np.vstack(self.images)
        self.annotations = np.array(self.annotations)

        # Shuffle the data
        perm = np.arange(self.images.shape[0])
        np.random.shuffle(perm)
        self.images = self.images[perm]
        self.annotations = self.annotations[perm]




    def _read_images(self):
        for folder in os.listdir(self.path):
            if os.path.isdir(self.path +"/"+folder) and max(self.class_names[1].isin([folder])):
                    class_index = int(self.class_names[self.class_names[1] == folder][0])
                    self.load_class(folder,class_index)
                    print(folder + " loaded")


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
        image = misc.imread(self.path + "/" + folder + "/" + image)
        nr_x = image.shape[0]/self.tile_size[0]
        nr_y = image.shape[1]/self.tile_size[1]

        for x_i in xrange(0, nr_x):
            for y_i in xrange(0, nr_y):
                self.images.append(image[y_i*self.tile_size[0]:(y_i+1)*self.tile_size[0], x_i*self.tile_size[1]:(x_i+1)*self.tile_size[1]])
                self.annotations.append(class_index)
                if self.images[len(self.images)-1].shape != (self.tile_size[0],self.tile_size[1]):
                    print("sadf")

                # show image
                # from PIL import Image
                # im = Image.fromarray(image[y_i*self.tile_size[0]:(y_i+1)*self.tile_size[0], x_i*self.tile_size[1]:(x_i+1)*self.tile_size[1]])
                # im.show()

        return None

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

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
    data_reader = Classification_BatchDataset("../Datasets/DeepScores/classification_data")
    #data_reader = Classification_BatchDataset("../../classification_data")