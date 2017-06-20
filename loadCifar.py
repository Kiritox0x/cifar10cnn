import pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict


def get_cifar_info(path):
    info = unpickle('{}/batches.meta'.format(path))
    num_cases_per_batch = info['num_cases_per_batch']
    label_names = info['label_names']
    num_vis = info['num_vis']
    return num_cases_per_batch, label_names, num_vis

def one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1
	return np.eye(num_classes, dtype=float)[class_numbers]

def load_cifar10_train(path):
    data, labels = [], []
    for i in range(1,6):
        print('extracting data_batch_{}'.format(i))
        batch_data = unpickle('{}/data_batch_{}'.format(path, i))
        if i==1:
            data = batch_data['data']
            labels = batch_data['labels']
        else:
            data = np.vstack((data,batch_data['data']))
            labels = np.hstack((labels,batch_data['labels']))
    print('Train data loaded successful')
    raw_float = np.array(data, dtype=float)/255.0
    data = raw_float.reshape([-1,3,32,32])
    data = data.transpose([0,2,3,1])
    labels = np.reshape(labels,([-1]))
    return data, labels, one_hot_encoded(class_numbers=labels)

def load_cifar10_test(path):
    data, labels = [], []
    batch_data = unpickle('{}/test_batch'.format(path))
    data = batch_data['data']
    labels = batch_data['labels']
    print('Test data loaded successful')
    raw_float = np.array(data, dtype=float)/255.0
    data = raw_float.reshape([-1,3,32,32])
    data = data.transpose([0,2,3,1])
    labels = np.reshape(labels,([-1]))
    return data, labels, one_hot_encoded(class_numbers=labels)

def show_rgb_image(data, index):
    img = data[index].reshape(3,1024)
    img = img.T
    img = img.reshape(32,32,3)
    plt.imshow(img)
    
def show_image(data,index):
    img = np.reshape(data[index, :], (32, 32))
    plt.imshow(img, cmap='Greys_r')
    plt.axis('off')

