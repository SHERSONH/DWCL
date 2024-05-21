from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import scipy.io as sio
 


class Caltech_6V(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)        
        scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(view):
        # for i in [0, 3]:
            self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class Caltech_5V(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)        
        scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(view):
        # for i in [0, 3]: 
            self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class Caltech20class(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = data['X'].shape[1]
        self.multi_view = []
        self.labels = np.array(np.squeeze(data['Y'])).astype(np.int32)
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(view):
            self.multi_view.append(data['X'][0, i].astype(np.float32))
            print(data['X'][0, i].shape)
            self.dims.append(data['X'][0, i].shape[1])
        scaler = MinMaxScaler()
        for i in range(self.view):
            self.multi_view[i] = scaler.fit_transform(self.multi_view[i])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
            
        for i in range(self.view):
            data_view = self.multi_view[i]
            data_getitem.append(torch.from_numpy(data_view[idx]))
        return data_getitem, self.labels[idx], torch.from_numpy(np.array(idx)).long()
    
class NUSWIDE(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        # scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        # for i in range(5000):
        #     print(data['X1'][i][-1])
        # X1 = data['X1'][:, :-1]
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)][:, :-1].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)][:, :-1].shape)
            self.dims.append(data['X' + str(i + 1)][:, :-1].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class DHA(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)

        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MSRCv1(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = data['X'].shape[1]
        self.multi_view = []
        self.labels = np.array(np.squeeze(data['Y'])).astype(np.int32)
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(view):
            self.multi_view.append(data['X'][0, i].astype(np.float32))
            print(data['X'][0, i].shape)
            self.dims.append(data['X'][0, i].shape[1])
        scaler = MinMaxScaler()
        for i in range(self.view):
            self.multi_view[i] = scaler.fit_transform(self.multi_view[i])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size
 
    def __getitem__(self, idx):
        data_getitem = []
            
        for i in range(self.view):
            data_view = self.multi_view[i]
            data_getitem.append(torch.from_numpy(data_view[idx]))
        return data_getitem, self.labels[idx], torch.from_numpy(np.array(idx)).long()

class scene(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = data['X'].shape[1]
        self.multi_view = []
        self.labels = np.array(np.squeeze(data['Y'])).astype(np.int32)
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(view):
            self.multi_view.append(data['X'][0, i].astype(np.float32))
            print(data['X'][0, i].shape)
            self.dims.append(data['X'][0, i].shape[1])
        scaler = MinMaxScaler()
        for i in range(self.view):
            self.multi_view[i] = scaler.fit_transform(self.multi_view[i])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
            
        for i in range(self.view):
            data_view = self.multi_view[i]
            data_getitem.append(torch.from_numpy(data_view[idx]))
        return data_getitem, self.labels[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.multi_view = []
        self.labels = np.array(np.squeeze(data['Y'])).astype(np.int32)
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)

        X1 = data['X1'].reshape(data['X1'].shape[0], data['X1'].shape[1] * data['X1'].shape[2]).astype(np.float32)
        X2 = data['X2'].reshape(data['X2'].shape[0], data['X2'].shape[1] * data['X2'].shape[2]).astype(np.float32)
        X3 = data['X3'].reshape(data['X3'].shape[0], data['X3'].shape[1] * data['X3'].shape[2]).astype(np.float32)
        print(X1.shape)
        print(X2.shape)
        print(X3.shape)
        self.multi_view.append(X1)
        self.multi_view.append(X2)
        self.multi_view.append(X3)
        self.dims.append(X1.shape[1])
        self.dims.append(X2.shape[1])
        self.dims.append(X3.shape[1])

        self.view = len(self.multi_view)

        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
            
        for i in range(self.view):
            data_view = self.multi_view[i]
            data_getitem.append(torch.from_numpy(data_view[idx]))
        return data_getitem, self.labels[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "Caltech":
        dataset = Caltech_6V('data/Caltech.mat', view=6)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "Caltech_5V":
        dataset = Caltech_5V('data/Caltech_5V.mat', view=5)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "Caltech20class":
        dataset = Caltech20class('data/Caltech20class.mat', view=6)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('data/NUSWIDE.mat', view=5)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "DHA":
        dataset = DHA('data/DHA.mat', view=2)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_nu
    elif dataset == "MSRCv1":
        dataset = MSRCv1('data/MSRCv1.mat', view=5)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num

    elif dataset == "scene":
        dataset = scene('data/Scene15.mat', view=3)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num


    elif dataset == "Fashion":
        dataset = Fashion('data/Fashion.mat', view=3)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num

    
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
