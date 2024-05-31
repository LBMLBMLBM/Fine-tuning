import torch
import numpy as np
import os
from PIL import Image
import pickle
import tarfile

class cub200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(cub200, self).__init__()

        self.root = root  # 数据集的根目录
        self.train = train  # 是否为训练集
        self.transform = transform  # 数据转换函数

        # 检查是否已经提取并处理好数据
        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            # 如果数据还未处理,则执行数据提取和处理
            self._extract()

        # 根据训练/测试模式,加载对应的数据和标签
        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )

    def __len__(self):
        """返回数据集的大小"""
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        """根据索引返回图像和标签"""
        if self.train:
            img, label = self.train_data[idx], self.train_label[idx]
        else:
            img, label = self.test_data[idx], self.test_label[idx]
        img = Image.fromarray(img)  # 将numpy数组转换为PIL图像
        if self.transform is not None:
            img = self.transform(img)  # 应用数据转换操作
        return img, label

    def _check_processed(self):
        """检查是否已经提取并处理好数据集"""
        assert os.path.isdir(self.root) == True  # 检查根目录是否存在
        assert os.path.isfile(os.path.join(self.root, 'CUB_200_2011.tgz')) == True  # 检查原始数据集是否存在
        return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))  # 检查是否已经生成了处理后的数据

    def _extract(self):
        """处理原始数据集"""
        processed_data_path = os.path.join(self.root, 'processed')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)  # 创建处理后的数据存储目录

        cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        images_txt_path = 'CUB_200_2011/images.txt'
        train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'

        tar = tarfile.open(cub_tgz_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember(images_txt_path))
        train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
        if not (images_txt and train_test_split_txt):
            print('Extract image.txt and train_test_split.txt Error!')
            raise RuntimeError('cub-200-1011')

        images_txt = images_txt.read().decode('utf-8').splitlines()
        train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

        id2name = np.genfromtxt(images_txt, dtype=str)  # 获取图像ID和文件名的映射
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)  # 获取训练/测试标签

        print('Finish loading images.txt and train_test_split.txt')
        train_data, train_labels = [], []
        test_data, test_labels = [], []
        print('Start extract images..')
        cnt, train_cnt, test_cnt = 0, 0, 0
        for _id in range(id2name.shape[0]):
            cnt += 1

            image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = tar.extractfile(tar.getmember(image_path))
            if not image:
                print('get image: ' + image_path + ' error')
                raise RuntimeError
            image = Image.open(image)
            label = int(id2name[_id, 1][:3]) - 1  # 从文件名中提取类别标签

            if image.getbands()[0] == 'L':
                image = image.convert('RGB')  # 确保图像为RGB格式
            image_np = np.array(image)
            image.close()

            if id2train[_id, 1] == 1:
                train_cnt += 1
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_cnt += 1
                test_data.append(image_np)
                test_labels.append(label)
            if cnt % 1000 == 0:
                print('{} images have been extracted'.format(cnt))
        print('Total images: {}, training images: {}. testing images: {}'.format(cnt, train_cnt, test_cnt))
        tar.close()

        # 将处理后的数据保存到磁盘
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))
