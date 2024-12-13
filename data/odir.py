from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from data.data_utils import subsample_instances  # 假设这些函数与 CIFAR 的版本类似
# from data_utils import subsample_instances

class ODIRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.targets = []  # 添加 targets 属性以存储标签数据

        # 假设每个类别的图片都在单独的文件夹中
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 遍历所有类别文件夹，加载图片路径和对应的标签
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.img_paths.append(img_path)
                self.targets.append(self.class_to_idx[cls_name])  # 将类别索引作为标签存储到 targets 中

        # 为每个样本分配唯一索引
        self.uq_idxs = np.arange(len(self.img_paths))
        self.data = self.img_paths  # 将 img_paths 指定为 data 属性，保持一致性

    def __len__(self):
        return len(self.data)  # 或 len(self.targets)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]
        uq_idx = self.uq_idxs[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, uq_idx
    
def subsample_dataset(dataset, idxs):
    # 确保 idxs 中的索引在 dataset.data 的实际大小范围内
    idxs = [idx for idx in idxs if idx < len(dataset.data)]
    
    # 当 idxs 不为空时，对数据集进行子集采样
    if len(idxs) > 0:
        dataset.data = np.array(dataset.data)[idxs].tolist()  # 将 data 转换为列表
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]
        return dataset
    else:
        return None



def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):
    # 选择包含的类别，并生成对应的索引
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {k: i for i, k in enumerate(include_classes)}

    dataset = subsample_dataset(dataset, cls_idxs)

    # 如果需要对标签进行变换，可以解除以下代码注释
    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):
    train_classes = np.unique(train_dataset.targets)

    # 获取训练和验证集的索引
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]
        v_ = np.random.choice(cls_idxs, replace=False, size=(int(val_split * len(cls_idxs)),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_odir_dataset(train_transform, test_transform, train_classes=range(2, 8),
                     prop_train_labels=0.8, split_train_val=False, seed=0):
    """
    获取 ODIR 数据集的训练、验证、测试划分，同时区分标注和未标注数据。
    :param train_transform: 训练集的图像增强方法
    :param test_transform: 测试集的图像增强方法
    :param train_classes: 作为标注训练集的类别
    :param prop_train_labels: 有标注训练集的比例
    :param split_train_val: 是否划分训练集为训练和验证集
    :param seed: 随机种子
    :return: 一个包含训练集、验证集、未标注数据集和测试集的字典
    """
    np.random.seed(seed)

    # 初始化整个训练集
    whole_training_set = ODIRDataset(root_dir="ODIR_5k", transform=train_transform)

    # 获取标注的训练集
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # 划分训练集和验证集
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # 自动生成未标注的数据集类别（排除 train_classes）
    unlabelled_classes = set(np.unique(whole_training_set.targets)) - set(train_classes)

    # 获取未标注的数据集
    unlabelled_dataset = subsample_classes(deepcopy(whole_training_set), include_classes=unlabelled_classes)
    unlabelled_indices = set(unlabelled_dataset.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(unlabelled_dataset), np.array(list(unlabelled_indices)))

    # 初始化测试集
    test_dataset = ODIRDataset(root_dir="ODIR_5k", transform=test_transform)

    # 如果需要划分训练集和验证集
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    # 整理所有数据集
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':
    # 调用 get_odir_dataset
    datasets = get_odir_dataset(
        train_transform=None,
        test_transform=None,
        train_classes=range(2, 8),  # 训练集的类别 2-7
        prop_train_labels=0.75,
        split_train_val=False
    )

    # 打印训练集和未标注数据集的详细信息
    train_labelled = datasets['train_labelled']
    train_unlabelled = datasets['train_unlabelled']

    print(f"Train classes: {train_classes}")
    print(f"Unlabelled classes: {unlabelled_classes}")


    print('Dataset details:')
    if train_labelled is not None:
        print(f"Train Labelled: {len(train_labelled)} samples")
        print(f"Train Labelled Classes: {np.unique(train_labelled.targets)}")
    else:
        print("Train Labelled: None")

    if train_unlabelled is not None:
        print(f"Train Unlabelled: {len(train_unlabelled)} samples")
        print(f"Train Unlabelled Classes: {np.unique(train_unlabelled.targets)}")
    else:
        print("Train Unlabelled: None")

    # 打印其他数据集的详细信息
    for name, dataset in datasets.items():
        if name not in ['train_labelled', 'train_unlabelled']:  # 已打印的跳过
            if dataset is not None:
                print(f'{name}: {len(dataset)} samples')
                print(f'{name} classes: {np.unique(dataset.targets)}')
            else:
                print(f"{name}: None")

