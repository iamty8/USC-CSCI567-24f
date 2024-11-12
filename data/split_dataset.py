import os
import shutil
from sklearn.model_selection import train_test_split

# 设置源数据集目录、目标训练集目录和目标测试集目录
source_dir = 'datasets/NWPU'
new_dir = 'datasets/NWPU_split'
train_dir = os.path.join(new_dir, 'train')
test_dir = os.path.join(new_dir, 'test')

# 确保训练集和测试集目录存在
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 遍历每个类别
for class_name in sorted(os.listdir(source_dir)):
    # 忽略非目录文件和训练/测试目录
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path) or class_name in ['train', 'test']:
        continue

    # 获取类别内所有图片
    images = [os.path.join(class_path, img) for img in os.listdir(class_path) if
              img.lower().endswith(('png', 'jpg', 'jpeg'))]

    # 分割数据集为训练集和测试集（70%训练，30%测试）
    train_imgs, test_imgs = train_test_split(images, test_size=0.3, random_state=42)

    # 为当前类别创建训练和测试目录
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # 移动图片到对应的训练/测试目录
    for img in train_imgs:
        shutil.move(img, os.path.join(train_class_dir, os.path.basename(img)))
    for img in test_imgs:
        shutil.move(img, os.path.join(test_class_dir, os.path.basename(img)))

print("数据集分割完成。")