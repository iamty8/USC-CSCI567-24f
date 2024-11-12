from PIL import Image
import os
import glob


def adjust_gamma(image, gamma=1.0):
    inv_gamma = gamma
    if image.mode == 'RGB':
        lut = [pow(x / 255., inv_gamma) * 255 for x in range(256)] * 3
    else:
        # 灰度图或其他模式
        lut = [pow(x / 255., inv_gamma) * 255 for x in range(256)]
    image = image.point(lut)
    return image


def process_directory(source_dir, target_dir, gamma=0.6):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for subdir, dirs, files in os.walk(source_dir):
        for dir in dirs:
            full_dir_path = os.path.join(subdir, dir)
            target_subdir = os.path.join(target_dir, os.path.relpath(full_dir_path, source_dir))
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)

            for file in glob.glob(full_dir_path + "/*.jpeg"):
                image = Image.open(file)
                adjusted_image = adjust_gamma(image, gamma=gamma)
                save_path = os.path.join(target_subdir, os.path.basename(file))
                adjusted_image.save(save_path, "JPEG")


source_train_dir = 'datasets/mstar/train'
target_train_dir = 'datasets/mstar_gamma/train'
source_test_dir = 'datasets/mstar/test'
target_test_dir = 'datasets/mstar_gamma/test'

process_directory(source_train_dir, target_train_dir, gamma=0.6)
process_directory(source_test_dir, target_test_dir, gamma=0.6)