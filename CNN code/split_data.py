import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # If the folder exists, delete it first
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # make sure random reproducibility
    random.seed(0)

    # 10% of the dataset to the test set
    split_rate = 0.2

    # to the unziped kaggle_datas folder
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "dataset")
    origin_plankton_path = os.path.join(data_root, "kaggle_datas")
    assert os.path.exists(origin_plankton_path), "path '{}' does not exist.".format(origin_plankton_path)

    flower_class = [cla for cla in os.listdir(origin_plankton_path)
                    if os.path.isdir(os.path.join(origin_plankton_path, cla))]

    # create folder to save training set
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in flower_class:
        # create folder for different attributes
        mk_file(os.path.join(train_root, cla))

    # create folder to save testing set
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in flower_class:
        # create folder for different attributes
        mk_file(os.path.join(val_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_plankton_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # index of random testing set
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # copy the file distributed to the testing set into corresponded content
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # copy the file distributed to the training set into corresponded content
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
