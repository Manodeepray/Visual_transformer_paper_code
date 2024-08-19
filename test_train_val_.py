from shutil import copyfile
import os

def dir_split(data_dir,new_dataset_dir):
    os.makedirs(new_dataset_dir , exist_ok =True)


    train_dir = os.path.join(new_dataset_dir, 'train')
    test_dir = os.path.join(new_dataset_dir , 'test')
    val_dir = os.path.join(new_dataset_dir, 'val')


    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        dir_size = len(os.listdir(class_path))

        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)    
        val_class_dir = os.path.join(val_dir, class_name)    


        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)

        images = os.listdir(class_path)

        train_size = int(len(images)*0.6)
        test_size = int(len(images)*0.8)
        val_size = int(len(images))
        
        for i in range(train_size):
            copyfile(os.path.join(class_path, images[i]), os.path.join(train_class_dir, images[i]))

        for i in range(train_size,test_size):
            copyfile(os.path.join(class_path, images[i]), os.path.join(test_class_dir, images[i]))

        for i in range(test_size,val_size):
            copyfile(os.path.join(class_path, images[i]), os.path.join(val_class_dir, images[i]))





    return train_dir , test_dir , val_dir

split_dataset = 'test_train_dataset'

dataset = 'dataset'
train_dir , test_dir , val_dir = dir_split(dataset , split_dataset)