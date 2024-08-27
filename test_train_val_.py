from shutil import copyfile
import os
import pandas as pd
import shutil

def class_splitter(csv_path , images_dir , split_dataset):
    df = pd.read_csv(csv_path)
    list_dirs = df.columns
    list_dirs = list(list_dirs)

    image_paths = []
    split_dataset = 'class_split_dataset'
    images_dir = "images_dataset"



    os.makedirs(split_dataset , exist_ok =True)
    x = df.iloc[:,0].values
    y = df.iloc[:,1:].values


    for item in list_dirs:
        print(item)
        os.makedirs(os.path.join(split_dataset,str(item)) , exist_ok = True)







    for i in range(len(df)):
        image_path = os.path.join(images_dir,str(x[i])+".jpg")
        if df['MEL'][i] == 1 :
            new_path = os.path.join(split_dataset,"MEL",str(x[i])+".jpg")
            shutil.copyfile(image_path, new_path)
        elif df['NV'][i] == 1 :
            new_path = os.path.join(split_dataset,"NV",str(x[i])+".jpg")
            shutil.copyfile(image_path, new_path)
        elif df['BCC'][i] == 1 :
            new_path = os.path.join(split_dataset,"BCC",str(x[i])+".jpg")
            shutil.copyfile(image_path, new_path)
        elif df['AKIEC'][i] == 1 :
            new_path = os.path.join(split_dataset,"AKIEC",str(x[i])+".jpg")
            shutil.copyfile(image_path, new_path)
        elif df['BKL'][i] == 1 :
            new_path = os.path.join(split_dataset,"BKL",str(x[i])+".jpg")
            shutil.copyfile(image_path, new_path)
        elif df['DF'][i] == 1 :
            new_path = os.path.join(split_dataset,"DF",str(x[i])+".jpg")
            shutil.copyfile(image_path, new_path)
        elif df['VASC'][i] == 1 :
            new_path = os.path.join(split_dataset,"VASC",str(x[i])+".jpg")
            shutil.copyfile(image_path, new_path)
        
    
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



    os.rmdir(os.path.join(train_dir,"image"))
    os.rmdir(os.path.join(test_dir,"image"))
    os.rmdir(os.path.join(val_dir,"image"))
    
    
    return train_dir , test_dir , val_dir

def main():
    csv_path = "/GroundTruth.csv"
    split_dataset = 'class_split_dataset'
    images_dir = "images_dataset"

    class_splitter(csv_path=csv_path,images_dir=images_dir , split_dataset= split_dataset)

    split_dataset = 'test_train_dataset'

    dataset = 'class_split_dataset'
    train_dir , test_dir , val_dir = dir_split(dataset , split_dataset)


if __name__== "__main__":
    main()