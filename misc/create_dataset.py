import urllib.request
import os
from sklearn.model_selection import train_test_split

# Names of classes in yoga-82 text files
class_names_82 = [
    # "Warrior_I_Pose_or_Virabhadrasana_I_",
    # "Warrior_II_Pose_or_Virabhadrasana_II_",
    # "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_",
    # "Tree_Pose_or_Vrksasana_",
    "Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_"
]

# Names of classes in my dataset
class_names = [
    # "Warrior_I",
    # "Warrior_II",
    # "Triangle",
    # "Tree",
    "Standing_Splits"
]

Yoga_82_path = "path/to/yoga82"
dataset_path = "path/to/data/dir"
intermediary_path = "path/to/intermediate/dir"

if __name__ == "__main__":
    for i in range(len(class_names_82)):
        counter = 0
        class_name = class_names[i]
        name_82 = class_names_82[i]
        intermediary_file_name = intermediary_path + class_name
        try:
            os.mkdir(intermediary_file_name)
        except:
            pass

        url_file = Yoga_82_path + "/" + name_82 + ".txt"
        filenames = os.listdir(os.getcwd() + "/" + class_name)
        with open(url_file) as f:
            d = f.readline()
            while d:
                url = d.split('\t')[1].strip('\n')
                filename = class_name + '_' + str(counter) + ".jpg"
                try:
                    urllib.request.urlretrieve(url, intermediary_file_name + "/" + filename)
                    filenames.append(filename)
                except:
                    pass
                counter += 1
                d = f.readline()

        print("Read " + str(counter) + " files\n")

        # Randomly split dataset into train, test and validation using 60-20-20 split
        train_val, test = train_test_split(filenames, test_size=0.2, train_size=0.8)
        train, val = train_test_split(train_val, test_size=0.25, train_size=0.75)

        # Move data from intermediary file to dataset dir
        for filename in train:
            os.replace(intermediary_file_name + "/" + filename, dataset_path + class_name + "/Train/" + filename)

        for filename in test:
            os.replace(intermediary_file_name + "/" + filename, dataset_path + class_name + "/Test/" + filename)

        for filename in val:
            os.replace(intermediary_file_name + "/" + filename, dataset_path + class_name + "/Validation/" + filename)

        print("Finished reading files for: ", class_name)
