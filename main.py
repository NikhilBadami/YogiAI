from utils.dataloader import load_data
from utils.model import create_model, predict_with_static_image, predict_with_video, train_model
from tensorflow import keras

class_labels = {
    "Warrior_I": 0,
    "Warrior_II": 1,
    "Tree": 2,
    "Triangle": 3,
    "Standing_Splits": 4
}

data_path = "/Users/nikhilbadami/Pose Estimation/YogiAI/data/"

config = {
    "create_model": True,
    "load_model": True,
    "train_model": False,
    "eval_model": True,
    "predict_static": False,
    "predict_video": True,
    "read_pickle": True,
    "save_pickle": True,
    "display_stats": False
}

if __name__ == "__main__":
    train_dataset, test_dataset, val_dataset = load_data(config, data_path, class_labels)
    model = None
    if config["create_model"]:
        model = create_model()
    if config["load_model"]:
        model = keras.models.load_model("/Users/nikhilbadami/Pose Estimation/YogiAI/saved_models/")
    if config["train_model"]:
        train_model(model, config, train_dataset, val_dataset)
    if config["eval_model"]:
        loss, acc = model.evaluate(test_dataset)
        print(f"loss: {loss}\nacc: {acc}")
    if config["predict_static"]:
        predict_with_static_image(model, class_labels, data_path)
    if config["predict_video"]:
        predict_with_video(model, class_labels)
