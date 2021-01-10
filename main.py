from utils.dataloader import load_data
from utils.model import create_model, predict_with_static_image, predict_with_video

class_labels = {
    "Warrior_I": 0,
    "Warrior_II": 1,
    "Tree": 2,
    "Triangle": 3,
    "Standing_Splits": 4
}

data_path = "/Users/nikhilbadami/Pose Estimation/YogiAI/data/"
saved_model_path = "/Users/nikhilbadami/Pose Estimation/YogiAI/saved_models/"

config = {
    "create_model": True,
    "load_model": False,
    "train_model": True,
    "eval_model": True,
    "predict_static": False,
    "predict_video": False,
    "read_pickle": True,
    "save_pickle": True
}

if __name__ == "__main__":
    train_dataset, test_dataset, val_dataset = load_data(config, data_path, class_labels)
    model = None
    if config["create_model"]:
        model = create_model()
    if config["train_model"]:
        model.fit(train_dataset, epochs=100)
    if config["eval_model"]:
        loss, acc = model.evaluate(val_dataset)
        print(f"loss: {loss}\nacc: {acc}")
    if config["predict_static"]:
        predict_with_static_image(model, class_labels, data_path)
    if config["predict_video"]:
        predict_with_video(model, class_labels)
