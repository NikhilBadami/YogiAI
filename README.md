# YogiAI
### Installation
Create a virtual environment using either conda or venv. Install
the environment using `pip install requirements.txt`

Follow the instructions here: https://google.github.io/mediapipe/getting_started/python.html To install mediapipe

### Dataset
This project uses the Yoga82 dataset which can be found here: https://neurohive.io/en/news/yoga-82-new-dataset-with-complex-yoga-poses/
Follow the instructions to download and then use `create_dataset.py` to download and split the dataset.
By default, only the Warrior_I, Warrior_II, Tree, Triangle and StandingSplits datasets are loaded

### Running the  net
In `main.py` modify the `config` dictionary to specify which features you want to execute.
Then run `python main.py` from the command line

### References
1.	https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1932&context=etd_projects
2.	http://cs230.stanford.edu/projects_winter_2019/reports/15813480.pdf
3.	https://www.amarchenkova.com/2018/03/25/convolutional-neural-network-yoga-poses/
4.	https://neurohive.io/en/news/yoga-82-new-dataset-with-complex-yoga-poses/
5. https://arxiv.org/abs/2004.10362