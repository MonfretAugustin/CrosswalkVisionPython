# CrosswalkVisionPython

CrosswalkVisionPython propose a PyTorch algorithm to predict the position of the entry and the exit of crosswalk.
This project is inspired by the project [ImVisible](https://github.com/samuelyu2002/ImVisible "ImVisible") also available on github. This project was created for a CentraleSupelec classes.

## How to Install and Run the Project

The file `requirements.txt` list all the package necessary to run the project. Also you will have to download all the dataset and put it inside the dataset folder which is currently empty.
You can download the dataset with those two links: [Part 1](https://dl.orangedox.com/KrVSsK) and [Part 2](https://dl.orangedox.com/CMjgtX) (23Go).

## How to use the project

The file `Model/train.py` is used to train the model, and the file `Model/test.py` is used to check the perfomance of the train model.
The file `Model/NeuralNetwork.py` encode the model in PyTorch.
The folder `TrainingModel` save the model at different epoch and also some charts.

Finally, the file `CoreML.py` is used to convert the PyTorch model into a CoreML model.
