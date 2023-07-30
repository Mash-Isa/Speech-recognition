# Speech Recognition

This project aims to demonstrate an audio-based speech recognition system using deep learning techniques. The model is trained to classify spoken audio commands into different classes corresponding to specific voice commands. The project uses the Keras library and Convolutional Neural Networks (CNNs) for audio processing and classification.

## Folder Structure

- `app.py`: This file contains a Streamlit web application that allows users to interact with the trained speech recognition model by capturing voice commands and displaying the predicted class label on the screen.
- `model.h5`: This file stores the pre-trained CNN model used for speech recognition.
- `requirements.txt`: This file lists all the required Python packages and their versions needed to run the application.
- `Speech-recognition.ipynb`: This Jupyter Notebook provides the step-by-step process of data preparation, model building, training, and evaluation of the speech recognition system.
- `training.py`: This script contains the code used for training the speech recognition model.

## Installation

To run the Chess Position Prediction App locally, follow these steps:

1. Clone this repository: `git clone https://github.com/Mash-Isa/CV-Chess-Position-Prediction.git`
2. Change into the project directory: `cd chess-position-prediction`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Streamlit app: `streamlit run app.py`

## How to Use

1. Run the Streamlit app for the speech recognition project, and a user interface will open in your web browser.
2. To interact with the app, use your microphone to capture voice commands.
3. Click on the "Record" button to start capturing your voice command, and then click on "Stop" when you're done speaking.
4. The app will process the recorded audio and display the predicted class label corresponding to the recognized voice command.

## Acknowledgments

The development of this project was inspired by various resources from the Kaggle community and other open-source contributors. Some of the references include:

- [Speech Representation and Data Exploration](https://www.kaggle.com/code/davids1992/speech-representation-and-data-exploration)
- [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/code/leangab/tensorflow-speech-recognition-challenge)
- [Voice Command Detection](https://www.kaggle.com/code/araspirbadian/voice-command-detection/notebook)
- [Speech Recognition](https://www.kaggle.com/code/fathykhader/speech-recognition)

Happy speech recognition! 🎤🚀
