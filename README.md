DialogueINAB
=============
The code and dataset for paper "DialogueINAB: An interaction network between interlocutors’ attitude and behavior for dialogue emotion detection"

Datasets
============
1.IEMOCAP

This dataset, collected and released by the SAIL lab at the University of Southern California, contains binary conversational texts conducted by ten interlocutors in pairs. The top 8 interlocutors from sessions 1 to 4 of this dataset serve as the training set. The dataset contains 152 dialogues with a total of 7433 utterances and 6 emotion categories: happy, sad, neutral, angry, excited, and frustrated. 
  
2.MELD

The multimodal dataset MELD is an extension of the dataset text EmotionLines. MELD contains more than 1400 multi-party dialogues and more than 13000 utterances from the TV series Friends, which includes seven emotion labels, i.e., happiness/joy, anger, fear, disgust, sadness, surprise, and neutrality. 
  
3.AVEC 
This dataset is based on the dataset SEMAINE, and the dialogue text we need is taken from his human-computer interaction video data. All dialogs are annotated by four real-valued sentiment attributes, including arousal, expectancy, valence, and power, and comments are available every 0.2 seconds in the original database. Nevertheless, to tailor the annotation to our requirements, we calculate the mean of the attribute values for a sentence to get the utterance-level attribute.
  
Environment
===========
* Python=3.8
* Pytorch>=1.7.1
* pandas>=1.2.4
* numpy>=1.20.1
* torch-geometric==2.0.4
* torch-scatter==2.0.7
* scikit-learn==0.24.1

Usage
============
1.Unpack the dataset files in data folder.

2.Put the data folder into the code folder.

3.acorrding to the dataset task, modify the dataset path in the file run_train.py and run run_train.py
