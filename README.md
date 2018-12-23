# Video-Analysis-Research
A research project to explore new ways to explore and understand a video.
 

## INTRODUCTION:
Deep learning for Video Text Analysis:

Using Deep learning techniques, find a new approach that analyses a video and then present it in understandable language using NLP techniques.

Generating semantic descriptions and knowledge between frames of video to understand the activity relation in them.

# APPROACHES
## 1. Image Captioning of all frames and summarising the complete text.
## 2. Detecting objects in the video and finding a result using YOLO.

### Why is the particular topic chosen? Because this problem is a million-dollar problem at present and is being researched  in many of the premier institutions of the world. 
Solving the visual symbol grounding problem has long been a goal of artificial intelligence. The field appears to be advancing closer to this goal with recent breakthroughs in deep learning for natural language grounding in static images. In this paper, we propose to translate videos directly to sentences using a unified deep neural network with both convolutional and recurrent structure. Described video datasets are scarce, and most existing methods have been applied to toy domains with a small vocabulary of possible words.

For most people, watching a brief video and describing what happened (in words) is an easy task. For machines, extracting the meaning from video pixels and generating natural-sounding language is a very complex problem. Solutions have been proposed for narrow domains with a small set of known actions and objects

## SOLUTION

The idea behind the project is to make an IMAGE CAPTION GENERATING model with a great accuracy. To make an IMAGE CAPTION GENERATING we would need train a NEURAL NETWORK implementing state of the art concept for image processing as well as handling continuous data. While training the model we would first require to extract the features of the training images with the help of CNN (Convolutional Neural Network), once the features are extracted, the word embedding along with the features will be passed through RNN (Recurrent Neural Networks) layers.
After training the model its time to generate the caption given an image. For the getting the best caption out of various captions Beam search would be used and the captions getting best BLEU (Bilingual Evaluation Understudy) score would be the final caption output. The caption we get as output, given an image is then delivered to the user through headphone like device in the form of voice message.  

### We plan to extract features for each frame, mean pool the features across the entire video and input this at every time step to the LSTM network. The LSTM outputs one word at each time step, based on the video features (and the previous word) until it picks the end-of-sentence tag and extends them to generate sentences describing events in videos. These models work by first applying a feature transformation on an image to generate a fixed dimensional vector representation. They then use a sequence model, specifically a Recurrent Neural Network (RNN), to “decode” the vector into a sentence (i.e. a sequence of words). In this work, we plan to apply the same principle of “translating” a visual vector into an English sentence and show that it works well for describing dynamic videos as well as static images


YOLOv3 is extremely fast and accurate. YOLO (You Only Look Once), is a network for object detection. The object detection task consists in determining the location on the image where certain objects are present, as well as classifying those objects. Previous methods for this, like R-CNN and its variations, used a pipeline to perform this task in multiple steps. This can be slow to run and also hard to optimize, because each individual component must be trained separately. YOLO, does it all with a single neural network.
 
After using yolo approach we get much understanding about the content inside the video and we can easily apply our NLP methods for further processing of our model after applying all NLP methods we finally found the description about the video
Whats Ahead:
Image segmentation is a technique to partitioning a digital image into multiple segments from which we can easily extract the main features inside the image by using the various kinds of classifiers we can classify the image features which is useful for detection of our aim results easily. 



## Limitations
As we apply our attention mechanism into the analysis the captioning description is present in poor grammar which can be solve by Natural Language Understanding.
It unable to detect the sentiments inside the video and also the results about the event at the end as the model is not trained with that kind of dataset we can add this kind of feature if that kind of relevant data is available.
Can’t give best results in complex videos in which so many events occurs but describe about the content inside the video file is well. 


