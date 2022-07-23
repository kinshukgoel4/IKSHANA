Development isn’t a fraction but a benefit of whole. Life of a visually impaired person is in no way easy. The inability to be aware of one’s surroundings and recognizing objects plays a huge impact on the quality of life they lead. It takes more time for them to understand the things in front of them. The absence of basic object recognition in a person’s life compromises their safety. To create an affordable and minimalistic gadget which will scan for objects in real time using a camera 
module and offer voice instruction to the user via an output speaker.It is capable of detecting objects within a certain pre-established range and is reliable for ensuring the safety of the user.Thus, we provide a secure and compact device for the visually impaired which is not only cost 
effective but also helps them recognize certain objects without physically touching them

Basic idea:
The idea is to create a device that detects the objects and gives audio output in real time. 

Components used:-
1. Pi zero w only  board
2. Pi Camera
3. Camera module
4. SD card
5. Li polymer battery
6. Jumper wires 30 30
7. Case(card board/plastic)


Working:
The following procedure that happens in the running loop of detecting objects is as follows: 
Step 1: Taking the video input When the power supply is started, the Pi camera 
starts recording the video. This video is taken as input using Video capture function in the open 
CV package and running frames are captured by setting the Frame width, height and buffer size.
Step 2: Detecting the objects In this project a pre-trained object detection 
model- “ssd_mobilenet_v3_large”, trained on coco dataset is used. The weights that are used 
is “frozen _inference graph. pb”. A class name list is created which contains class 
file consisting of the coco names. SSD mobile net model is taken as configuration path and 
frozen graph is taken as weights path. Using open CV detection model function with weight 
path and configuration path as parameters object detection in the running frames is done.
When an object is detected successfully, it compares which is the suitable match for the 
detected object in the trained dataset and matches if the percentage is above 60% or the 
coco name which has highest match percentage.
Step 3: Giving AudioOutput
After the object is determined, the name of the object is taken from the class name list and is 
appended as text and given as audio output which comes from a speaker which is connected through bluetooth
