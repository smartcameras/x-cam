# Cross Camera Visual Overlap Recognition

Official code of the paper "Cross-Camera Visual Overlap Recognition".


## Installation

### Requirements
* C++ (GNU 7.5)
* ZeroMQ ()
* OpenCV (3.4.1)
* Boost Library (1.65)
* DLib
* DBoW2

### Tested on

Tested on Linux machine with Ubuntu 18.04 LTS.

### Instructions





---

### Arguments

The software can be run by passing additional arguments. The arguments have pre-defined positions. The minimum number of parameters is 6. The maximum number is 19. 
The software does not run if the number of arguments is less than the minimum or higher than the maximum, and a warning and help is provided.
The arguments are listed below according to their position along with their brief explanation and default values. 

1. agent_id: the unique identifier of the agent/camera (e.g., 1)
2. rcv_tcp_port: the port number to set in ZMQ for the receiver (request-reply mode). Default: 5555.
3. snd_tcp_port: the port number to set in ZMQ for the sender. Default: 5556.
4. videopath: . Default: "".
5. respath: . Default: "../results/";
6. strVocFile: . Default: "./Vocabulary/ORBvoc.txt";
7. num_max_frames: the maximum number of frames to process in a recorded video. Default: -1 (no max number, the whole image sequence is processed).
8. max_num_kps: the maximum target number of keypoints that can be localised in an image. Default: 1000.
9. matching_th: threshold when matching binary local features. Default: -1. 
10. b_snn: boolean to use Lowe's ratio test to filter out ambiguous matches for binary features. The test defines a distance ratio between the closest and the second closest binary feature (or second nearest neighbour, SNN). Deafult: false.
11. snn_th: threshold on Lowe's ratio test. Matches whose test is lower than this threshold are discarded. When the threshold is lower (e.g., 0.6), the test is more restrictive, enforcing a larger distance between the first and second closest neighbours for a query binary feature (fewer matches). When the threshold is higher (e.g., 0.8), the test is more permissive, allowing more matches that can be also erroneous. The recommended value is between 0.6 and 0.8. Default: 0.6.
12. gf_type: DBoW, NetVLAD, DeepBit. Default: DBoW.
13. gf_dim: the dimensionality of the global feature (vector representing the whole image). Default: -1. 
14. featpath: . Default: "../feats/backyard/view1/netvlad/netvlad_feats.txt";
15. collaborative_mode: . Default: true.
16. init_wnd: Number of frames for initialisation window. Default: 30.
17. feat_sharing_freq: Frequency of sharing features: Default: 5.
18. synch_mode: . Default: false.
19. vprcallfilename: . Default: "";

### Outputs

Each agent (camera) is outputting three files in .txt format, and :
1. agent(ID)\_vpr\_res.txt
2. running\_times\_agent(ID).txt
3. running\_times\_VPR\_agent(ID).txt

The first output file (agent(ID)\_vpr\_res.txt) provides a summary of the detection results. The file contains the following columns:
- Index of the current processed frame 
- Index of the frame from which features has been shared (i.e., the current frame index in this implementation)
- Index of the other agent/camera
- Index of the current processed frame by the other camera/agent
- Index of the the detected overlapping frame in the other camera
- Number of matched (ORB) binary features

## Credits

## Enquiries, Question and Comments

If you have any further enquiries, question, or comments, please contact <email>a.xompero@qmul.ac.uk</email>. 
If you would like to file a bug report or a feature request, use the Github issue tracker. 


## Licence

This work is licensed under the MIT License.  To view a copy of this license, see
[LICENSE](LICENSE).

