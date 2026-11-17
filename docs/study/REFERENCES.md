# References

Every link was requested on 2026-11-16 and answered, and every arXiv identifier was
checked against the paper's title, except the entries marked †: those hosts refused
or timed out for an automated request, so the address is given from memory and should
be confirmed in a browser.

Each entry says which chapters it supports.

## Textbooks and courses

| Reference | Use it for | Chapters |
|---|---|---|
| K. J. Åström and R. M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*, 2nd ed. Free online: <https://fbswiki.org/wiki/index.php/Feedback_Systems:_An_Introduction_for_Scientists_and_Engineers> | Feedback, first-order systems, frequency-domain stability, delay | 6 |
| K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*, 2017. Free online: <https://hades.mech.northwestern.edu/index.php/Modern_Robotics> | Frames, kinematics, wheeled mobile robots | 3 |
| R. Rajamani, *Vehicle Dynamics and Control*, 2nd ed., Springer, 2012. <https://link.springer.com/book/10.1007/978-1-4614-1433-9> | What a dynamic vehicle model adds to the kinematic one | 3, 14 |
| R. Tedrake, *Underactuated Robotics* (MIT course notes). <https://underactuated.mit.edu/> | A deeper treatment of dynamics and control | 6 |
| A. Sakai et al., *PythonRobotics*. <https://atsushisakai.github.io/PythonRobotics/> · <https://github.com/AtsushiSakai/PythonRobotics> | Small runnable implementations of pure pursuit, bicycle models and more | 3, 4 |

## Autonomous driving and path tracking

| Reference | Use it for | Chapters |
|---|---|---|
| R. C. Coulter, *Implementation of the Pure Pursuit Path Tracking Algorithm*, CMU-RI-TR-92-01, 1992. <https://www.ri.cmu.edu/publications/implementation-of-the-pure-pursuit-path-tracking-algorithm/> | The geometry behind `pursuit_steer` | 4, 6 |
| B. Paden, M. Čáp, S. Z. Yong, D. Yershov and E. Frazzoli, *A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles*, 2016. <https://arxiv.org/abs/1604.07446> | Vehicle models (§III) and path-tracking controllers (§V) in context | 1, 3, 4, 6 |
| J. Kong, M. Pfeiffer, G. Schildbach and F. Borrelli, *Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design*, IEEE Intelligent Vehicles Symposium, 2015. † <https://ieeexplore.ieee.org/document/7225830> | When the kinematic bicycle model is good enough | 3 |
| CARLA simulator. <https://carla.org/> · leaderboard: <https://leaderboard.carla.org/> | A full driving simulator and an example of closed-loop evaluation at scale | 11, 14 |
| RoboRacer (formerly F1TENTH). <https://roboracer.ai/> | A community and course around 1:10 autonomous cars | 14 |

## Imitation learning and behaviour cloning

| Reference | Use it for | Chapters |
|---|---|---|
| D. A. Pomerleau, *ALVINN: An Autonomous Land Vehicle in a Neural Network*, NeurIPS 1988. <https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html> | The first neural network to steer a vehicle | 8 |
| S. Ross and J. A. Bagnell, *Efficient Reductions for Imitation Learning*, AISTATS 2010. <https://proceedings.mlr.press/v9/ross10a.html> | Why behaviour cloning's error grows with the square of the horizon | 8, 11 |
| S. Ross, G. J. Gordon and J. A. Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011 (DAgger). <https://arxiv.org/abs/1011.0686> | Training on the states the learner visits | 8 |
| M. Laskey, J. Lee, R. Fox, A. Dragan and K. Goldberg, *DART: Noise Injection for Robust Imitation Learning*, CoRL 2017. <https://arxiv.org/abs/1703.09327> | The method `collect` uses | 8, 9 |
| M. Bojarski et al., *End to End Learning for Self-Driving Cars*, 2016 (PilotNet). <https://arxiv.org/abs/1604.07316> | The network shape `ChunkNet` follows; off-centre cameras as recovery data | 8, 10 |
| F. Codevilla, E. Santana, A. M. López and A. Gaidon, *Exploring the Limitations of Behavior Cloning for Autonomous Driving*, ICCV 2019. <https://arxiv.org/abs/1904.08980> | Offline metrics against driving performance | 8, 11 |
| T. Osa et al., *An Algorithmic Perspective on Imitation Learning*, 2018. <https://arxiv.org/abs/1811.06711> | A survey of the field | 8 |
| Ornstein–Uhlenbeck process, Wikipedia. † <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process> | The disturbance model in `collect` | 8 |

## Action chunking and learned visuomotor policies

| Reference | Use it for | Chapters |
|---|---|---|
| T. Z. Zhao, V. Kumar, S. Levine and C. Finn, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023 (ACT). <https://arxiv.org/abs/2304.13705> | Action chunking and temporal ensembling | 5, 10 |
| C. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023. <https://arxiv.org/abs/2303.04137> | Generating action sequences by denoising | 10, 13 |
| K. Black, M. Y. Galliker and S. Levine, *Real-Time Execution of Action Chunking Flow Policies*, 2025. <https://arxiv.org/abs/2506.07339> | Real-time chunking: the remedy for slow models | 5, 13, 14 |
| LeRobot's RTC module. <https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/rtc> | An implementation of the above | 13 |

## Vision-language-action models

| Reference | Use it for | Chapters |
|---|---|---|
| M. Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, 2025. <https://arxiv.org/abs/2506.01844> | The model this project aims at | 5, 13 |
| SmolVLA, Hugging Face blog. <https://huggingface.co/blog/smolvla> | An accessible introduction | 13 |
| SmolVLA in the LeRobot documentation. <https://huggingface.co/docs/lerobot/smolvla> | Fine-tuning and running it | 13 |
| `lerobot/smolvla_base` model card. <https://huggingface.co/lerobot/smolvla_base> | The base checkpoint | 13, 14 |
| SmolVLM2 backbone model card. <https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct> | The VLM inside SmolVLA | 13 |
| K. Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. <https://arxiv.org/abs/2410.24164> | Flow-matching action experts | 13 |
| Physical Intelligence, *π0.5: a Vision-Language-Action Model with Open-World Generalization*, 2025. <https://arxiv.org/abs/2504.16054> · openpi: <https://github.com/Physical-Intelligence/openpi> | A larger VLA, benchmarked on the Jetson | 13, 14 |
| M. J. Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024. <https://arxiv.org/abs/2406.09246> | Tokenised actions | 13 |
| A. Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023. <https://arxiv.org/abs/2307.15818> | The origin of the term VLA | 13 |

## LeRobot

| Reference | Use it for | Chapters |
|---|---|---|
| LeRobot repository. <https://github.com/huggingface/lerobot> | Source: datasets, policies, training | 13 |
| LeRobot documentation. <https://huggingface.co/docs/lerobot/index> | Installation, training, robots | 13 |
| LeRobotDataset v3. <https://huggingface.co/docs/lerobot/lerobot-dataset-v3> | The dataset format for the proposed conversion | 9, 13 |
| Asynchronous inference. <https://huggingface.co/docs/lerobot/async> | LeRobot's own answer to slow, chunked policies | 2, 5, 13 |

## ROS 2

| Reference | Use it for | Chapters |
|---|---|---|
| ROS 2 Humble documentation. <https://docs.ros.org/en/humble/index.html> | Everything | 12 |
| Tutorials. <https://docs.ros.org/en/humble/Tutorials.html> | Nodes, topics, parameters, hands-on | 12 |
| Concepts: topics. <https://docs.ros.org/en/humble/Concepts/Basic/About-Topics.html> | Publish and subscribe | 12 |
| Concepts: parameters. <https://docs.ros.org/en/humble/Concepts/Basic/About-Parameters.html> | Node configuration | 12 |
| Concepts: quality of service. <https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html> | Queue depth, reliability | 12 |
| Launch tutorials. <https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html> | Launch files | 12 |
| `sensor_msgs`. <https://docs.ros.org/en/humble/p/sensor_msgs/> · `geometry_msgs`. <https://docs.ros.org/en/humble/p/geometry_msgs/> | The message types on `/drive/*` | 12 |
| REP 103, *Standard Units of Measure and Coordinate Conventions*. <https://www.ros.org/reps/rep-0103.html> | The conventions `/drive/cmd` departs from | 12 |
| REP 117, *Informational Distance Measurements*. <https://www.ros.org/reps/rep-0117.html> | How range sensors report "nothing" and "too close" | 7, 12 |

## NVIDIA Jetson and TensorRT

| Reference | Use it for | Chapters |
|---|---|---|
| jetson-orin-nano-vla. <https://github.com/eetmie/jetson-orin-nano-vla> | Exporting and running VLAs on the Orin Nano; the source of every Jetson latency figure in this guide | 4, 13, 14 |
| Jetson Orin Nano developer kit, getting started. <https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit> | Setting up the board | 14 |
| Jetson Orin Nano Super developer kit. <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/> | The target hardware | 14 |
| JetPack SDK. <https://developer.nvidia.com/embedded/jetpack> | The board's software stack | 14 |
| TensorRT documentation. <https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html> | Engines, precision, building | 14 |
| ONNX. <https://onnx.ai/> | The exchange format between training and TensorRT | 14 |
| NVIDIA Container Toolkit, installation. <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html> | Required by `make build` | 15 |

## Tools used by the code

| Reference | Use it for | Chapters |
|---|---|---|
| NumPy for absolute beginners. <https://numpy.org/doc/stable/user/absolute_beginners.html> | Arrays, shapes, indexing | 0 |
| `numpy.savez_compressed`. <https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html> | The episode file format | 9 |
| PyTorch, Learn the Basics. <https://pytorch.org/tutorials/beginner/basics/intro.html> | Modules, autograd, training loops | 0, 10 |
| `torch.nn.SmoothL1Loss`. <https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html> | The training loss | 10 |
| `CosineAnnealingLR`. <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html> | The learning-rate schedule | 10 |
| OpenCV tutorials. † <https://docs.opencv.org/4.x/d9/df8/tutorial_root.html> | `resize`, `remap`, drawing | 3, 9 |
| Python `threading`. <https://docs.python.org/3/library/threading.html> | `Condition`, used by `ThreadedWorker` | 2 |
| Docker Compose. <https://docs.docker.com/compose/> | How the `make` targets run | 15 |
| Mermaid. <https://mermaid.js.org/> | The diagrams in this guide | — |

## Inside this repository

| File | What it is |
|---|---|
| [`README.md`](../../README.md) | What the project is, and what is not there yet |
| [`docs/driving.md`](../driving.md) | How to run everything; the measurements quoted here as *repo* |
| [`tests/`](../../tests) | The executable specification: 81 tests |
| `runs/drive/reports/*.json`, `runs/drive/models/*.json` | Evaluation and training reports (local, not in git) |
