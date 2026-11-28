# Tài liệu tham khảo

*Bản tiếng Việt · [English](REFERENCES.md)*

*Bản tiếng Việt · [English](REFERENCES.vi.md)*

Mọi liên kết đã được truy cập ngày 2026-11-16 và có phản hồi, và mọi mã định danh arXiv đã được
đối chiếu với tiêu đề bài báo, trừ các mục đánh dấu †: các máy chủ đó từ chối hoặc hết thời gian
chờ với một yêu cầu tự động, nên địa chỉ được ghi theo trí nhớ và nên được xác nhận lại trên trình
duyệt.

Mỗi mục ghi rõ nó hỗ trợ những chương nào.

## Sách giáo khoa và khoá học

| Tài liệu | Dùng để | Chương |
|---|---|---|
| K. J. Åström và R. M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*, tái bản lần 2. Đọc miễn phí: <https://fbswiki.org/wiki/index.php/Feedback_Systems:_An_Introduction_for_Scientists_and_Engineers> | Phản hồi, hệ bậc một, ổn định trong miền tần số, độ trễ | 6 |
| K. M. Lynch và F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*, 2017. Đọc miễn phí: <https://hades.mech.northwestern.edu/index.php/Modern_Robotics> | Hệ toạ độ, động học, robot di động có bánh | 3 |
| R. Rajamani, *Vehicle Dynamics and Control*, tái bản lần 2, Springer, 2012. <https://link.springer.com/book/10.1007/978-1-4614-1433-9> | Mô hình động lực học của xe thêm được gì so với mô hình động học | 3, 14 |
| R. Tedrake, *Underactuated Robotics* (bài giảng của MIT). <https://underactuated.mit.edu/> | Một cách trình bày sâu hơn về động lực học và điều khiển | 6 |
| A. Sakai và cộng sự, *PythonRobotics*. <https://atsushisakai.github.io/PythonRobotics/> · <https://github.com/AtsushiSakai/PythonRobotics> | Các bản cài đặt nhỏ, chạy được, của pure pursuit, mô hình xe đạp và nhiều thứ khác | 3, 4 |

## Lái tự động và bám quỹ đạo

| Tài liệu | Dùng để | Chương |
|---|---|---|
| R. C. Coulter, *Implementation of the Pure Pursuit Path Tracking Algorithm*, CMU-RI-TR-92-01, 1992. <https://www.ri.cmu.edu/publications/implementation-of-the-pure-pursuit-path-tracking-algorithm/> | Hình học đằng sau `pursuit_steer` | 4, 6 |
| B. Paden, M. Čáp, S. Z. Yong, D. Yershov và E. Frazzoli, *A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles*, 2016. <https://arxiv.org/abs/1604.07446> | Các mô hình xe (§III) và bộ điều khiển bám quỹ đạo (§V) trong bối cảnh chung | 1, 3, 4, 6 |
| J. Kong, M. Pfeiffer, G. Schildbach và F. Borrelli, *Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design*, IEEE Intelligent Vehicles Symposium, 2015. † <https://ieeexplore.ieee.org/document/7225830> | Khi nào mô hình xe đạp động học là đủ tốt | 3 |
| Bộ mô phỏng CARLA. <https://carla.org/> · bảng xếp hạng: <https://leaderboard.carla.org/> | Một bộ mô phỏng lái xe đầy đủ, và một ví dụ về đánh giá vòng kín ở quy mô lớn | 11, 14 |
| RoboRacer (trước đây là F1TENTH). <https://roboracer.ai/> | Một cộng đồng và khoá học xoay quanh xe tự hành tỉ lệ 1:10 | 14 |

## Học bắt chước và behaviour cloning

| Tài liệu | Dùng để | Chương |
|---|---|---|
| D. A. Pomerleau, *ALVINN: An Autonomous Land Vehicle in a Neural Network*, NeurIPS 1988. <https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html> | Mạng nơ-ron đầu tiên lái một chiếc xe | 8 |
| S. Ross và J. A. Bagnell, *Efficient Reductions for Imitation Learning*, AISTATS 2010. <https://proceedings.mlr.press/v9/ross10a.html> | Vì sao sai số của behaviour cloning tăng theo bình phương độ dài chân trời | 8, 11 |
| S. Ross, G. J. Gordon và J. A. Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011 (DAgger). <https://arxiv.org/abs/1011.0686> | Train trên những state mà người học tự đi tới | 8 |
| M. Laskey, J. Lee, R. Fox, A. Dragan và K. Goldberg, *DART: Noise Injection for Robust Imitation Learning*, CoRL 2017. <https://arxiv.org/abs/1703.09327> | Phương pháp mà `collect` dùng | 8, 9 |
| M. Bojarski và cộng sự, *End to End Learning for Self-Driving Cars*, 2016 (PilotNet). <https://arxiv.org/abs/1604.07316> | Hình dạng mạng mà `ChunkNet` đi theo; camera lệch tâm làm dữ liệu phục hồi | 8, 10 |
| F. Codevilla, E. Santana, A. M. López và A. Gaidon, *Exploring the Limitations of Behavior Cloning for Autonomous Driving*, ICCV 2019. <https://arxiv.org/abs/1904.08980> | Chỉ số offline so với chất lượng lái thực tế | 8, 11 |
| T. Osa và cộng sự, *An Algorithmic Perspective on Imitation Learning*, 2018. <https://arxiv.org/abs/1811.06711> | Một bài tổng quan về lĩnh vực | 8 |
| Quá trình Ornstein–Uhlenbeck, Wikipedia. † <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process> | Mô hình nhiễu trong `collect` | 8 |

## Action chunking và các policy thị giác–vận động được học

| Tài liệu | Dùng để | Chương |
|---|---|---|
| T. Z. Zhao, V. Kumar, S. Levine và C. Finn, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023 (ACT). <https://arxiv.org/abs/2304.13705> | Action chunking và temporal ensembling | 5, 10 |
| C. Chi và cộng sự, *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023. <https://arxiv.org/abs/2303.04137> | Sinh chuỗi action bằng cách khử nhiễu | 10, 13 |
| K. Black, M. Y. Galliker và S. Levine, *Real-Time Execution of Action Chunking Flow Policies*, 2025. <https://arxiv.org/abs/2506.07339> | Real-time chunking: cách chữa cho các model chậm | 5, 13, 14 |
| Module RTC của LeRobot. <https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/rtc> | Một bản cài đặt của bài trên | 13 |

## Các model vision-language-action

| Tài liệu | Dùng để | Chương |
|---|---|---|
| M. Shukor và cộng sự, *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, 2025. <https://arxiv.org/abs/2506.01844> | Model mà dự án này hướng tới | 5, 13 |
| SmolVLA, blog của Hugging Face. <https://huggingface.co/blog/smolvla> | Một bài giới thiệu dễ tiếp cận | 13 |
| SmolVLA trong tài liệu LeRobot. <https://huggingface.co/docs/lerobot/smolvla> | Fine-tune và chạy nó | 13 |
| Model card của `lerobot/smolvla_base`. <https://huggingface.co/lerobot/smolvla_base> | Checkpoint gốc | 13, 14 |
| Model card của backbone SmolVLM2. <https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct> | VLM bên trong SmolVLA | 13 |
| K. Black và cộng sự, *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. <https://arxiv.org/abs/2410.24164> | Các action expert dùng flow matching | 13 |
| Physical Intelligence, *π0.5: a Vision-Language-Action Model with Open-World Generalization*, 2025. <https://arxiv.org/abs/2504.16054> · openpi: <https://github.com/Physical-Intelligence/openpi> | Một VLA lớn hơn, đã được đo trên Jetson | 13, 14 |
| M. J. Kim và cộng sự, *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024. <https://arxiv.org/abs/2406.09246> | Action được token hoá | 13 |
| A. Brohan và cộng sự, *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023. <https://arxiv.org/abs/2307.15818> | Nguồn gốc của thuật ngữ VLA | 13 |

## LeRobot

| Tài liệu | Dùng để | Chương |
|---|---|---|
| Repository LeRobot. <https://github.com/huggingface/lerobot> | Mã nguồn: bộ dữ liệu, policy, train | 13 |
| Tài liệu LeRobot. <https://huggingface.co/docs/lerobot/index> | Cài đặt, train, robot | 13 |
| LeRobotDataset v3. <https://huggingface.co/docs/lerobot/lerobot-dataset-v3> | Định dạng bộ dữ liệu cho phép chuyển đổi được đề xuất | 9, 13 |
| Suy luận bất đồng bộ. <https://huggingface.co/docs/lerobot/async> | Câu trả lời riêng của LeRobot cho các policy chậm, chạy theo chunk | 2, 5, 13 |

## ROS 2

| Tài liệu | Dùng để | Chương |
|---|---|---|
| Tài liệu ROS 2 Humble. <https://docs.ros.org/en/humble/index.html> | Mọi thứ | 12 |
| Các bài hướng dẫn. <https://docs.ros.org/en/humble/Tutorials.html> | Node, topic, tham số, thực hành | 12 |
| Khái niệm: topic. <https://docs.ros.org/en/humble/Concepts/Basic/About-Topics.html> | Publish và subscribe | 12 |
| Khái niệm: tham số. <https://docs.ros.org/en/humble/Concepts/Basic/About-Parameters.html> | Cấu hình node | 12 |
| Khái niệm: chất lượng dịch vụ (QoS). <https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html> | Độ sâu hàng đợi, độ tin cậy | 12 |
| Hướng dẫn launch. <https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html> | File launch | 12 |
| `sensor_msgs`. <https://docs.ros.org/en/humble/p/sensor_msgs/> · `geometry_msgs`. <https://docs.ros.org/en/humble/p/geometry_msgs/> | Các kiểu message trên `/drive/*` | 12 |
| REP 103, *Standard Units of Measure and Coordinate Conventions*. <https://www.ros.org/reps/rep-0103.html> | Các quy ước mà `/drive/cmd` không theo | 12 |
| REP 117, *Informational Distance Measurements*. <https://www.ros.org/reps/rep-0117.html> | Cách cảm biến khoảng cách báo "không có gì" và "quá gần" | 7, 12 |

## NVIDIA Jetson và TensorRT

| Tài liệu | Dùng để | Chương |
|---|---|---|
| jetson-orin-nano-vla. <https://github.com/eetmie/jetson-orin-nano-vla> | Export và chạy VLA trên Orin Nano; nguồn của mọi con số độ trễ trên Jetson trong giáo trình này | 4, 13, 14 |
| Bộ kit phát triển Jetson Orin Nano, bắt đầu. <https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit> | Cài đặt board | 14 |
| Bộ kit phát triển Jetson Orin Nano Super. <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/> | Phần cứng mục tiêu | 14 |
| JetPack SDK. <https://developer.nvidia.com/embedded/jetpack> | Bộ phần mềm của board | 14 |
| Tài liệu TensorRT. <https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html> | Engine, độ chính xác số, build | 14 |
| ONNX. <https://onnx.ai/> | Định dạng trung gian giữa train và TensorRT | 14 |
| NVIDIA Container Toolkit, cài đặt. <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html> | Cần cho `make build` | 15 |

## Các công cụ mà mã nguồn dùng

| Tài liệu | Dùng để | Chương |
|---|---|---|
| NumPy for absolute beginners. <https://numpy.org/doc/stable/user/absolute_beginners.html> | Mảng, shape, chỉ mục | 0 |
| `numpy.savez_compressed`. <https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html> | Định dạng file episode | 9 |
| PyTorch, Learn the Basics. <https://pytorch.org/tutorials/beginner/basics/intro.html> | Module, autograd, vòng lặp train | 0, 10 |
| `torch.nn.SmoothL1Loss`. <https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html> | Hàm loss khi train | 10 |
| `CosineAnnealingLR`. <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html> | Lịch learning rate | 10 |
| Các bài hướng dẫn OpenCV. † <https://docs.opencv.org/4.x/d9/df8/tutorial_root.html> | `resize`, `remap`, vẽ | 3, 9 |
| Python `threading`. <https://docs.python.org/3/library/threading.html> | `Condition`, được `ThreadedWorker` dùng | 2 |
| Docker Compose. <https://docs.docker.com/compose/> | Cách các target của `make` chạy | 15 |
| Mermaid. <https://mermaid.js.org/> | Các sơ đồ trong giáo trình này | — |

## Bên trong repository này

| File | Nó là gì |
|---|---|
| [`README.vi.md`](../../README.vi.md) | Dự án là gì, và những gì chưa có |
| [`docs/driving.vi.md`](../driving.vi.md) | Cách chạy mọi thứ; các số đo được trích ở đây với nhãn *repo* |
| [`tests/`](../../tests) | Bản đặc tả có thể chạy được: 82 test |
| `runs/drive/reports/*.json`, `runs/drive/models/*.json` | Các báo cáo đánh giá và train (cục bộ, không có trong git) |
