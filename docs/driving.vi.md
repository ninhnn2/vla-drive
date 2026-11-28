# Lái xe

*Bản tiếng Việt · [English](driving.md)*

*Bản tiếng Việt · [English](driving.vi.md)*

`drive/` là một stack lái xe nhỏ được xây quanh một policy trả lời chậm và theo từng chunk,
đúng như một VLA: SmolVLA mất khoảng 100 ms trên Orin Nano và trả về 50 action tương lai. Nó chạy
trên một PC có numpy và OpenCV, với một chiếc xe tỉ lệ 1:10 mô phỏng. Trên Jetson chỉ có policy
thay đổi.

```
camera, tốc độ ─► policy (chậm, ~10 Hz, thread riêng) ─► chunk các [steer, throttle]
                                                            │
                 vòng điều khiển, 50 Hz:  ChunkExecutor ◄───┘   action cho *bây giờ*
                                               │
                 cảm biến khoảng cách ─► SafetySupervisor       có tiếng nói cuối cùng
                                               │
                                            động cơ
```

## Thử ngay

Mọi thứ chạy trong một container: ROS 2 Humble, PyTorch có CUDA, và cửa sổ hiện trên màn hình
của bạn. Cần Docker với NVIDIA container toolkit.

```bash
make build                     # một lần: image (9 GB)
make sim                       # xem lần lái mặc định; q để thoát, e là dừng khẩn cấp
make sim ARGS='--policy lane-camera'          # lái chỉ từ ảnh camera
make sim ARGS='--obstacle 8'                  # một cái cọc mà policy bỏ qua: bộ an toàn dừng xe
make sim ARGS='--obstacle 8 --no-safety'      # ... và chuyện gì xảy ra khi không có nó
make shell                     # một shell trong container; ./drive.sh --help
make test
make help                      # mọi target
```

Trên Mac (Apple Silicon), các target đó chạy `drive/` trực tiếp thay vào đó, trong một `.venv` mà
`make setup` tạo bằng [uv](https://docs.astral.sh/uv/), và việc train dùng GPU của Mac qua backend
MPS của PyTorch. Docker trên Mac không có GPU, nên container
([`compose.mac.yml`](../docker/compose.mac.yml)) chỉ dành cho ROS 2: arm64, PyTorch cho CPU, cửa sổ
qua XQuartz. `NATIVE=0` cũng chạy `drive/` trong đó; `NATIVE=1` dùng `.venv` trên Linux.

`./drive.sh` cũng chạy thẳng trên một PC có numpy và OpenCV, và tự dùng `.venv` khi có; chỉ việc
train và policy học được mới cần PyTorch.

```bash
./drive.sh sim --latency-ms 400 --max-plan-age-ms 1500 --video runs/slow.mp4
./drive.sh sweep                               # sai số bám đường theo độ trễ suy luận
./drive.sh sim --log runs/actions.csv          # mỗi nhịp: action của policy, của động cơ, và lý do
```

Bên dưới HUD, một bảng debug cho thấy policy đang xuất ra gì: chunk đang dùng (với phần bị bỏ qua
do độ trễ và action đang được phát) và 5 s cuối của các lệnh từ model so với những gì tới được
động cơ.

Đường đua: `--track peanut` (mặc định), `oval`, `trefoil`; `--reverse` cho xe chạy theo chiều kim
đồng hồ.

`--show` và `--realtime` chạy theo đồng hồ thật, với suy luận trên một thread thật, như trên xe.
Không có chúng, lần chạy nhanh nhất mà PC cho phép và lặp lại chính xác.

## Bên trong có gì

| | |
|---|---|
| [`sim/`](../drive/sim) | đường đua, một mô hình xe đạp động học có độ trễ servo, một camera hướng trước được dựng từ bản đồ nhìn từ trên xuống, một cảm biến khoảng cách hướng trước |
| [`policy/`](../drive/policy) | `pure-pursuit`: một teacher đọc bản đồ và pose của xe (chỉ trong bộ mô phỏng). `lane-camera`: tìm con đường trong ảnh, không cần bản đồ. `vla.BackendPolicy`: bọc một backend TensorRT từ [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla) |
| [`control/`](../drive/control) | `SimulatedWorker` / `ThreadedWorker`: mỗi lần một lượt suy luận, tách khỏi vòng điều khiển. `ChunkExecutor`: action cho bây giờ từ chunk mới nhất |
| [`safety.py`](../drive/safety.py) | dừng khẩn cấp, watchdog kế hoạch cũ, kiểm tra NaN, giới hạn tốc độ đánh lái và tốc độ xe, phanh khi cảm biến khoảng cách thấy vật |
| [`loop.py`](../drive/loop.py) | một nhịp điều khiển |
| [`episode.py`](../drive/episode.py), [`cli.py`](../drive/cli.py) | cả một lần lái mô phỏng và các số đo của nó |
| [`recorder.py`](../drive/recorder.py), [`viewer.py`](../drive/viewer.py) | ghi một lần lái xuống đĩa thành một episode; cửa sổ và video |
| [`learn/`](../drive/learn) | dây chuyền dữ liệu, mỗi giai đoạn một module: collect, label, validate, browse, train, evaluate, status |
| [`policy/cnn.py`](../drive/policy/cnn.py) | lái bằng một mạng được `learn/` train ra |
| [`platform.py`](../drive/platform.py) | những gì vòng lặp cần từ một chiếc xe: chỗ nối cho ROS 2, CARLA hoặc phần cứng |
| [`ros/src/drive_ros`](../ros/src/drive_ros) | các node ROS 2: xe mô phỏng, vòng điều khiển, một bộ ghi, một cửa sổ teleop |
| [`docker/`](../docker), [`Makefile`](../Makefile) | container và các lệnh chạy trong nó |
| [`config.py`](../drive/config.py) | mọi con số: chiếc xe, camera, các giới hạn |

Một action là `[steer, throttle]`, cả hai trong [-1, 1]: steer +1 là đánh lái hết cỡ sang trái,
throttle -1 là phanh hết cỡ. State mà policy nhận là `[speed, steer]`. Đơn vị là SI.

## Bộ mô phỏng cho thấy gì

Đo bằng `./drive.sh`, các lần chạy 30 s, teacher ở 1,5 m/s trên đường đua 22 m.

**Chunk che được độ trễ, tới một mức nào đó.** Một chunk được đọc tại `now - t_obs`, nên các action
dành cho khoảng thời gian tính toán bị bỏ qua (`ChunkExecutor`). Từ 0 tới 200 ms xe giữ trong phạm
vi 2 cm rms so với đường tâm:

| độ trễ | có bù: rms / max | phát từ đầu chunk: rms / max |
|---:|---:|---:|
| 0 ms | 1.8 / 3.5 cm | 1.6 / 3.0 cm |
| 100 ms | 1.8 / 3.4 cm | 0.7 / 2.2 cm |
| 200 ms | 2.0 / 3.6 cm | 3.8 / 8.8 cm |
| 250 ms | 3.1 / 6.6 cm | 10.0 / 23.3 cm |
| 400 ms | ra khỏi đường | ra khỏi đường |

(`./drive.sh sweep --max-plan-age-ms 1500 --latencies 0,100,200,250,400`.) Dưới 100 ms, bù độ trễ
không quan trọng ở đây; các con số không bù thấp hơn chỉ vì độ trễ tình cờ triệt tiêu thói cắt góc
của pure pursuit.

**Quá khoảng 250 ms ở 1,5 m/s thì vòng lặp mất ổn định.** Mỗi kế hoạch giả định nó bắt đầu tại
thời điểm observation của nó, nhưng xe dùng khoảng độ trễ tiếp theo để chạy theo kế hoạch trước,
nên mọi hiệu chỉnh đều tới muộn và vọt lố. Điều này phụ thuộc tốc độ: ở 0,6 m/s, 400 ms vẫn giữ
được 1,6 cm rms. Một model chậm như vậy cần một kế hoạch có điều kiện trên các action đã cam kết
(real-time chunking), hoặc một chiếc xe chậm hơn. Thứ đó chưa được xây ở đây.

**Watchdog phải lớn hơn hai lần độ trễ.** Kế hoạch đang dùng có tuổi từ một tới hai lần độ trễ.
Với giới hạn mặc định 600 ms, một model 400 ms bị phanh một nửa thời gian và một model 650 ms không
bao giờ chạy được (`stale` trong kết quả). Nâng giới hạn là một quyết định về việc xe được phép đi
bao xa dựa trên thông tin cũ, không phải một núm tinh chỉnh.

**Supervisor không cần policy phải đúng.** Với một cái cọc ở 8 m, teacher lái thẳng vào nó;
supervisor phanh dựa trên cảm biến khoảng cách của chính nó và dừng cách 0,30 m. Với `--no-safety`
xe đâm vào.

## Từ các lần lái tới một mạng biết lái

Toàn bộ vòng học bắt chước, đủ nhỏ để chạy lại trong vài phút. Đó là vòng mà một lần fine-tune VLA
đi qua, với một mạng 0,3 triệu tham số thay cho SmolVLA.

```
collect ─┐
         ├─▶ label ─▶ validate ─▶ (browse) ─▶ train ─▶ evaluate
ROS 2 ───┘                                      ▲          │
  record        thêm lần lái, hoặc đa dạng hơn  └──────────┘
```

```bash
make collect                # teacher lái 40 lần trong bộ mô phỏng: 12 000 khung hình, ~35 s
make validate               # bộ dữ liệu có đầy đủ, nhất quán, đủ đa dạng không?
make browse                 # mọi khung hình, với nhãn của nó vẽ lên mặt đường
make train V=v1             # ~20 s trên RTX 4060, ~2 phút trên M3 (MPS), ~10 phút trên CPU của nó
make evaluate V=v1          # cho nó lái trên mọi đường đua, cả hai chiều
make sim ARGS='--policy cnn --checkpoint v1 --track trefoil'
make status                 # dự án đang có gì, và bước tiếp theo
```

`collect` chạy nhanh nhất mà PC cho phép và không hiển thị gì. Để xem teacher lái trong lúc ghi,
theo thời gian thực: `make collect ARGS='--project runs/demo --episodes 2 --show'` (`--video FILE`
ghi ra cùng khung nhìn đó; cả hai đều không thay đổi những gì được ghi). Để chỉ xem một số
episode: `make browse ARGS="--match 'sim2-*'"`.

`--seed` quyết định mọi lần lái: cùng seed với một `--name` khác sẽ ghi lại đúng các lần lái đó.
Muốn thêm dữ liệu thì cũng cần seed mới: `make collect ARGS='--name sim2 --seed 100'`.

Một dự án là một thư mục, `runs/drive` trừ khi `--project` chỉ định khác:

```
runs/drive/episodes/<name>/episode.npz   t, images (N,240,320,3), states [speed, steer], actions [steer, throttle]
                           meta.json     nhiệm vụ, tần số, ý nghĩa các cột
                           labels.npz    plans (N,16,2), valid (N,), dt, source
           models/<version>.pt, .json    mạng, và quá trình train của nó
           reports/<name>.json           một lần đánh giá
```

| giai đoạn | việc nó làm |
|---|---|
| [`collect`](../drive/learn/collect.py) | teacher pure pursuit lái với tay lái bị nhiễu; mỗi khung hình được lưu kèm kế hoạch của teacher từ điểm đó |
| [`label`](../drive/learn/label.py) | cho các episode không có kế hoạch (bản ghi ROS 2, lần lái của người): kế hoạch là 16 action theo sau khung hình, cách nhau 0,1 s. Các khung hình trong 1,5 s cuối không có đủ tương lai và bị bỏ ra |
| [`validate`](../drive/learn/validate.py) | báo lỗi với những gì làm việc train sai (mảng không đọc được hoặc không khớp, NaN, giá trị ngoài miền, timestamp đi lùi, kích thước ảnh lẫn lộn); cảnh báo với những gì làm nó kém (khoảng trống, khung hình đứng hình, xe đứng yên, quá ít khung hình, chỉ có lái hoàn hảo) |
| [`browse`](../drive/learn/browse.py) | một cửa sổ (hoặc `--video`) với từng khung hình, state và action của nó, và kế hoạch đã gán nhãn được cho chạy tới và vẽ lên mặt đường. Đường đi của một nhãn tốt bám theo con đường |
| [`train`](../drive/learn/train.py) | hồi quy trên các kế hoạch; giữ lại nguyên episode để validation; ảnh lật gương và thay đổi độ sáng làm augmentation. Một phiên bản không bao giờ bị ghi đè |
| [`evaluate`](../drive/learn/evaluate.py) | cho mỗi model lái trên mọi đường đua, chiều xuôi và chiều ngược, rồi so sánh chúng. Một lần lái đạt khi xe không bao giờ ra khỏi đường, không đâm vào gì và vẫn tiếp tục chạy |
| [`status`](../drive/learn/status.py) | episode, nhãn, model, báo cáo, và việc cần làm tiếp |

Một giai đoạn là một module có `add_parser` và `run(project, args)`, được liệt kê trong
[`learn/__init__.py`](../drive/learn/__init__.py); thêm một giai đoạn là viết module đó.

**Nhãn là một kế hoạch, không phải một lớp.** Ở đây không ai vẽ khung: một nhãn lái xe là việc cần
làm tiếp theo, và nó có sẵn miễn phí từ một teacher hoặc từ chính bản ghi. Thứ cần con mắt của con
người là các nhãn có hợp lý hay không, và `browse` sinh ra để làm việc đó.

**Model** ([`learn/model.py`](../drive/learn/model.py)): năm lớp tích chập có stride trên ảnh
96x128, nối thêm state, một MLP ra 16 x `[steer, throttle]`. Giống một VLA, nó trả lời bằng một
chunk, nên cùng executor và cách xử lý độ trễ được áp dụng.

Đo được (`make evaluate V="v1 v2 pure-pursuit"`, các lần lái 40 s với độ trễ 100 ms). `v1` được
train trên 40 episode của bộ mô phỏng, tất cả trên đường đua peanut; `v2` trên các episode đó cộng
thêm hai lần lái ghi qua ROS 2 trên đường oval và trefoil:

| model | lần lái đạt | tốc độ trung bình | rms so với đường tâm | tệ nhất |
|---|---:|---:|---:|---:|
| v1 | 6/6 | 0.84 m/s | 1.3 cm | 4.0 cm |
| v2 | 6/6 | 0.90 m/s | 1.2 cm | 3.8 cm |
| pure-pursuit (teacher) | 6/6 | 1.47 m/s | 1.5 cm | 3.4 cm |

Các mạng lái chậm hơn teacher: tốc độ mục tiêu của teacher thay đổi theo từng episode và không có
trong ảnh, nên chúng chọn một mức trung bình thận trọng.

**Vì sao tay lái của teacher bị làm nhiễu khi ghi.** Một mạng chỉ được cho xem lái hoàn hảo chưa
bao giờ thấy xe lệch khỏi đường tâm; sai sót nhỏ đầu tiên của nó đưa xe tới một chỗ mà nó không có
câu trả lời, và sai số lớn dần. Vì vậy `collect` làm nhiễu tay lái được *áp dụng* và ghi lại việc
teacher *sẽ* làm từ bất cứ chỗ nào xe kết thúc (`--steer-noise`, mặc định 0.25). Cùng dây chuyền đó
với `--steer-noise 0` (`--project runs/perfect`):

| dữ liệu train | sai số steer validation | lần lái đạt | trên đường đua peanut |
|---|---:|---:|---|
| tay lái bị nhiễu (`v1`) | 0.024 | 6/6 | 1,5 cm rms, không bao giờ ra khỏi đường |
| chỉ lái hoàn hảo | 0.015 | 0/6 | ra khỏi đường 75 % thời gian |

Mạng train trên dữ liệu lái hoàn hảo có điểm *tốt hơn* trên các khung hình giữ lại nhưng không lái
được. Sai số offline đo nó sao chép các khung hình được cho tốt tới đâu, không phải chuyện gì xảy
ra khi chính sai sót của nó đẩy xe đi. `validate` cảnh báo về một bộ dữ liệu như vậy ("perfect
driving only"), và `evaluate` mới là bài kiểm tra có giá trị.

## ROS 2

Cùng vòng lặp đó, với chiếc xe nằm sau các topic. Build một lần bằng `make ros-build`, rồi mỗi lệnh
trong một terminal riêng:

```bash
make ros-sim                      # chiếc xe mô phỏng, có cửa sổ           [TRACK=trefoil]
make ros-drive                    # vòng điều khiển lái nó                 [POLICY=cnn V=v1]
make ros-teleop                   # ... hoặc bạn lái: giữ chuột trái trong cửa sổ
make ros-record NAME=ros          # ghi lần lái; Ctrl-C lưu vào runs/drive/episodes/ros-NNN
make ros-view                     # rqt_image_view trên camera
```

hoặc chạy chiếc xe, tài xế và bộ ghi cùng lúc:
`make ros-launch ARGS='policy:=cnn checkpoint:=v1 record:=true'`. Một bản ghi không có nhãn;
`make label`, rồi `make validate`, và nó thành dữ liệu train.

| topic | kiểu | |
|---|---|---|
| `/drive/image` | `sensor_msgs/Image` | camera, rgb8, 25 Hz |
| `/drive/state` | `geometry_msgs/TwistStamped` | `linear.x`: tốc độ theo m/s; `angular.z`: vị trí tay lái, -1..1 |
| `/drive/range` | `sensor_msgs/Range` | khoảng cách tới vật gần nhất phía trước |
| `/drive/pose` | `geometry_msgs/Pose2D` | xe đang ở đâu; chỉ bộ mô phỏng publish nó (teacher cần nó) |
| `/drive/cmd` | `geometry_msgs/Twist` | `linear.x`: throttle, `angular.z`: steer, cả hai -1..1 |

| node | |
|---|---|
| [`sim`](../ros/src/drive_ros/drive_ros/sim_node.py) | `drive.sim.World` chạy từng bước ở 50 Hz; phanh khi không có lệnh nào tới trong 0,5 s |
| [`driver`](../ros/src/drive_ros/drive_ros/drive_node.py) | `DriveLoop`, không đổi, trên [`RosPlatform`](../ros/src/drive_ros/drive_ros/platform.py): policy trên thread riêng, supervisor ở cuối |
| [`teleop`](../ros/src/drive_ros/drive_ros/teleop_node.py) | một cửa sổ có camera; con trỏ chuột là cần lái. Lệnh của nó đi thẳng tới xe, bỏ qua supervisor |
| [`recorder`](../ros/src/drive_ros/drive_ros/recorder_node.py) | ảnh, state và lệnh, 10 lần mỗi giây, vào một episode |

Các topic là toàn bộ giao diện. Một chiếc xe thật, hoặc một cầu nối CARLA, publish các cảm biến và
tuân theo `/drive/cmd` sẽ thế chỗ bộ mô phỏng, và tài xế, bộ ghi cùng dây chuyền train không thay
đổi. Cả hai thứ đó đều chưa tồn tại.

Mỗi target `make` là một container riêng trên mạng của máy host, nên các node tìm thấy nhau;
`ROS_LOCALHOST_ONLY=1` giữ chúng không lọt ra mạng LAN. Cửa sổ (OpenCV, rqt, rviz2) mở trên X server
của máy host; trên Mac là XQuartz, thứ mà các container kết nối tới qua TCP (`make gui` kiểm tra
rằng nó cho phép).

## Từ đây tới một chiếc xe thật

1. **Platform.** Một node trên xe publish camera, tốc độ và khoảng cách, và tuân theo
   `/drive/cmd`, như [`sim_node.py`](../ros/src/drive_ros/drive_ros/sim_node.py) làm cho xe mô
   phỏng.
2. **Dữ liệu.** Lái bằng tay trong khi bộ ghi đang chạy, rồi gán nhãn và validate. Với một VLA,
   `python -m drive export` (trong một môi trường có LeRobot) ghi các episode thành một
   LeRobotDataset: state `[speed, steer]`, action `[steer, throttle]`.
3. **Fine-tune và export.** Fine-tune SmolVLA bằng LeRobot, rồi, với
   [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), chạy `./export.sh` cho
   checkpoint trên PC và `./run.sh` cho bundle trên Jetson để có độ trễ thật của nó.
4. **Policy.** Nạp backend của bundle và bọc nó:
   `BackendPolicy(backend, chunk_size=50, noise_width=32, chunk_dt_s=1 / fps)` trong một
   `ThreadedWorker`. Trên PC, `--policy vla --checkpoint <lerobot checkpoint>` làm việc này với
   model PyTorch (`drive/policy/lerobot_backend.py`); một backend TensorRT chưa được nối vào, và
   chưa có gì chạy trên Jetson.

Chưa có: platform thật, một cầu nối CARLA, một VLA trên Jetson. (Một SmolVLA fine-tune trên các
episode lái bộ mô phỏng trên PC: docs/study, Lab 15.) Lái bằng tay với `teleop` mới được khởi động
và thấy mở được cửa sổ, chưa lái để ghi dữ liệu. Bộ mô phỏng là mặt phẳng và chỉ có động học: không
trượt bánh, không có nhiễu hay phơi sáng của camera, vật cản được vẽ trên mặt đất. Hãy coi các con
số của nó là cách vòng lặp hoạt động, không phải cách một chiếc xe thật sẽ chạy.
