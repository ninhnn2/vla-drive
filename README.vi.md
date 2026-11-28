# vla-drive

*Bản tiếng Việt · [English](README.md)*

*Bản tiếng Việt · [English](README.vi.md)*

Một stack xe tự lái nhỏ để học, được xây quanh cách một model vision-language-action (VLA)
trả lời: chậm, và bằng một chunk các action tương lai. Nó có một chiếc xe tỉ lệ 1:10 mô phỏng,
một vòng điều khiển 50 Hz với bộ giám sát an toàn, một dây chuyền dữ liệu đưa các lần lái được
ghi lại thành một mạng đã train, và các node ROS 2 bao quanh tất cả. Không cần robot; mọi thứ
chạy trong một container Docker trên một PC.

Mục tiêu là một chiếc Jetson Orin Nano Super lái bằng một VLA đã fine-tune. Một nửa của việc đó
đã có ở đây: một SmolVLA được fine-tune trên các episode của bộ mô phỏng lái chiếc xe mô phỏng
từ một PC; nó chưa chạy trên Jetson. Các policy còn lại là một teacher viết bằng script, một bộ
bám làn cổ điển và một mạng 0,3 triệu tham số được dây chuyền train ra.

```
camera, tốc độ ─► policy (chậm, thread riêng) ─► chunk các [steer, throttle]
                                                    │
              vòng điều khiển, 50 Hz: ChunkExecutor ◄┘   action cho *bây giờ*
                                          │
              cảm biến khoảng cách ─► SafetySupervisor   có tiếng nói cuối cùng
                                          │
                                       động cơ
```

## Cách dùng

Cần Docker với NVIDIA container toolkit.

```bash
make build                  # một lần: ROS 2 Humble + PyTorch có CUDA, 9 GB
make sim                    # xem chiếc xe mô phỏng chạy; q để thoát, e là dừng khẩn cấp
make test
make help                   # mọi target
```

Trên máy Mac chip Apple Silicon không có GPU NVIDIA cho Docker, nên các target đó chạy `drive/`
trực tiếp trong một `.venv` và train trên GPU của Mac (MPS). Chỉ ROS 2 dùng container, build cho
arm64 không có CUDA, với cửa sổ hiển thị qua XQuartz. Cần [uv](https://docs.astral.sh/uv/)
(`brew install uv`); với ROS 2 cần thêm Docker Desktop và XQuartz (`make gui` cho biết cần đặt gì).

```bash
make setup                  # một lần: Python 3.10, numpy, OpenCV và PyTorch trong .venv
make sim
make test                   # không có các test ROS 2; make ros-test chạy chúng trong container
```

Từ các lần lái tới một mạng biết lái:

```bash
make collect                # một teacher lái 40 lần trong bộ mô phỏng, với tay lái bị nhiễu
make validate               # bộ dữ liệu có đầy đủ, nhất quán, đủ đa dạng không?
make browse                 # mọi khung hình, với nhãn của nó vẽ lên mặt đường
make train V=v1             # khoảng 20 s trên RTX 4060, 2 phút trên Mac M3
make evaluate V=v1          # cho nó lái trên mọi đường đua, cả hai chiều
make sim ARGS='--policy cnn --checkpoint v1 --track trefoil'
./drive.sh probe v1 errors  # nhìn vào bên trong: layer, lịch sử, sai số, activation, saliency, hành vi
```

Một khoá thực hành bằng tiếng Việt, từ thu thập dữ liệu tới đánh giá và debug mạng, mọi lệnh
kèm output thật: [docs/thuc-hanh](docs/thuc-hanh/README.md). Phần lý thuyết nằm trong
[docs/study](docs/study/README.vi.md), có cả bản tiếng Việt và tiếng Anh.

Cùng vòng lặp đó qua ROS 2 (`make ros-build` một lần, rồi mỗi lệnh một terminal):

```bash
make ros-sim                # chiếc xe mô phỏng, có cửa sổ
make ros-drive POLICY=cnn V=v1
make ros-teleop             # hoặc tự lái bằng chuột
make ros-record             # ghi lần lái thành một episode; Ctrl-C để lưu
```

`./drive.sh` cũng chạy được ngoài container trên một PC có numpy và OpenCV; train và policy học
được cần PyTorch, thứ mà container và `.venv` của `make setup` đều có.

## VLA

Một SmolVLA (khoảng 450 triệu tham số) được fine-tune trên cùng các episode đó. Nó chạy trong
một môi trường LeRobot riêng, bên ngoài container: `make vla-setup` tạo môi trường này
(`.venv-vla`, Python 3.12, LeRobot 0.5.1 và gradio), và `./vla.sh` chạy Python của nó với
repository này trong đường dẫn. Cách đó dùng được cả trên Linux và trên Mac, nơi model chạy trên
GPU của Apple (MPS). Cách cài và mọi tham số có trong Lab 15 của
[docs/study/15-hands-on-labs.vi.md](docs/study/15-hands-on-labs.vi.md).

```bash
make vla-setup                          # một lần: khoảng 1,6 GB
./vla.sh -m drive export                # các episode dưới dạng LeRobotDataset, trong runs/drive/lerobot
./vla.sh lerobot-train --policy.path=lerobot/smolvla_base ...   # 2000 bước batch 16 trên GPU laptop 8 GB
./vla.sh -m drive sim --policy vla --checkpoint smolvla-v1 --show
./vla.sh -m drive evaluate smolvla-v1 --latency-ms 125
```

Để học và debug nó ([chương 17](docs/study/17-debugging-the-model.vi.md) của [giáo trình](docs/study/README.vi.md)):

```bash
./vla.sh tools/probe_vla.py             # chín probe offline: độ chính xác theo bước của chunk, nhiễu, đối xứng, thời gian
./vla.sh tools/see_vla.py               # nó nhìn vào đâu trong ảnh: đầu vào, attention, occlusion
./vla.sh tools/vla_app.py --open        # một web app ở http://127.0.0.1:7860: đưa ảnh vào, nhận chunk ra
```

Mỗi công cụ nhận `--device cuda|mps|cpu` (hoặc `VLA_DEVICE=…`): cùng một khung hình trên hai
thiết bị là cách kiểm tra nhanh rằng GPU tính ra đúng những gì CPU tính. Trên Mac,
[docs/thuc-hanh/08-vla-tren-mac.md](docs/thuc-hanh/08-vla-tren-mac.md) đi từ một `runs/` trống
tới một SmolVLA đã fine-tune và ba công cụ trên, kèm thời gian đo trên một máy M3.

## Nó cho thấy gì

Đo trong bộ mô phỏng; chi tiết và các lệnh có trong [docs/driving.vi.md](docs/driving.vi.md).

- Một mạng train trên 12 000 khung hình từ một đường đua lái được cả ba đường đua, cả hai chiều,
  trong phạm vi 1,3 cm rms so với đường tâm.
- Cùng mạng đó, nếu chỉ train trên dữ liệu lái *hoàn hảo*, có điểm tốt hơn trên các khung hình
  giữ lại (sai số steer 0,015 so với 0,024) nhưng lao ra khỏi đường ở mọi đường đua. Một policy
  được đánh giá bằng cách cho nó lái, không phải bằng loss validation.
- Thực thi các chunk theo thời điểm của observation che được độ trễ suy luận tới khoảng 200 ms ở
  1,5 m/s. Quá khoảng 250 ms thì vòng lặp mất ổn định ở tốc độ đó.
- Supervisor dừng xe cách 0,30 m trước một vật cản mà policy đang lao thẳng vào.
- SmolVLA đã fine-tune vượt qua cả sáu lần lái đánh giá ở 1,44 m/s với 125 ms mỗi lần suy luận,
  nhưng lạng lách: 4,8 cm rms. Teacher của nó bám trong 1,5 cm ở gần cùng tốc độ (1,47 m/s,
  100 ms).
  Che từng phần ảnh cho thấy nó lái theo con đường từ 0,4 m phía trước tới đường chân trời, đúng
  chỗ teacher nhắm. Vì vậy việc lạng lách nhiều khả năng đến từ mức chính xác của câu trả lời hơn
  là từ chỗ nó nhìn vào.

## Cấu trúc thư mục

```
drive/          bộ mô phỏng, các policy, vòng điều khiển, bộ giám sát an toàn
drive/learn/    dây chuyền dữ liệu: collect, label, validate, browse, train, evaluate, export, status
ros/            các node ROS 2: xe mô phỏng, tài xế, teleop, bộ ghi
tools/          probe_vla, see_vla, vla_app: học và debug một VLA đã fine-tune
docs/study/     giáo trình: 18 chương kèm lab, bảng thuật ngữ và sơ đồ draw.io
docker/         container mà mọi thứ chạy trong đó (compose.mac.yml: trên Mac); Makefile chứa các lệnh
tests/          make test
runs/           bộ dữ liệu, model và báo cáo (không có trong git)
```

## Chưa có

Một chiếc xe thật, một cầu nối tới CARLA, và một VLA trên Jetson. SmolVLA đã fine-tune chạy bằng
PyTorch trên PC; nó chưa được export sang TensorRT hay chạy trên board. Bộ mô phỏng là mặt phẳng
và chỉ có động học, nên các con số của nó mô tả cách vòng điều khiển hoạt động, không phải cách
một chiếc xe thật sẽ chạy.

## Lời cảm ơn

Gửi tới [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), dự án chạy các
model VLA trên Orin Nano 8 GB bằng TensorRT và là nơi dự án này bắt đầu; `drive/policy/vla.py`
được viết để bọc các backend của nó.

## Giấy phép

MIT.
