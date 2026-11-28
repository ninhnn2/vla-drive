# 15 · Lab thực hành

*Bản tiếng Việt · [English](15-hands-on-labs.md)*

*Bản tiếng Việt · [English](15-hands-on-labs.vi.md)*

Mười lăm bài thực hành, theo thứ tự phụ thuộc. Mỗi bài mất 15 tới 45 phút, trừ Lab 15 (khoảng
một giờ, phần lớn là chờ).

## Trước khi bắt đầu

```bash
make build        # một lần: container (khoảng 9 GB). Cần Docker và NVIDIA container toolkit
make test         # 82 test phải qua
```

- `make <target>` chạy trong container. Các target mở cửa sổ (`sim`, `browse`,
  `collect --show`, `ros-sim`, `ros-teleop`) cần một màn hình X.
- `./drive.sh …` chạy cùng các lệnh đó trực tiếp trên một PC có numpy và OpenCV, không mở cửa
  sổ trừ khi bạn truyền `--show`. Lab 1–9 làm được theo cách này; lab 10–13 cần container.
- Các kết quả ghi *expected* (kỳ vọng) được đo cho giáo trình này ngày 2026-11-16. Các lần chạy
  pure pursuit là tất định: kết quả của bạn phải khớp chính xác. Các lần chạy có render camera
  có thể khác ở chữ số cuối.
- Viết dự đoán của bạn **trước** mỗi lần chạy. Giá trị của một lab nằm ở khoảng cách giữa điều
  bạn chờ đợi và điều thực sự xảy ra.
- Để các thí nghiệm ngoài `runs/drive`: dùng `--project runs/<tên>`.

Ghi lại tiến độ trong [bảng theo dõi tiến độ học](STUDY-TRACKER.vi.md).

---

## Lab 1: Chạy bộ mô phỏng

**Mục tiêu.** Thấy cả hệ thống chạy, và học cách đọc HUD và kết quả.
**Cần trước.** [Chương 1](01-system-architecture.vi.md).
**File.** [`drive/cli.py`](../../drive/cli.py), [`drive/episode.py`](../../drive/episode.py), [`drive/viewer.py`](../../drive/viewer.py).

```bash
make sim                                   # q để thoát, e bật/tắt dừng khẩn cấp
./drive.sh sim --out runs/lab1/result.json # không mở cửa sổ, kết quả dạng JSON
./drive.sh sim --video runs/lab1/drive.mp4 # ghi HUD thành video
```

**Kỳ vọng.**
```
pure-pursuit on peanut, 100.0 ms per inference (300 inferences), latency compensation on, safety on
  1.99 laps in 30.0 s at 1.46 m/s
  off the centerline: rms 1.8 cm, max 3.4 cm, off the road 0.0 % of the time
  safety overrode the policy: stale 0.1 s, steer_rate 0.06 s
```

**Quan sát.** Bên trái: camera của xe. Bên phải: bản đồ, chiếc xe, và màu lục lam là kế hoạch
trong chunk hiện tại. Bên dưới: "policy in control" màu xanh lá, hoặc các lý do của supervisor
màu đỏ. Bấm `e` và xem dòng đó thay đổi.

**Bảng debug** ở dưới cùng cho thấy policy xuất ra gì
([`Hud.debug_panel`](../../drive/viewer.py#L114)):

- dòng trên cùng: action mà chunk đưa ra cho *lúc này* (`model:`) và thứ supervisor gửi tới
  động cơ (`-> motors:`), màu đỏ kèm lý do khi hai thứ khác nhau;
- bên trái: cả chunk đang dùng, steer màu lục lam và throttle màu cam, theo thời gian kể từ
  observation của nó. Dải xám là phần bị bỏ qua vì nó dành cho khoảng thời gian policy bận tính;
  vạch trắng là action đang được phát;
- bên phải: 5 s gần nhất. Nét đậm là thứ model đòi, nét mảnh là thứ động cơ nhận được; các vạch
  đỏ đánh dấu những nhịp supervisor đã gạt policy.

Để giữ lại mọi nhịp cho sau này, thêm `--log runs/lab1/actions.csv`: mỗi nhịp 20 ms một hàng,
gồm action của chunk, action của động cơ, tuổi kế hoạch và các lý do của supervisor.

**Câu hỏi.**
1. Vì sao có 300 lượt suy luận trong 30 s?
2. `stale 0.1 s` và `steer_rate 0.06 s` từ đâu ra?
3. Với `--show`, `latency_ms_mean` hơn 100 một chút. Vì sao?

**Thử thách.** Chạy `--track trefoil --reverse`. Khả năng bám đường của teacher có đổi không? Có nên đổi không?

---

## Lab 2: Hiểu policy pure pursuit

**Mục tiêu.** Nối công thức pursuit với code và với hành vi của xe.
**Cần trước.** [Chương 4](04-driving-policies.vi.md), mục 1.
**File.** [`drive/policy/pure_pursuit.py`](../../drive/policy/pure_pursuit.py).

```bash
python3 -c "
from drive.policy.pure_pursuit import pursuit_steer
from drive.config import VehicleParams
car = VehicleParams()
for target in [(1.0, 0.0), (1.0, 0.3), (1.0, -0.3), (0.5, 0.3), (0.1, 1.0)]:
    print(target, round(pursuit_steer(*target, car), 3))"

./drive.sh sim --speed 0.8
./drive.sh sim --speed 1.5
./drive.sh sim --speed 2.5
```

**Kỳ vọng.** `(1.0, 0.3)` → 0.338 (đã tính ở chương 4); `(0.5, 0.3)` và `(0.1, 1.0)` → 1.0 (bị
cắt). Với `--speed 0.8`: 0,78 m/s, rms 0,9 cm. Với `--speed 2.5`: tốc độ trung bình 1,76 m/s,
rms 3,5 cm, và `speed 5.5 s` trong các lý do của supervisor.

**Quan sát.** rms thay đổi theo tốc độ thế nào, và tốc độ trung bình thấp hơn hẳn mục tiêu ở 2,5.

**Câu hỏi.**
1. Tự tính `pursuit_steer(1.0, 0.3)` bằng tay.
2. Vì sao tốc độ trung bình là 1,76 mà không phải 2,5? Kể cả hai cơ chế giới hạn nó.
3. Vì sao rms tăng theo tốc độ với một bộ điều khiển có hệ số tắt dần không đổi?

**Thử thách.** Trong một bản sao nháp, đổi khoảng nhìn trước thành hằng số 0,5 m và hằng số
1,4 m (tự dựng `PurePursuitPolicy` và cho nó lái theo khuôn của `episode.build`). Liên hệ điều
bạn thấy với $\omega_n = \sqrt2\,v/\ell$.

---

## Lab 3: Chạy policy lane camera

**Mục tiêu.** Thấy một policy chỉ dùng ảnh, và giới hạn của nó.
**Cần trước.** [Chương 4](04-driving-policies.vi.md), mục 2.
**File.** [`drive/policy/lane_camera.py`](../../drive/policy/lane_camera.py), [`drive/sim/render.py`](../../drive/sim/render.py).

```bash
make sim ARGS='--policy lane-camera'
./drive.sh sim --policy lane-camera
./drive.sh sim --policy lane-camera --track trefoil
python3 -c "
from drive.sim import World
from drive.policy.lane_camera import road_mask
import cv2, numpy as np
w = World(); img = w.image()
cv2.imwrite('runs/lab3_image.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imwrite('runs/lab3_mask.png', road_mask(img).astype(np.uint8) * 255)"
```

**Kỳ vọng.** Trên peanut: 1,2 vòng ở 0,86 m/s, rms 7,3 cm, max 19,2 cm, không bao giờ ra khỏi đường.

**Quan sát.** Kế hoạch màu lục lam là một cung tròn độ cong không đổi (một action lặp lại). Mở
hai file PNG: mặt nạ là mặt đường, tính cả bầu trời.

**Câu hỏi.**
1. Mặt nạ đánh dấu bầu trời là "mặt đường". Vì sao điều đó không làm hỏng policy?
2. Vì sao khả năng bám đường của nó tệ hơn teacher bốn lần?
3. Phần nào trong `road_mask` sẽ hỏng đầu tiên trên một mặt sàn thật?

**Thử thách.** Tìm `--speed` cao nhất mà nó vẫn chạy hết một vòng không ra khỏi đường.

---

## Lab 4: Chèn độ trễ suy luận

**Mục tiêu.** Cảm nhận độ trễ làm gì, và gặp watchdog.
**Cần trước.** [Chương 2](02-data-flow-and-lifecycle.vi.md) và [5](05-action-chunking-and-latency.vi.md).
**File.** [`drive/control/inference.py`](../../drive/control/inference.py), [`drive/safety.py`](../../drive/safety.py).

```bash
for L in 0 100 200 250 400 650; do ./drive.sh sim --latency-ms $L | sed -n '1,2p;4p'; done
make sim ARGS='--latency-ms 400'
```

**Kỳ vọng.** `stale`: 0,0, 0,1, 0,2, 0,3, 13,9, 30,0 s. Số vòng: 2,00, 1,99, 1,98, 1,97, 0,14, 0,00.

**Quan sát.** Ở 400 ms xe giật cục: chạy, phanh, chạy. Trong HUD, "plan age" vượt quá 600 ms và
dòng trạng thái chuyển đỏ ở mỗi chu kỳ.

**Câu hỏi.**
1. Vì sao ở 400 ms xe bị phanh gần đúng một nửa thời gian?
2. Vì sao ở 650 ms nó không bao giờ chạy?
3. Độ trễ lớn nhất mà watchdog mặc định cho phép mà không phanh là bao nhiêu?

**Thử thách.** Thêm `--jitter-ms 40` vào lần chạy 250 ms. Đoán xem `stale` có xuất hiện không,
biết rằng jitter làm một số lượt suy luận chậm hơn 300 ms.

---

## Lab 5: So sánh chunk có bù và không bù

**Mục tiêu.** Tái tạo phép đo trung tâm của dự án.
**Cần trước.** [Chương 5](05-action-chunking-and-latency.vi.md).
**File.** [`drive/control/executor.py`](../../drive/control/executor.py), [`drive/types.py`](../../drive/types.py).

```bash
./drive.sh sweep --max-plan-age-ms 1500 --latencies 0,50,100,200,250,300,400
make sim ARGS='--latency-ms 250 --max-plan-age-ms 1500 --no-latency-compensation'
make sim ARGS='--latency-ms 250 --max-plan-age-ms 1500'
```

**Kỳ vọng** (rms / max cm):

| ms | có bù | không bù |
|---:|---:|---:|
| 0 | 1.8 / 3.5 | 1.6 / 3.0 |
| 100 | 1.8 / 3.4 | 0.7 / 2.2 |
| 200 | 2.0 / 3.6 | 3.8 / 8.8 |
| 250 | 3.1 / 6.6 | 10.0 / 23.3 |
| 300 | 20.2 / 31.0 | 27.2 / 54.4 |
| 400 | 65.2 / 123.9 | 49.0 / 89.0 |

**Quan sát.** Trong hai cửa sổ ở 250 ms: chiếc xe không bù vào cua muộn ở mọi khúc cua.

**Câu hỏi.**
1. Vì sao "không bù" lại tốt hơn ở 100 ms?
2. Vì sao thí nghiệm này cần `--max-plan-age-ms 1500`?
3. Bù độ trễ phát mọi action đúng thời điểm dự định. Vì sao xe vẫn hỏng ở 400 ms?

**Thử thách.** Chạy test `ExecutorTests` (`python3 -m unittest tests.test_drive.ExecutorTests -v`)
và, với mỗi test, nêu tính chất nào của executor mà nó cố định.

---

## Lab 6: Kiểm tra an toàn với một vật cản

**Mục tiêu.** Xem supervisor gạt bỏ một policy đang sai.
**Cần trước.** [Chương 7](07-safety-supervisor.vi.md).
**File.** [`drive/safety.py`](../../drive/safety.py), [`drive/sim/world.py`](../../drive/sim/world.py).

```bash
make sim ARGS='--obstacle 8'
./drive.sh sim --obstacle 8
./drive.sh sim --obstacle 8 --speed 0.8
```

**Kỳ vọng.**
```
  0.33 laps in 30.0 s at 0.24 m/s
  obstacles: not hit, closest 0.3 m
  safety overrode the policy: obstacle_brake 0.22 s, obstacle_slow 0.22 s, obstacle_stop 24.48 s, stale 0.1 s, steer_rate 0.06 s
```

**Quan sát.** Kế hoạch màu lục lam đi xuyên thẳng qua cọc tiêu: teacher không biết nó ở đó. Dòng
trạng thái hiện `obstacle_slow`, rồi `obstacle_brake`, rồi `obstacle_stop`.

**Câu hỏi.**
1. Ở 1,5 m/s, `obstacle_slow` bắt đầu ở khoảng cách nào? `obstacle_brake`?
2. Vì sao xe dừng ở 0,30 m mà không sớm hơn?
3. Vì sao `slow` và `brake` xen kẽ nhau?

**Thử thách.** Đặt hai cọc tiêu: `--obstacle 8 --obstacle 15`. Chuyện gì xảy ra, và vì sao xe không
bao giờ tới được cọc thứ hai?

---

## Lab 7: Tắt an toàn và phân tích kết quả

**Mục tiêu.** Biết supervisor đã ngăn chặn điều gì.
**Cần trước.** Lab 6.
**File.** [`drive/loop.py`](../../drive/loop.py#L45).

```bash
./drive.sh sim --obstacle 8 --no-safety
./drive.sh sim --no-safety                       # không có vật cản
./drive.sh sim --latency-ms 650 --no-safety
```

**Kỳ vọng.** Có cọc tiêu: `obstacles: HIT, closest 0.0 m`, 1,99 vòng ở 1,46 m/s (xe mô phỏng chạy
xuyên qua). Không có cọc tiêu: khả năng bám đường giống hệt khi có an toàn.

**Quan sát.** Dòng đầu giờ ghi `safety OFF`, và không còn dòng "safety overrode".

**Câu hỏi.**
1. Trong một lần chạy sạch, bật hay tắt an toàn cho cùng khả năng bám đường. Điều đó nói gì về cái giá của supervisor?
2. Với `--no-safety`, thứ gì đi tới động cơ trước khi chunk đầu tiên tới? (Đọc nhánh `else` trong `DriveLoop.step`.)
3. Ở 650 ms không có an toàn, xe chạy được. Như vậy có tốt hơn không?

**Thử thách.** Liệt kê mọi lớp bảo vệ mà `--no-safety` gỡ bỏ, theo bảng luật ở chương 7.

---

## Lab 8: Thu thập các episode để train

**Mục tiêu.** Tạo ra một bộ dữ liệu và hiểu nó chứa gì.
**Cần trước.** [Chương 8](08-imitation-learning.vi.md) và [9](09-dataset-pipeline.vi.md).
**File.** [`drive/learn/collect.py`](../../drive/learn/collect.py), [`drive/recorder.py`](../../drive/recorder.py).

```bash
make collect ARGS='--project runs/lab8 --episodes 2 --show'      # xem hai lần lái đang được ghi
./drive.sh collect --project runs/lab8 --name more --seed 1 --episodes 6 --duration-s 20
./drive.sh status --project runs/lab8
```

Sau đó đọc một episode bằng đoạn code ở [chương 9](09-dataset-pipeline.vi.md#đọc-một-episode).

**Kỳ vọng.** Hai episode 300 khung hình, rồi sáu episode 200 khung hình. `status` báo 8 episode,
1800 khung hình, tất cả do teacher gán nhãn, và gợi ý validate rồi train.

**Quan sát.** Trong cửa sổ: vệt màu đỏ tươi (magenta) lang thang khắp mặt đường trong khi kế hoạch
màu lục lam luôn hướng về đường tâm. Dòng dưới cùng cho thấy thứ teacher muốn so với thứ đã
được áp vào.

**Câu hỏi.**
1. Mảng nào chứa steer bị làm nhiễu, và mảng nào chứa kế hoạch sạch?
2. Vì sao lệnh thứ hai cần `--name more --seed 1`?
3. 6 episode dài 20 s là bao nhiêu khung hình, và vì sao?

**Thử thách.** Collect với `--steer-noise 0.5`. Xe đi lệch xa tới đâu? Nó có ra khỏi đường không?
Dữ liệu như vậy có còn hữu ích không?

---

## Lab 9: Validate và xem một bộ dữ liệu

**Mục tiêu.** Đánh giá một bộ dữ liệu trước khi train trên nó.
**Cần trước.** Lab 8.
**File.** [`drive/learn/validate.py`](../../drive/learn/validate.py), [`drive/learn/browse.py`](../../drive/learn/browse.py).

```bash
./drive.sh validate --project runs/lab8; echo "exit $?"
make browse ARGS='--project runs/lab8'                       # d/a khung hình, n/p episode, phím cách, q
make browse ARGS="--project runs/lab8 --match 'more-*'"
./drive.sh browse --project runs/lab8 --every 5 --video runs/lab8/browse.mp4

./drive.sh collect --project runs/lab9 --steer-noise 0 --episodes 3 --duration-s 10
./drive.sh validate --project runs/lab9
```

**Kỳ vọng.** `runs/lab8`: `dataset: ok` với một cảnh báo (1800 khung hình là ít).
`runs/lab9`: thêm cả cảnh báo "perfect driving only".

**Quan sát.** Trong `browse`, quỹ đạo màu lục lam phải đi theo con đường phía trước ở mọi khung
hình, uốn về phía đường tâm khi xe đang lệch khỏi đó.

**Câu hỏi.**
1. Một nhãn *tồi* trông thế nào trong `browse`?
2. Vì sao `validate` báo "perfect driving only" cho `runs/lab9`? Nó so hai mảng nào?
3. `validate` báo FAIL với những vấn đề nào, và chỉ WARN với những vấn đề nào? Vì sao chia như vậy?

**Thử thách.** Làm thí nghiệm "cố ý làm hỏng một bộ dữ liệu" của chương 9.

---

## Lab 10: Train một CNN

**Mục tiêu.** Train một mạng và đọc báo cáo train của nó.
**Cần trước.** [Chương 10](10-model-training.vi.md). Cần container.
**File.** [`drive/learn/train.py`](../../drive/learn/train.py), [`drive/learn/model.py`](../../drive/learn/model.py).

```bash
make collect  ARGS='--project runs/lab10'            # 40 episode, khoảng 35 s
make validate ARGS='--project runs/lab10'
make train    V=v1 ARGS='--project runs/lab10'       # khoảng 20 s trên GPU laptop
make status   ARGS='--project runs/lab10'
python3 -c "
import json; r = json.load(open('runs/lab10/models/v1.json'))
print(r['frames'], r['validation_frames'], r['parameters'], r['device'], r['seconds'])
for h in r['history']: print(h)"
```

**Kỳ vọng.** 10 800 khung hình train, 1 200 khung hình validation, 323 124 tham số. Với seed mặc
định, sai số steer validation cuối cùng gần 0,024 (giá trị *repo* cho cùng thiết lập: 0,0242).

**Quan sát.** Cả hai loss giảm ở mọi epoch; loss validation luôn nằm trên loss train.

**Câu hỏi.**
1. Vì sao là 10 800 và 1 200?
2. Suy ra con số 323 124.
3. Chạy lại `make train V=v1 …`. Chuyện gì xảy ra, và vì sao đó là hành vi đúng?

**Thử thách.** Train với `--seed 1` dưới tên `v1s1`. Sai số validation cuối cùng khác nhau bao
nhiêu? Giữ cả hai cho Lab 11.

---

## Lab 11: Đánh giá trên các đường đua khác nhau

**Mục tiêu.** Đánh giá một model bằng cách cho nó lái, và so phán quyết đó với con số offline.
**Cần trước.** Lab 10, [chương 11](11-model-evaluation.vi.md).
**File.** [`drive/learn/evaluate.py`](../../drive/learn/evaluate.py).

```bash
make evaluate V="v1 pure-pursuit" ARGS='--project runs/lab10'
make sim ARGS='--project runs/lab10 --policy cnn --checkpoint v1 --track trefoil'

# phản ví dụ
make collect  ARGS='--project runs/lab11 --steer-noise 0'
make train    V=v1 ARGS='--project runs/lab11'
make evaluate V=v1 ARGS='--project runs/lab11'
```

**Kỳ vọng.** Model dữ liệu bị làm nhiễu: 6/6 (*repo*: 0,84 m/s, 1,3 cm rms). Model dữ liệu hoàn
hảo: sai số steer validation thấp hơn và 0/6 (*repo*: 0,015, ra khỏi đường 62–85 % thời gian).

**Quan sát.** Xem model dữ liệu hoàn hảo trong một cửa sổ:
`make sim ARGS='--project runs/lab11 --policy cnn --checkpoint v1'`. Để ý thời điểm nó bắt đầu
trôi lệch lần đầu, và việc nó không bao giờ quay lại.

**Câu hỏi.**
1. Model nào có sai số validation tốt hơn? Model nào lái được?
2. Mạng chỉ được train trên peanut. Việc đạt trên trefoil cho thấy gì, và không cho thấy gì?
3. Vì sao mạng lái chậm hơn teacher của nó?

**Thử thách.** Viết bài test hồi phục của chương 11 và tìm độ lệch xuất phát lớn nhất mà từ đó
model của bạn còn quay về được đường tâm.

---

## Lab 12: Chạy mô phỏng qua ROS 2

**Mục tiêu.** Chạy cùng vòng lặp qua nhiều tiến trình và xem xét nó bằng các công cụ ROS.
**Cần trước.** [Chương 12](12-ros2-integration.vi.md).
**File.** [`ros/src/drive_ros/drive_ros/`](../../ros/src/drive_ros/drive_ros).

```bash
make ros-build                         # một lần
# terminal 1
make ros-sim
# terminal 2
make ros-drive                         # hoặc: make ros-drive POLICY=cnn V=v1
# terminal 3
make shell
  ros2 node list
  ros2 topic list -t
  ros2 topic hz /drive/image           # khoảng 25
  ros2 topic hz /drive/cmd             # khoảng 50
  ros2 topic echo /drive/state --once
  ros2 param list /drive_driver
```

**Kỳ vọng.** Các node `/drive_sim` và `/drive_driver`; năm topic `/drive/*`; driver ghi một dòng
trạng thái mỗi 2 s (tốc độ, số lượt suy luận, các lần bị gạt nếu có).

**Quan sát.** Dừng driver bằng Ctrl-C khi xe đang chạy. Xe tự phanh khoảng nửa giây sau.

**Câu hỏi.**
1. Driver không ghi gì về việc phanh khi bạn dừng nó. Ai đã dừng xe?
2. Chuyện gì xảy ra nếu bạn khởi động driver với `TRACK=oval` trong khi bộ mô phỏng đang ở peanut? Vì sao chỉ ảnh hưởng tới teacher?
3. `make ros-launch ARGS='obstacles:=8'`: node nào biết về cọc tiêu, và driver biết về nó bằng cách nào?

**Thử thách.** `make ros-drive ARGS='-p latency_ms:=300'`. `latency_ms` này khác `--latency-ms` của
bộ mô phỏng ở chỗ nào?

---

## Lab 13: Ghi dữ liệu lái bằng teleop

**Mục tiêu.** Ghi một lần lái qua ROS 2 và biến nó thành dữ liệu train.
**Cần trước.** Lab 12.
**File.** [`teleop_node.py`](../../ros/src/drive_ros/drive_ros/teleop_node.py), [`recorder_node.py`](../../ros/src/drive_ros/drive_ros/recorder_node.py), [`label.py`](../../drive/learn/label.py).

> **Trạng thái.** Ghi một lần lái do *node driver* thực hiện thì đã được test
> (`NodeTests.test_teacher_drives_the_simulated_car_and_the_drive_is_recorded`). Ghi một lần
> lái *bằng tay* qua teleop là **[UNTESTED]**: `docs/driving.md` nói teleop "has been started
> and seen to open its window, not driven for a recording". Vì vậy phần B là một thí nghiệm
> theo đúng nghĩa. Hãy ghi lại những gì bạn tìm thấy.

**Phần A: ghi teacher.**
```bash
# terminal 1                         # terminal 2                    # terminal 3
make ros-sim TRACK=oval              make ros-drive TRACK=oval       make ros-record ARGS='-p project:=runs/lab13'
# sau 40 s, Ctrl-C ở terminal 3: "N frames (M s) saved to runs/lab13/episodes/ros-000"
./drive.sh status   --project runs/lab13        # 1 episode, chưa gán nhãn → tiếp theo: label
./drive.sh label    --project runs/lab13
./drive.sh validate --project runs/lab13
make browse ARGS='--project runs/lab13'
```

**Phần B: tự ghi chính bạn.** Thay terminal 2 bằng `make ros-teleop`. Giữ nút trái chuột trong
cửa sổ: bên trái/phải điểm giữa là lái, phía trên điểm giữa là ga, phía dưới là phanh. Ghi một
vòng, rồi label, validate và browse nó.

**Kỳ vọng (A).** Khoảng 10 khung hình cho mỗi giây lái; sau `label`, 15 khung hình cuối không
dùng được; `validate` báo chỉ có một episode là lỗi FAIL cho việc train (cần episode thứ hai).

**Quan sát.** Trong `browse`, kế hoạch giờ là các action đã *xảy ra sau đó*, không phải ý định
của teacher. Với lần lái của chính bạn, xem nó so với con đường thế nào.

**Câu hỏi.**
1. Vì sao recorder chờ lệnh đầu tiên?
2. Episode bạn lái bằng tay là "lái hoàn hảo", "lái bị làm nhiễu", hay không phải cả hai? Một mạng sẽ học được gì từ nó?
3. Lệnh teleop đi vòng qua supervisor. Điều đó có nghĩa gì với `actions` được ghi so với một bản ghi từ node driver?

**Thử thách.** Ghi các lần hồi phục có chủ ý: lái về phía mép đường khi recorder đang tắt, bật nó
lên, và chỉ ghi lúc quay về. Vì sao việc này có thể giá trị hơn ghi cả vòng?

---

## Lab 14: Phân tích kiến trúc triển khai VLA

**Mục tiêu.** Tách cái đã có với cái còn trong kế hoạch, và lên kế hoạch cho bước tiếp theo.
**Cần trước.** [Chương 13](13-vla-and-smolvla.vi.md) và [14](14-jetson-deployment.vi.md).
**File.** [`drive/policy/vla.py`](../../drive/policy/vla.py), [`tests/test_drive.py`](../../tests/test_drive.py#L367).

Một lab đọc và thiết kế; không chạy model nào.

```bash
sed -n 1,47p drive/policy/vla.py
python3 -m unittest tests.test_drive.PolicyTests.test_backend_policy_adapts_a_vla_backend -v
grep -n '"vla"\|BackendPolicy' -r drive ros --include=*.py
```

**Nhiệm vụ.**
1. Vẽ con đường episode → LeRobotDataset → fine-tune → export → TensorRT → Jetson →
   `BackendPolicy`. Tô màu từng ô: đã có và đã chạy / có nhưng chỉ chạy với stub / không có ở đây.
   (Từ Lab 15, hai mũi tên đầu đã chạy trên PC; hãy kiểm tra còn lại những gì.)
2. Viết một backend giả có chunk lái lệch trái 0,2 với 30 % ga, bọc nó trong `BackendPolicy`, và
   cho nó lái bộ mô phỏng:
   ```python
   from types import SimpleNamespace
   import numpy as np
   from drive.control import SimulatedWorker
   from drive.loop import DriveLoop
   from drive.policy.vla import BackendPolicy
   from drive.safety import SafetySupervisor
   from drive.sim import World

   class Fake:
       def infer(self, obs):
           chunk = np.zeros((50, 32), np.float32); chunk[:, 0], chunk[:, 1] = 0.2, 0.3
           return SimpleNamespace(chunk=chunk)

   world = World()
   loop = DriveLoop(world, SimulatedWorker(BackendPolicy(Fake(), chunk_size=50, noise_width=32, chunk_dt_s=1 / 30), 0.108),
                    SafetySupervisor())
   for _ in range(250):
       tick = loop.step()
   print(world.state, tick.decision)
   ```
   Xe làm gì? Việc này chứng minh điều gì, và không chứng minh điều gì?
3. Với một model độ trễ 108 ms: nêu khoảng tuổi kế hoạch, watchdog mặc định có cho phép không,
   và xe đi được bao xa trên kế hoạch cũ nhất ở 1,5 m/s.
4. Lặp lại nhiệm vụ 3 cho 335 ms và 643 ms. Với mỗi trường hợp, chọn một tốc độ và một giới hạn
   watchdog mà bạn bảo vệ được, dẫn các lần quét ở chương 5.
5. Quyết định thứ gì đi vào cột `action` khi chuyển các episode từ `collect` sang LeRobotDataset,
   và lập luận cho nó (chương 13).

**Câu hỏi.**
1. `--policy vla` giờ nạp một checkpoint LeRobot. Cần gì để nó nạp một bundle TensorRT, và code đó sẽ nằm ở đâu?
2. Vì sao chạy model đã fine-tune trong bộ mô phỏng, trên PC, là bước đúng trước khi lên Jetson?
3. Những việc còn lại nào ở chương 14 làm được mà không cần phần cứng?

---

## Lab 15: Fine-tune SmolVLA trên dữ liệu lái xe, ngay trên laptop này

**Mục tiêu.** Đưa chính các episode của dự án đi hết con đường tới một SmolVLA đã fine-tune lái
được chiếc xe mô phỏng, trên một GPU laptop, và đánh giá nó theo đúng cách đánh giá CNN.
**Cần trước.** Lab 8–11, [chương 13](13-vla-and-smolvla.vi.md). Một GPU NVIDIA từ 8 GB trở lên,
khoảng 15 GB đĩa trống **trên một ổ còn chỗ**, và mạng cho lần tải đầu tiên (khoảng 4 GB gói
cài đặt, 3 GB model).
**File.** [`drive/learn/export.py`](../../drive/learn/export.py),
[`drive/policy/lerobot_backend.py`](../../drive/policy/lerobot_backend.py),
[`drive/policy/vla.py`](../../drive/policy/vla.py).

**Trạng thái.** Mọi lệnh bên dưới đã được chạy cho giáo trình này ngày 2026-11-17 trên chiếc
laptop chứa repository này (RTX 4060 Laptop GPU, 8 GB, driver 535) với LeRobot 0.5.2 (bản clone
cục bộ tại `/home/fanning/data1/AI_SOURCE/VLA/lerobot`, commit `3dd19d043`). Các con số ghi
*expected* lấy từ những lần chạy đó. Phần Colab ở cuối **chưa** được chạy.

### Bước 0: một môi trường LeRobot, làm một lần (khoảng 30 phút, chủ yếu là tải)

Container của dự án có PyTorch nhưng không có LeRobot, và LeRobot cần Python 3.12. Vì vậy
LeRobot có môi trường ảo riêng, nằm ngoài repository. Đặt nó, và mọi cache, trên một ổ đĩa còn
chỗ: trên laptop này `/` chỉ còn trống 21 GB, và cả uv lẫn Hugging Face mặc định đều cache vào
`~/.cache` (nằm trên `/`).

```bash
D=/home/fanning/data1/AI_SOURCE/smolvla-local          # mọi thứ trong một thư mục; xoá nó là hoàn tác
mkdir -p $D && cd $D
export UV_CACHE_DIR=$D/uv-cache UV_PYTHON_INSTALL_DIR=$D/python HF_HOME=$D/hf
uv venv --python 3.12 .venv
uv pip install --python .venv "/home/fanning/data1/AI_SOURCE/VLA/lerobot[smolvla,dataset,training]"
.venv/bin/python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
```

*Kỳ vọng:* `2.11.0+cu128 True 0.5.2` và một `.venv` 7,4 GB. Bản PyTorch build cho CUDA 12.8 chạy
được trên driver 535 (CUDA 12.2) ở đây.

Các extra là `smolvla,dataset,training`. Notebook Colab ghi `train`, extra này không tồn tại
trong LeRobot 0.5.2.

LeRobot cài `opencv-python-headless`, bản không mở được cửa sổ. Để dùng `--show` (Bước 3), đổi nó
sang bản có cửa sổ cùng phiên bản, một lần:

```bash
uv pip uninstall --python .venv opencv-python-headless
uv pip install --python .venv "opencv-python==4.13.0.92"     # đúng phiên bản của bản headless
```

**Mọi shell sau này** đều cần đặt lại vị trí cache, nếu không model sẽ bị tải về `/` lần thứ hai:

```bash
export D=/home/fanning/data1/AI_SOURCE/smolvla-local HF_HOME=/home/fanning/data1/AI_SOURCE/smolvla-local/hf
export V=$D/.venv/bin/python                           # Python của môi trường LeRobot
cd /home/fanning/data1/AI_SOURCE/vla-drive && export PYTHONPATH=.
```

Với `PYTHONPATH=.`, `$V -m drive …` chạy CLI của repository này (cùng các lệnh như `./drive.sh`)
bên trong môi trường LeRobot.

**Trên Mac (Apple Silicon)**, hoặc ở bất cứ đâu không có thư mục đó: `make vla-setup` tạo cùng loại
môi trường trong `.venv-vla` (Python 3.12, LeRobot **0.5.1** từ PyPI, bản phát hành gần nhất với
bản checkout 0.5.2 ở trên, và gradio, với bản OpenCV có cửa sổ), và `./vla.sh` đóng vai `$V` với
`PYTHONPATH=.` và `PYTORCH_ENABLE_MPS_FALLBACK=1` đã đặt sẵn. Có hai khác biệt: 0.5.1 không có
`--dataset.eval_split`, `--eval_steps` hay `--max_eval_samples` (bỏ chúng khỏi Bước 2), và device
là `--policy.device=mps` với `--batch_size=8` trong 16 GB. Trên M3, một bước batch 8 mất khoảng
2 s. Toàn bộ con đường trên Mac, kèm output, nằm ở
[docs/thuc-hanh/08-vla-tren-mac.md](../thuc-hanh/08-vla-tren-mac.md).

### Bước 1: xuất các episode (vài giây)

```bash
./drive.sh validate                  # bộ dữ liệu phải được gán nhãn và hợp lệ trước
$V -m drive export                   # -> runs/drive/lerobot
```

*Kỳ vọng* (82 episode trong `runs/drive`): `24876 frames in 82 episodes at 10 Hz`, một bộ dữ liệu
35 MB, và lệnh `lerobot-train` đầy đủ được in ở cuối. Chạy lại thì nó từ chối ("exists"): một
bộ dữ liệu chỉ được ghi một lần.

Xem những gì đã được ghi:

```bash
cat runs/drive/lerobot/meta/info.json | head -40
$V - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/vla-drive", root="runs/drive/lerobot",
                    delta_timestamps={"action": [k / 10 for k in range(16)]})
item = ds[100]
print(len(ds), item["observation.images.front"].shape, item["observation.state"], item["action"].shape)
EOF
```

`item["action"]` là `(16, 2)`: mục tiêu train mà LeRobot dựng cho khung hình đó từ action của 16
khung hình tiếp theo.

### Bước 2: fine-tune (khoảng 13 phút)

```bash
mkdir -p runs/drive/vla
HF_HUB_DISABLE_TELEMETRY=1 $D/.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/vla-drive --dataset.root=$PWD/runs/drive/lerobot \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --dataset.eval_split=0.1 --eval_steps=500 --max_eval_samples=512 \
  --batch_size=16 --steps=2000 --log_freq=50 --save_freq=1000 \
  --policy.push_to_hub=false --policy.device=cuda --wandb.enable=false \
  --output_dir=runs/drive/vla/smolvla-v1 --job_name=smolvla-v1 2>&1 | tee runs/drive/vla/smolvla-v1.log
```

Mỗi lựa chọn làm gì:

| Cờ | Vì sao |
|---|---|
| `--policy.path=lerobot/smolvla_base` | bắt đầu từ model 450 triệu tham số đã pre-train. **Không** dùng `--policy.type=smolvla` như trong notebook Colab: trong LeRobot 0.5.2, cách đó dựng VLM với trọng số chưa train (`load_vlm_weights` mặc định là false) rồi đóng băng nó |
| `--dataset.root=…` | một bộ dữ liệu cục bộ; không tải về hay đẩy lên gì cả |
| `--rename_map` | `smolvla_base` chờ các camera tên `camera1`, `camera2`, `camera3`. Không có cờ này: "Feature mismatch between dataset/environment and policy config". Một camera trên ba được chấp nhận |
| `--policy.chunk_size=16` | 16 action cách nhau 0,1 s: đúng 1,5 s mà CNN dự đoán. Giá trị mặc định 50 sẽ là 5 s ở 10 Hz. Trọng số không phụ thuộc vào nó |
| `--dataset.eval_split=0.1 --eval_steps=500` | giữ lại 9 trên 82 episode và báo loss của chúng mỗi 500 bước |
| `--batch_size=16` | vừa trong 8 GB. 32 thì hết bộ nhớ (với chunk 50; chưa thử lại với 16) |
| `--policy.push_to_hub=false` | mặc định là **true**; không có cờ này LeRobot đòi một repository trên Hub để đẩy lên |
| `--output_dir=runs/drive/vla/smolvla-v1` | nơi `Project.vla("smolvla-v1")` tìm tới; `runs/` không nằm trong git |

*Kỳ vọng:* `73 train, 9 eval`, `num_learnable_params=99880992 (100M)` trên 450 triệu, 2,5–2,7 bước
mỗi giây, 2,9 GB bộ nhớ GPU, tổng cộng 13 phút.

| Bước | Loss train | Loss trên tập giữ lại |
|---:|---:|---:|
| 50 | 1.018 | |
| 500 | khoảng 0.47 | 0.651 |
| 1000 | | 0.484 |
| 1500 | | 0.499 |
| 2000 | | 0.469 |

Checkpoint mà các bước sau dùng là `runs/drive/vla/smolvla-v1/checkpoints/last/pretrained_model/`
(0,9 GB trọng số, cấu hình, và các bộ tiền/hậu xử lý với thống kê chuẩn hoá của bộ dữ liệu này).

### Bước 3: cho nó lái (khoảng một phút)

```bash
$V -m drive sim --policy vla --checkpoint smolvla-v1 --latency-ms 125 \
   --video runs/drive/vla/smolvla-v1-peanut.mp4
$V -m drive sim --policy vla --checkpoint smolvla-v1 --latency-ms 0 --show \
   --log runs/drive/vla/smolvla-v1-actions.csv                                   # một cửa sổ, thời gian thật
```

Với `--show`, suy luận chạy trên một thread thật, nên `--latency-ms 0` không thêm gì và độ trễ là
của chính model (*measured* 150–190 ms khi mở cửa sổ). Bảng debug dưới HUD (Lab 1) cho thấy mỗi
chunk 16 action SmolVLA xuất ra, phần bị bỏ qua trong lúc nó tính, và lệnh của model so với lệnh
tới động cơ. `--log` ghi cùng thông tin đó theo từng nhịp ra CSV.

`--checkpoint smolvla-v1` được tìm trong `runs/drive/vla/`; đường dẫn tới bất kỳ thư mục
`pretrained_model` nào cũng dùng được. `--latency-ms 125` vì đó là thời gian một lượt suy luận mất
trên laptop này (*measured*: trung bình 124 ms, sau lần gọi đầu 460 ms; model nạp trong khoảng 7 s).

*Kỳ vọng:*
```
vla on peanut, 125.0 ms per inference (215 inferences), latency compensation on, safety on
  1.88 laps in 30.0 s at 1.39 m/s
  off the centerline: rms 4.4 cm, max 10.5 cm, off the road 0.0 % of the time
  safety overrode the policy: stale 0.14 s, steer_rate 0.06 s
```

### Bước 4: đánh giá nó (khoảng 5 phút)

```bash
$V -m drive evaluate smolvla-v1 --latency-ms 125       # ghi runs/drive/reports/smolvla-v1.json
```

*Kỳ vọng:*

```
smolvla-v1, 125 ms per inference, 40 s per drive
track               laps   m/s  rms cm  max cm  off road  result
peanut              2.60  1.44     5.6    14.2     0.0 %  pass
peanut reversed     2.63  1.46     5.8    15.0     0.0 %  pass
oval                2.80  1.43     3.6    12.5     0.0 %  pass
oval reversed       2.78  1.43     4.2    12.5     0.0 %  pass
trefoil             2.40  1.44     3.8     9.3     0.0 %  pass
trefoil reversed    2.40  1.45     5.6    12.5     0.0 %  pass
6/6 drives passed
```

Tóm tắt: 1,44 m/s, 4,8 cm rms, 15,0 cm tệ nhất; supervisor chỉ can thiệp lúc khởi động
(`stale 0.14 s`, `steer_rate` 0,02–0,06 s). Đặt cạnh các model khác ở
[chương 13](13-vla-and-smolvla.vi.md#con-đường-chỉ-dùng-pc-đã-làm-một-lần): SmolVLA lái ở tốc độ của teacher,
trong khi các CNN lái ở 0,84–0,90 m/s, và bám đường kém chặt hơn khoảng ba lần so với cả hai.

`evaluate` ghi `runs/drive/reports/smolvla-v1.json`, và thay nó nếu bạn chạy lại với thiết lập khác.

### Quan sát

- Trong video hoặc cửa sổ, kế hoạch màu lục lam uốn theo con đường, và xe lái ở tốc độ gần với
  teacher.
- Nó lạng lách nhiều hơn CNN: 4–6 cm rms so với 1–2 cm.
- Loss trên tập giữ lại ngừng cải thiện sau khoảng 1000 bước, nhưng điều đó không nói gì về việc
  train thêm có lái tốt hơn hay không.

### Câu hỏi

1. Vì sao `--action intended` là mặc định của export, và `--action applied` sẽ dạy điều gì?
2. Chunk mục tiêu cho khung hình $i$ được dựng từ các khung hình $i \dots i+15$. Vì sao nó khác kế
   hoạch của teacher ở khung hình $i$ với các episode từ `collect`, và nó ảnh hưởng nhiều nhất tới
   đầu nào của chunk? Phần nào của chunk thực sự được xe thực hiện ở độ trễ 125 ms?
3. Sai số steer validation của CNN là 0,024; trên các khung hình của `sim2-039`, action đầu tiên
   của SmolVLA lệch khỏi kế hoạch của teacher khoảng 0,1. Model nào lái nhanh hơn mà vẫn đạt?
   Điều đó, một lần nữa, nói gì về sai số offline?
4. Vì sao `--latency-ms 125` là thiết lập đúng ở đây, và bạn sẽ đặt bao nhiêu cho Jetson?

### Thử thách

- Export với `--action applied` vào một thư mục khác, fine-tune `smolvla-applied`, rồi đánh giá cả
  hai. Dự đoán trước.
- Quét độ trễ: `$V -m drive sweep --policy vla --checkpoint smolvla-v1 --max-plan-age-ms 1500 --latencies 125,200,250,300`
  (chưa chạy cho giáo trình này; mỗi hàng là một lần lái 30 s cho mỗi chế độ, tốn khoảng 1,5 s
  tính toán cho mỗi giây mô phỏng). So với sự sụp đổ của CNN ở 300 ms ([chương 11](11-model-evaluation.vi.md)).
- Train lâu hơn (`--steps=10000`, khoảng một giờ) và kiểm tra xem việc lái có cải thiện trong khi
  loss trên tập giữ lại thì không.

### Các lỗi thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `Feature mismatch … Missing features: ['observation.images.camera1', …]` | tên camera khác với của model đã pre-train | `--rename_map` như trên |
| `'repo_id' argument missing` | `push_to_hub` mặc định là true | `--policy.push_to_hub=false` |
| `CUDA out of memory` | batch quá lớn cho 8 GB | `--batch_size=16` hoặc 8 |
| đĩa bị đầy | chưa đặt `HF_HOME`: model đã vào `~/.cache` | export `HF_HOME` ở mọi shell |
| `export needs LeRobot` / `the vla policy needs LeRobot` | lệnh chạy bằng Python của hệ thống | dùng `$V -m drive …` |
| `no model 'smolvla-v1'` | `--output_dir` của lần chạy không phải `runs/drive/vla/smolvla-v1` | truyền đường dẫn tới `pretrained_model` thay vào |
| loss giảm nhưng xe lái tệ | model học sai thứ (kiểm tra `--action`, rename map, chunk size) | đánh giá sớm; đừng tin loss |

### Chạy lần fine-tune dài trên Colab thay vào đó **[UNTESTED]**

Laptop là đủ cho lần chạy ngắn ở trên. Với nhiều bước hơn hẳn hoặc batch 64, dùng
[notebook Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb)
với các thay đổi sau, chưa thay đổi nào được thử cho giáo trình này:

1. **Đưa bộ dữ liệu lên đó.** Bản export nặng 35 MB: tải thư mục `runs/drive/lerobot` lên Google
   Drive, hoặc đẩy nó lên một repository dataset **riêng tư** trên Hugging Face. Cả hai cách đều
   đưa dữ liệu của bạn ra khỏi máy; hãy quyết định việc đó một cách có chủ ý.
2. **Cài đặt** bằng `pip install -e ".[smolvla,dataset,training]"` (không phải `train`).
3. **Train** bằng lệnh ở Bước 2, nhưng với `--dataset.root=/content/drive/MyDrive/…/lerobot`, một
   `--batch_size` lớn hơn nếu GPU cho phép, và `--output_dir` trên Google Drive để một phiên bị
   ngắt có thể tiếp tục bằng `--resume=true`.
4. **Mang checkpoint về**: chép `checkpoints/last/pretrained_model/` vào
   `runs/drive/vla/<version>/checkpoints/last/` và chạy Bước 3–4 trên laptop.

GPU miễn phí và giới hạn phiên của Colab thay đổi và không được bảo đảm; hãy kiểm tra thứ bạn được
cấp trước khi bắt đầu một lần chạy dài.

---

## Đáp án một số câu hỏi

<details><summary>Lab 1</summary>

1. Mỗi 100 ms một lượt suy luận, lượt nối lượt: 30 s × 10 Hz.
2. 100 ms đầu chưa có kế hoạch nào (năm nhịp bị phanh); lệnh lái đầu tiên nhảy từ 0 lên hơn 0,16 mỗi nhịp trong ba nhịp.
3. `--show` dùng `ThreadedWorker`, có độ trễ là 100 ms ngủ cộng thời gian tính toán thật.
</details>

<details><summary>Lab 4</summary>

1. Tuổi kế hoạch dao động giữa 0,4 và 0,8 s và vượt 0,6 s trong 9 trên 20 nhịp.
2. Tuổi luôn trên 0,65 s.
3. Ngay dưới 300 ms (tuổi đạt tới gấp đôi độ trễ).
</details>

<details><summary>Lab 5</summary>

1. Độ trễ tình cờ triệt tiêu xu hướng vào cua sớm của pure pursuit.
2. Nếu không, watchdog phanh ở 400 ms và bạn đo watchdog, không phải khả năng bám đường.
3. Các action đúng thời điểm nhưng được tính từ thông tin cũ 0,4–0,8 s, với giả định xe đã đi theo kế hoạch này trong lúc đó; thực tế nó đi theo kế hoạch trước.
</details>

<details><summary>Lab 6</summary>

1. Thả trôi dưới 0,74 m, phanh dưới 0,58 m.
2. `stop_margin_m = 0.30`: nó được phép tiến lại cho tới khi khoảng trống bằng 0.
3. Phanh đưa tốc độ xuống dưới đường cong cho phép, phanh được nhả, và xe, vẫn đang tiến lại gần, lại vượt đường cong lần nữa.
</details>

<details><summary>Lab 7</summary>

1. Với một policy cư xử đúng mực, supervisor là trong suốt.
2. `np.zeros(2)`: bánh thẳng, ga bằng 0. Xe thả trôi thay vì phanh.
3. Không. Nó lái theo kế hoạch cũ tới 1,3 s mà không có gì ngăn lại: *measured*, 150 cm rms và ra khỏi đường một nửa thời gian.
</details>

<details><summary>Lab 11</summary>

1. Model dữ liệu hoàn hảo có sai số validation tốt hơn; model dữ liệu bị làm nhiễu thì lái được.
2. Rằng ánh xạ nó học được mang tính cục bộ (các khúc cua, không phải thuộc lòng một vòng đua). Không cho thấy nó đối phó được với một con đường có vẻ ngoài khác.
3. Tốc độ mục tiêu của teacher thay đổi theo từng episode và không nhìn thấy được trong ảnh.
</details>

<details><summary>Lab 12</summary>

1. Watchdog lệnh 0,5 s của chính chiếc xe mô phỏng.
2. Teacher đi theo bản đồ của sai con đường và lái ra ngoài. Các policy dùng camera không dùng bản đồ.
3. Chỉ bộ mô phỏng. Driver biết về nó qua `/drive/range`.
</details>
