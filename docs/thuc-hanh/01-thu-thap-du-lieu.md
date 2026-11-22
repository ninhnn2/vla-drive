# Bài 1 · Thu thập và hiểu dữ liệu

**Mục tiêu.** Học xong bài này, bạn sẽ:

- giải thích được vì sao lúc thu dữ liệu, xe bị **cố tình đẩy lệch**;
- mở một episode trên đĩa và nói được từng mảng (array) chứa gì;
- chạy `validate`, `browse` và đọc được output của chúng;
- dùng một bảng kiểm để nói **bộ dữ liệu này tốt hay xấu, vì sao**.

**Lý thuyết sâu hơn:** [chương 08 · Imitation learning](../study/08-imitation-learning.md) và
[chương 09 · Dataset pipeline](../study/09-dataset-pipeline.md) (tiếng Anh).

---

## 1.1 Ý tưởng: học bắt chước, và vì sao phải đẩy lệch xe

Model trong dự án học theo kiểu **behaviour cloning** (học bắt chước). Nó được xem rất
nhiều cặp ví dụ:

```
đầu vào:  ảnh camera 240x320  +  trạng thái [tốc độ m/s, góc lái hiện tại]
đầu ra:   kế hoạch 16 bước x [steer, throttle], mỗi bước cách nhau 0,1 s (tức 1,5 s tới)
```

Kế hoạch đúng (gọi là **label**, hay nhãn) do một **teacher** đưa ra. Teacher là
thuật toán pure pursuit, biết bản đồ và biết chính xác vị trí của xe. Model thì không
biết gì ngoài ảnh và trạng thái (tốc độ, góc lái), nên nó phải học cách đọc con đường từ
ảnh. Nó cũng không biết teacher đang muốn giữ tốc độ bao nhiêu. Bài 5 sẽ cho thấy chi
tiết nhỏ này quan trọng đến mức nào.

**Cái bẫy.** Nếu chỉ thu dữ liệu lúc teacher lái hoàn hảo, xe lúc nào cũng nằm giữa
làn. Model chưa bao giờ thấy cảnh "xe lệch về mép đường". Khi chạy thật, model sai
một chút (model nào cũng sai một chút), xe lệch một chút và gặp cảnh lạ. Ở cảnh lạ đó
model sai nhiều hơn, xe lệch thêm, và cứ thế lặp lại cho tới khi văng khỏi đường.
Hiện tượng này gọi là **covariate shift**: phân bố dữ liệu lúc chạy khác với lúc học.

**Cách sửa (DART).** Lúc thu dữ liệu, ta **cộng nhiễu vào góc lái thật sự đưa xuống
bánh xe**, để xe lắc lư và lệch khỏi tâm đường. Nhưng **nhãn luôn là kế hoạch sạch của
teacher** tính từ chỗ xe đang đứng. Nhờ vậy model được xem rất nhiều ví dụ "xe đang
lệch thì phải lái thế này để quay về".

```
            teacher muốn: steer +0.51 ──► nhãn (label) = kế hoạch sạch của teacher
                                │
  nhiễu ──► cộng vào ──► áp lên bánh xe: steer +0.25 ──► xe lệch đi ──► frame tiếp theo
```

Bài 5 sẽ chứng minh điều này bằng số liệu: cùng một model, train trên dữ liệu lái hoàn
hảo thì có sai số validation **tốt hơn**, nhưng **văng khỏi đường ở cả 6/6 lượt lái**.

## 1.2 Xem tận mắt (khoảng 1 phút)

Lệnh dưới đây thu 2 episode và mở cửa sổ để bạn xem trong lúc thu. Cần có màn hình:
trên Mac cửa sổ mở thẳng, còn trong container Linux thì dùng `make collect ARGS='...'`.

```bash
./drive.sh collect --project runs/demo --episodes 2 --show
```

Trong cửa sổ:

- **Bên trái** là camera của xe. **Đường cyan** là kế hoạch của teacher tính từ vị trí
  hiện tại, cũng chính là nhãn được lưu lại.
- **Bên phải** là bản đồ nhìn từ trên xuống. **Màu magenta** là quãng đường xe đã đi:
  nó lắc lư quanh tâm đường, đó là tác dụng của nhiễu.
- **Dòng `steering:`** ở phía dưới ghi, chẳng hạn,
  `teacher wants +0.51   applied +0.25   disturbance -0.26`, và chuyển sang **màu đỏ**
  khi nhiễu lớn hơn 0,2.

Mỗi episode dài 30 s, chạy theo thời gian thực. Bấm `q` để dừng; các episode đã thu vẫn
được giữ lại. Bật `--show` hay không thì nội dung ghi lại vẫn y hệt nhau, đã có test kiểm
tra điều này. Xem xong thì xóa đi: `rm -rf runs/demo`.

## 1.3 Thu bộ dữ liệu chính (khoảng 1 phút)

```bash
./drive.sh collect --project runs/hoc
```

```
recording 40 drives of 30 s to runs/hoc/episodes
  episode 1/40: peanut, 0.9 m/s
  episode 2/40: peanut reversed, 1.8 m/s
  episode 3/40: peanut, 1.0 m/s
  episode 4/40: peanut reversed, 1.2 m/s
  ...
  episode 39/40: peanut, 1.6 m/s
  episode 40/40: peanut reversed, 1.5 m/s
12000 frames in 40 episodes; the disturbed car strayed up to 59 cm from the centerline
  50 frames (0.4 %) were recorded with the car off the road
```

**Đọc output:**

| Dòng | Nghĩa |
|---|---|
| `episode 2/40: peanut reversed, 1.8 m/s` | Episode thứ 2 chạy trên track `peanut` theo chiều kim đồng hồ (`reversed`), teacher giữ tốc độ mục tiêu 1,8 m/s. Chiều chạy luân phiên giữa các episode để số cua trái và cua phải bằng nhau. Tốc độ mục tiêu được chọn ngẫu nhiên trong khoảng 0,9–1,8 m/s. |
| `12000 frames in 40 episodes` | 40 episode × 30 s × 10 frame/s = 12 000 frame. Camera ghi 10 lần/giây, trong khi vòng điều khiển chạy 50 lần/giây. |
| `strayed up to 59 cm` | Nhiễu đẩy xe lệch tâm tối đa 59 cm. Đường rộng 90 cm, nghĩa là mép đường cách tâm 45 cm, nên có lúc xe ra khỏi đường trong chốc lát. |
| `50 frames (0.4 %) ... off the road` | Bao nhiêu frame được ghi lúc xe đang ở ngoài đường. Vài phần trăm thì có ích, vì chúng dạy model cách quay về. Nếu quá 5 % thì lệnh in thêm `WARN`: lúc đó nhiễu quá mạnh, teacher không kéo xe về kịp, và model sẽ học cách "lái trên cỏ". |

**Các tham số bạn sẽ thay đổi ở bài 5:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--episodes` | 40 | số lượt lái |
| `--duration-s` | 30 | độ dài mỗi lượt, tính bằng giây |
| `--tracks` | `peanut` | danh sách track, cách nhau bởi dấu phẩy: `peanut,oval,trefoil` |
| `--steer-noise` | 0.25 | độ mạnh của nhiễu; 0 nghĩa là lái hoàn hảo |
| `--speeds` | `0.9,1.8` | khoảng tốc độ mục tiêu của teacher, hoặc một giá trị cố định cho mọi episode |
| `--seed` | 0 | cùng seed thì cùng dữ liệu, giống hệt nhau trên mọi máy |
| `--name` | `sim` | tên episode: `sim-000`, `sim-001`, ...; dùng tên khác để thu thêm vào cùng project |

## 1.4 Một episode trên đĩa

```
runs/hoc/episodes/sim-000/
  episode.npz     3.2 MB   những gì xe thấy và làm
  labels.npz       24 KB   nhãn: kế hoạch của teacher cho từng frame
  meta.json        <1 KB   task, tần số, tên các cột
```

Mở ra xem. Copy cả khối dưới đây (đây là một đoạn Python chạy thẳng trong terminal).
Trên Mac, nhớ chạy `source .venv/bin/activate` trước (xem [phần chuẩn bị](README.md#chuẩn-bị-làm-một-lần)).
Nếu không, `python3` sẽ là Python của hệ thống, không có numpy, và báo `ModuleNotFoundError`.

```bash
python3 - <<'EOF'
import json
import numpy as np

ep = "runs/hoc/episodes/sim-000"
data = np.load(f"{ep}/episode.npz")
labels = np.load(f"{ep}/labels.npz")
print("episode.npz:")
for k in data.files:
    print(f"  {k:8s} {str(data[k].shape):20s} {data[k].dtype}")
print("labels.npz:")
for k in labels.files:
    print(f"  {k:8s} {str(labels[k].shape):20s} {labels[k].dtype}")
print("meta.json:", json.load(open(f"{ep}/meta.json")))

i = 100
print(f"\nframe {i}: t = {data['t'][i]:.1f} s")
print("  state  [speed m/s, steer]  =", data["states"][i])
print("  action [steer, throttle]   =", data["actions"][i], " <- what went to the motors (with the disturbance)")
print("  label: the teacher's plan, 16 steps x [steer, throttle], 0.1 s apart:")
for k in (0, 1, 2, 5, 10, 15):
    print(f"    step {k:2d} (+{k * 0.1:.1f} s)  {labels['plans'][i][k]}")
print(f"  disturbance = applied - wanted = {data['actions'][i, 0] - labels['plans'][i, 0, 0]:+.3f}")
EOF
```

```
episode.npz:
  t        (300,)               float32
  images   (300, 240, 320, 3)   uint8
  states   (300, 2)             float32
  actions  (300, 2)             float32
labels.npz:
  plans    (300, 16, 2)         float32
  valid    (300,)               bool
  dt       ()                   float64
  source   ()                   <U7
meta.json: {'task': 'drive along the track', 'fps': 10.0, 'frames': 300, 'state': ['speed_mps', 'steer'], 'action': ['steer', 'throttle'], 'image': 'uint8 HxWx3 RGB'}

frame 100: t = 10.0 s
  state  [speed m/s, steer]  = [0.906244 0.277883]
  action [steer, throttle]   = [0.25381717 0.10874928]  <- what went to the motors (with the disturbance)
  label: the teacher's plan, 16 steps x [steer, throttle], 0.1 s apart:
    step  0 (+0.0 s)  [0.51100546 0.10874928]
    step  1 (+0.1 s)  [0.503968   0.10874928]
    step  2 (+0.2 s)  [0.48200908 0.10874928]
    step  5 (+0.5 s)  [0.41270152 0.10874928]
    step 10 (+1.0 s)  [0.362558   0.10874928]
    step 15 (+1.5 s)  [0.3386163  0.10874928]
  disturbance = applied - wanted = -0.257
```

**Đọc output:**

| Mảng | Shape | Nghĩa và đơn vị |
|---|---|---|
| `t` | (300,) | thời điểm của mỗi frame, tính bằng giây: 0.0, 0.1, ..., 29.9 |
| `images` | (300, 240, 320, 3) | ảnh RGB, kiểu `uint8` (0..255): cao 240, rộng 320 |
| `states` | (300, 2) | `[tốc độ m/s, góc lái hiện tại của bánh xe]`. Góc lái nằm trong [-1, 1], dương là rẽ trái |
| `actions` | (300, 2) | `[steer, throttle]` **đã đưa xuống bánh xe** ngay trước frame này, có cả nhiễu. Throttle âm là phanh |
| `plans` | (300, 16, 2) | **nhãn**: với mỗi frame, 16 bước `[steer, throttle]` mà teacher *muốn* làm, mỗi bước cách nhau 0,1 s |
| `valid` | (300,) | frame nào dùng được để train. Với teacher thì tất cả đều `True` |
| `source` | | `teacher` (nhãn do teacher nói ra) hoặc `actions` (nhãn suy ra từ hành động thật, dùng khi người lái hoặc ghi qua ROS 2) |

**Điều quan trọng nhất trong output:** ở frame 100, teacher muốn lái `+0.51` (rẽ trái
khá gắt), nhưng bánh xe nhận `+0.25` vì nhiễu đã kéo nó lệch đi `-0.26`. Model sẽ
**học theo `+0.51`**, chứ không phải `+0.25`. Đây là toàn bộ ý tưởng của DART, gói
trong một dòng số.

> **Kế hoạch không hằng số.** Steer giảm dần từ `+0.51` xuống `+0.34` trong 1,5 s. Nhiễu
> vừa đẩy xe lệch ra ngoài, nên trong kế hoạch, teacher bẻ mạnh để kéo xe về tâm đường
> trước, rồi chỉ còn bẻ vừa đủ để bám theo độ cong của đường. Throttle không đổi `0.109`
> vì xe đang ở tốc độ cân bằng: lực kéo vừa đủ bù lực cản.

## 1.5 Thống kê cả bộ dữ liệu

Một frame chưa nói lên được gì. Ta cần thống kê trên cả 12 000 frame. Đoạn dưới đây so
sánh `runs/hoc` với một bộ thu **không có nhiễu**. Bạn chưa có bộ `runs/hoc-perfect`
đâu: hoặc chạy `./drive.sh collect --project runs/hoc-perfect --steer-noise 0` trước
(1 phút), hoặc xóa nó khỏi danh sách trong đoạn code.

```bash
python3 - <<'EOF'
import numpy as np
from drive.learn.label import episodes, read_labels

for project in ("runs/hoc", "runs/hoc-perfect"):
    push, speed, wanted = [], [], []
    for ep in episodes(f"{project}/episodes"):
        data, labels = np.load(ep / "episode.npz"), read_labels(ep)
        push.append(data["actions"][:, 0] - labels["plans"][:, 0, 0])   # applied - wanted steering
        speed.append(data["states"][:, 0])
        wanted.append(labels["plans"][:, 0, 0])
    push, speed, wanted = map(np.concatenate, (push, speed, wanted))
    print(f"{project}: {len(push)} frames")
    print(f"  disturbance |applied - wanted|: median {np.median(np.abs(push)):.3f}, 95th percentile {np.percentile(np.abs(push), 95):.3f}")
    print(f"  wanted steering: {100 * (np.abs(wanted) > 0.3).mean():.0f} % of frames in sharp bends (|steer| > 0.3)")
    print(f"  speed below 0.5 m/s (starting off): {(speed < 0.5).sum()} frames = {100 * (speed < 0.5).mean():.1f} %")
EOF
```

```
runs/hoc: 12000 frames
  disturbance |applied - wanted|: median 0.170, 95th percentile 0.505
  wanted steering: 45 % of frames in sharp bends (|steer| > 0.3)
  speed below 0.5 m/s (starting off): 120 frames = 1.0 %
runs/hoc-perfect: 12000 frames
  disturbance |applied - wanted|: median 0.003, 95th percentile 0.007
  wanted steering: 45 % of frames in sharp bends (|steer| > 0.3)
  speed below 0.5 m/s (starting off): 120 frames = 1.0 %
```

**Đọc output:**

- **Nhiễu**: trung vị là 0,17 ở `runs/hoc` và 0,003 ở bộ không nhiễu. Cùng teacher, cùng
  track, cùng seed: khác biệt duy nhất là có nhiễu hay không.
- **45 % frame ở khúc cua gắt**: track `peanut` cong gần như liên tục. Một model chỉ
  biết lái thẳng sẽ sai ở gần nửa số frame.
- **Chỉ 1 % frame lúc xe đang xuất phát** (tốc độ dưới 0,5 m/s): mỗi episode chỉ có
  khoảng 1 s đầu để tăng tốc từ 0. Hãy nhớ con số này, vì ở bài 4 nó sẽ giải thích
  một hành vi lạ của model.

> **Nguyên tắc:** model chỉ giỏi ở những tình huống **chiếm đủ nhiều** trong dữ liệu.
> Tình huống nào chiếm 1 % thì model sẽ kém ở tình huống đó, kể cả khi các chỉ số
> trung bình trông rất đẹp.

## 1.6 `validate`: kiểm tra trước khi train

```bash
./drive.sh validate --project runs/hoc
```

```
40 episodes, 12000 frames, 19.9 min, images 240x320
  speed      0.0 to 1.74 m/s, mean 1.27
  steering   left 40 %, straight 21 %, right 39 %
  labels     12000 usable frames, from teacher
dataset: ok
```

Còn với bộ lái hoàn hảo:

```bash
./drive.sh validate --project runs/hoc-perfect
```

```
40 episodes, 12000 frames, 19.9 min, images 240x320
  speed      0.0 to 1.74 m/s, mean 1.27
  steering   left 39 %, straight 21 %, right 39 %
  labels     12000 usable frames, from teacher
  WARN  perfect driving only: the car never leaves the centerline, so a network trained on this cannot recover from its own errors (collect with --steer-noise)
dataset: ok, 1 warning(s)
```

`validate` kiểm tra hai mức. **FAIL** nghĩa là train trên bộ này sẽ sai hoặc không
train được. **WARN** nghĩa là nên xem lại. Lệnh trả về mã thoát 1 khi có FAIL.

| Kiểm tra | Mức | Bắt được lỗi gì |
|---|---|---|
| các mảng dài bằng nhau, có ít nhất 2 frame | FAIL | file ghi dở, bị cắt ngang |
| ảnh là `uint8 (N, H, W, 3)` | FAIL | lưu nhầm ảnh float hoặc ảnh xám |
| không có NaN hay vô cực | FAIL | cảm biến hỏng, phép chia cho 0 |
| action nằm trong [-1, 1], tốc độ ≥ 0 | FAIL | sai đơn vị (ví dụ lưu góc bằng độ) |
| thời gian tăng dần | FAIL | ghép nhầm hai file |
| tần số khớp `meta.json`, không có khoảng trống | WARN | rơi frame khi ghi qua ROS 2 |
| ảnh đứng yên trong lúc xe vẫn chạy | WARN | camera bị treo |
| tất cả episode cùng kích thước ảnh | FAIL | trộn hai loại camera |
| ít nhất 2 episode | FAIL | không còn gì để tách ra làm validation |
| ít hơn 2 000 frame | WARN | quá ít dữ liệu |
| xe đứng yên ở hơn 20 % số frame | WARN | ghi cả lúc không ai lái |
| **chỉ có lái hoàn hảo** | WARN | thiếu nhiễu DART, model sẽ không biết cách quay về |

> **`validate` không thấy được mọi thứ.** Nó không biết xe có đang nằm trên đường hay
> không (episode không lưu vị trí). Dòng "off the road" của `collect` ở mục 1.3 bù cho
> chỗ thiếu này. Nó cũng không biết nhãn có *đúng* hay không. Để kiểm tra điều đó, bạn
> phải nhìn bằng mắt.

## 1.7 `browse`: nhìn bằng mắt

```bash
./drive.sh browse --project runs/hoc --match 'sim-036'
```

Phím tắt: `d`/`a` sang frame sau/trước, `n`/`p` sang episode sau/trước, `space` để
chạy liên tục, `q` để thoát. Muốn xuất ra video thay vì mở cửa sổ (ví dụ trong container
không có màn hình), dùng
`./drive.sh browse --project runs/hoc --match 'sim-036' --video runs/hoc/sim-036.mp4`.

**Những gì cần tìm:**

1. **Đường cyan có bám theo con đường trong ảnh không?** Đường cyan là nhãn được "chạy
   thử" bằng mô hình xe rồi vẽ lại lên ảnh. Nhãn tốt thì đường cyan nằm trên mặt đường.
2. **Khi xe lệch khỏi tâm đường, đường cyan có cong về tâm không?** Đó chính là bài học
   "quay về" mà nhiễu DART tạo ra.
3. **Ảnh có lạ không?** Ví dụ ảnh đen, bị xé hình, hoặc chỉ toàn cỏ.

## 1.8 `status`: dự án đang ở đâu

```bash
./drive.sh status --project runs/hoc
```

```
project runs/hoc
  episodes   40: 12000 frames, 20.0 min
  labels     12000 usable frames (40 teacher)
  next       ./drive.sh validate, then ./drive.sh train v1
```

Dòng `next` gợi ý bước tiếp theo, đi theo thứ tự thu → gán nhãn → train → đánh giá.

## 1.9 Dữ liệu tốt hay xấu? Bảng kiểm

| Chỉ số | Xem ở đâu | Tốt | Đáng lo | Vì sao |
|---|---|---|---|---|
| FAIL | `validate` | 0 | ≥ 1 | không được phép train |
| Số frame | `validate` | ≥ 10 000 | < 2 000 (có WARN) | bài 5 (E3, E3b): track đơn giản này lái được với 3 000 frame nếu train đủ bước; bài toán khó hơn sẽ cần nhiều hơn |
| Cân bằng trái/phải | `validate`, dòng `steering` | chênh nhau ≤ 5 điểm % | chênh > 10 điểm | model sẽ rẽ lệch về một bên |
| Nhiễu (trung vị \|applied − wanted\|) | đoạn code ở mục 1.5 | 0,1–0,25 | < 0,02 (có WARN) | không có nhiễu thì model không biết quay về (bài 5) |
| Frame ngoài đường | `collect`, dòng cuối | < 5 % | > 5 % (có WARN) | model sẽ học lái trên cỏ (bài 5) |
| Tình huống hiếm | đoạn code ở mục 1.5 | mỗi tình huống quan trọng ≥ 5 % | ≈ 1 % | ví dụ xuất phát chỉ chiếm 1 %, và bài 4 cho thấy model xuất phát rất chậm |
| Đa dạng track | `--tracks` | nhiều track | 1 track | bài 4: train trên 1 track vẫn lái được cả 3 track, vì cả 3 cùng một kiểu đường; một con đường thật khác hẳn sẽ không như vậy |
| Nhãn bám đường | `browse` | đường cyan nằm trên mặt đường | đường cyan đâm ra cỏ | nhãn sai thì model học sai |

## 1.10 Bài tập

1. Thu 6 episode trên cả ba track, rồi chạy `validate`. Dòng `steering` thay đổi thế
   nào so với chỉ chạy `peanut`?
   ```bash
   ./drive.sh collect --project runs/bt1 --episodes 6 --tracks peanut,oval,trefoil
   ./drive.sh validate --project runs/bt1
   ```
2. Vì sao lúc train, dữ liệu validation được tách theo **cả episode** (4 episode cuối)
   chứ không bốc ngẫu nhiên 10 % số frame?
3. Ở frame 100 của `sim-000`, nếu train theo `actions` thay vì `plans` thì model học
   lái `+0.25` hay `+0.51`? Chuyện gì xảy ra khi model đó lái thật?
4. Chạy `./drive.sh collect --project runs/bt4 --episodes 4 --steer-noise 0.6` và đọc
   hai dòng cuối. Có `WARN` không? Vì sao không nên train trên bộ này?

<details><summary>Đáp án</summary>

1. Output đo được: `steering   left 36 %, straight 24 %, right 41 %`, so với 40/21/39 khi
   chỉ chạy `peanut`. Tỷ lệ `straight` tăng lên vì `oval` có hai đoạn thẳng dài. Trái và
   phải lệch nhau 5 điểm: với 6 episode, mỗi track chỉ được chạy một lần mỗi chiều, và
   tốc độ mục tiêu ngẫu nhiên khác nhau giữa các lần, nên chưa cân bằng được. Có thêm
   `WARN 1800 frames is little`: 6 episode là quá ít để train.
2. Hai frame cách nhau 0,1 s gần như giống hệt nhau. Nếu bốc ngẫu nhiên theo frame,
   gần như mọi frame validation đều có một "anh em sinh đôi" trong tập train, nên model
   chỉ cần nhớ là đạt điểm cao. Con số validation khi đó đẹp mà không có ý nghĩa gì.
   Tách theo episode mới đo được khả năng của model trên một lượt lái **mới**.
3. Model sẽ học `+0.25`, tức là học theo những gì đã xuống bánh xe, **kể cả nhiễu**.
   Nhưng còn một chuyện tệ hơn. Action ghi ở frame t là lệnh *vừa* được áp lên bánh xe,
   mà bánh xe (quay hơi trễ so với lệnh) lúc đó gần như đang ở đúng góc ấy. Góc hiện tại
   của bánh xe lại là một đầu vào của model (`state`). Model tìm ra một **lối tắt**: chép
   góc lái hiện tại là đoán đúng mà không cần nhìn đường. Đây là hiện tượng **copycat**.
   Bài 5 (E11) đo được đúng điều này: model chép trạng thái 0,77 cho mỗi 1,0 (con số của
   `v1` là 0,01), và khi lái thật thì lệch tâm nhiều gấp 2,7 lần. Đây là lý do `export`
   mặc định dùng `--action intended`.
4. Có: `257 frames (21.4 %) were recorded with the car off the road`, kèm `WARN`. Nhiễu
   0,6 mạnh tới mức teacher không kéo xe về kịp (xe lệch tới 261 cm), nên hơn một phần
   năm số frame được ghi lúc xe đang trên cỏ. Nhãn ở những frame đó (teacher vẫn biết đường nằm ở
   đâu) không thể suy ra được từ ảnh (ảnh toàn cỏ). Model bị buộc phải học một thứ
   không học được.
</details>

**Tiếp theo:** [Bài 2 · Giải phẫu model](02-giai-phau-model.md)
