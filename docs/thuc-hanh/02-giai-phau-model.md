# Bài 2 · Giải phẫu model

**Mục tiêu.** Học xong bài này, bạn sẽ:

- gọi tên được từng layer của `ChunkNet` và nói được mỗi layer làm gì;
- tự tính bằng tay **shape đầu ra** và **số tham số** của từng layer;
- hiểu **receptive field** (vùng nhìn) và vì sao nó quan trọng với việc lái xe;
- tự chạy model trên một frame và đọc được kế hoạch 16 bước mà nó trả ra.

**Cần có:** model `v1` từ [bài 3](03-train-va-doc-log.md). Bạn có thể đọc bài này trước
và chạy lệnh sau, hoặc train nhanh ngay bây giờ:
`./drive.sh train v1 --project runs/hoc` (khoảng 2 phút trên Mac M3).

**Code:** [`drive/learn/model.py`](../../drive/learn/model.py) (60 dòng). **Lý thuyết sâu
hơn:** [chương 10 · Model training](../study/10-model-training.md).

---

## 2.1 Bức tranh tổng thể

```
ảnh camera 240x320x3 (uint8)
   │  shrink (cv2.INTER_AREA): 240x320 -> 96x128, đúng như lúc train
   ▼
ảnh 96x128x3 ──► /255 − 0.5 ──► 5 lớp Conv2d + ReLU ──► Flatten ──► 768 số  ┐
                                    ("vision": nhìn)                        ├─► 770 số
trạng thái [tốc độ, góc lái] ──► [tốc độ/3, góc lái] ──────────────────► 2 số ┘
                                                                              │
                     3 lớp Linear (+ ReLU, cuối cùng là Tanh): "head", ra quyết định
                                                                              ▼
                                        32 số trong (−1, 1) ──► xếp thành 16 x [steer, throttle]
```

Kiến trúc này giống [PilotNet của NVIDIA](https://arxiv.org/abs/1604.07316): phần
**vision** biến ảnh thành một vector đặc trưng (feature vector), phần **head** ghép
vector đó với trạng thái xe rồi đưa ra kế hoạch.

## 2.2 Xem từng layer bằng `probe layers`

```bash
./drive.sh probe v1 layers --project runs/hoc
```

```
ChunkNet: 323,124 parameters. In: one 96x128 RGB frame (the camera's 240x320, shrunk) and the state [speed, steer]. Out: 16 steps x [steer, throttle], 0.1 s apart
  before the first layer: pixels/255 - 0.5 (so -0.5..0.5), channels first; speed / 3
  layer      what it is                       output shape       parameters   one unit sees
  input      image                            (1, 3, 96, 128)             0   1x1 px
  vision.0   Conv2d 3->24, 3x3, stride 2      (1, 24, 48, 64)           672   3x3 px of the 96x128 input
  vision.1   ReLU                             (1, 24, 48, 64)             0
  vision.2   Conv2d 24->36, 3x3, stride 2     (1, 36, 24, 32)         7,812   7x7 px of the 96x128 input
  vision.3   ReLU                             (1, 36, 24, 32)             0
  vision.4   Conv2d 36->48, 3x3, stride 2     (1, 48, 12, 16)        15,600   15x15 px of the 96x128 input
  vision.5   ReLU                             (1, 48, 12, 16)             0
  vision.6   Conv2d 48->64, 3x3, stride 2     (1, 64, 6, 8)          27,712   31x31 px of the 96x128 input
  vision.7   ReLU                             (1, 64, 6, 8)               0
  vision.8   Conv2d 64->64, 3x3, stride 2     (1, 64, 3, 4)          36,928   63x63 px of the 96x128 input
  vision.9   ReLU                             (1, 64, 3, 4)               0
  vision.10  Flatten                          (1, 768)                    0
  (concat)   vision features + state          (1, 770)                    0
  head.0     Linear 770->256                  (1, 256)              197,376
  head.1     ReLU                             (1, 256)                    0
  head.2     Linear 256->128                  (1, 128)               32,896
  head.3     ReLU                             (1, 128)                    0
  head.4     Linear 128->32                   (1, 32)                 4,128
  head.5     Tanh                             (1, 32)                     0
  (view)     reshape to the plan              (1, 16, 2)                  0
  vision (the 5 convolutions): 88,724 parameters (27 %); head (the 3 linear layers): 234,400 (73 %)
  the biggest layer is head.0 (Linear 770->256): 197,376 parameters, 61 % of the network
```

Shape có dạng `(batch, kênh, cao, rộng)` cho các layer ảnh và `(batch, số)` cho các
layer Linear. Batch ở đây là 1 vì ta đưa vào một frame.

## 2.3 Từng loại layer làm gì

| Layer | Làm gì | Trong model này |
|---|---|---|
| **Conv2d** (tích chập) | Trượt một bộ lọc 3x3 qua ảnh. Mỗi bộ lọc học phát hiện một mẫu (một cạnh, một vạch, một góc cong). Mỗi kênh đầu ra là bản đồ "mẫu này xuất hiện ở đâu". | 5 lớp, kênh tăng dần 24 → 36 → 48 → 64 → 64 |
| **stride 2** | Bộ lọc nhảy 2 pixel một lần, nên ảnh đầu ra **nhỏ đi một nửa** mỗi chiều. | 96x128 → 48x64 → 24x32 → 12x16 → 6x8 → 3x4 |
| **padding 1** | Viền thêm 1 pixel 0 quanh ảnh để bộ lọc 3x3 phủ được cả mép ảnh. | |
| **ReLU** | `max(0, x)`: giữ số dương, biến số âm thành 0. Đây là chỗ tạo ra tính phi tuyến; thiếu nó, 8 layer chồng lên nhau chỉ tương đương 1 layer tuyến tính. | sau mỗi Conv2d và Linear, trừ layer cuối |
| **Flatten** | Duỗi khối 64x3x4 thành một vector 768 số. | |
| **Linear** (fully connected) | Mỗi số đầu ra là một tổng có trọng số của **tất cả** các số đầu vào, cộng thêm bias. | 770 → 256 → 128 → 32 |
| **Tanh** | Ép mọi số vào khoảng (−1, 1), đúng bằng khoảng của steer và throttle. | layer cuối cùng |
| **view** | Xếp lại 32 số thành 16 bước × `[steer, throttle]`: số 0 là steer của bước 0, số 1 là throttle của bước 0, số 2 là steer của bước 1, ... | |

**Tiền xử lý trước layer đầu tiên.** Điểm ảnh được chia cho 255 rồi trừ 0,5, nên đầu
vào nằm trong [−0,5; 0,5]. Tốc độ được chia cho 3 (`SPEED_SCALE`), nên 3 m/s thành 1,0.
Mục đích là để mọi đầu vào có cùng độ lớn, giúp train ổn định hơn.

## 2.4 Tự tính bằng tay

**Shape đầu ra của Conv2d:** `out = floor((in + 2·padding − kernel) / stride) + 1`

```
vision.0:  cao  floor((96  + 2 − 3) / 2) + 1 = 48     rộng floor((128 + 2 − 3) / 2) + 1 = 64
vision.8:  cao  floor((6   + 2 − 3) / 2) + 1 = 3      rộng floor((8   + 2 − 3) / 2) + 1 = 4
Flatten:   64 kênh × 3 × 4 = 768;   ghép thêm [tốc độ, góc lái] → 770
```

**Số tham số:**

```
Conv2d:  kernel_h × kernel_w × kênh_vào × kênh_ra + kênh_ra (bias)
  vision.0:  3 × 3 × 3  × 24 + 24 =    672
  vision.2:  3 × 3 × 24 × 36 + 36 =  7 812
Linear:  vào × ra + ra (bias)
  head.0:    770 × 256 + 256      = 197 376
  head.4:    128 × 32  + 32       =   4 128
```

**Nhận xét quan trọng:** 5 lớp tích chập chỉ chiếm 27 % số tham số, còn **một mình
`head.0` chiếm 61 %**. Lý do là Conv2d dùng chung một bộ lọc nhỏ cho mọi vị trí trong
ảnh, còn Linear nối mọi đầu vào với mọi đầu ra. Bài 4 sẽ cho thấy hơn nửa số unit của
`head.1` không bao giờ hoạt động, nghĩa là layer to nhất cũng là layer lãng phí nhất.

## 2.5 Receptive field: mỗi unit "nhìn thấy" bao nhiêu

Một unit ở `vision.0` chỉ nhìn thấy 3x3 pixel. Một unit ở `vision.2` nhìn 3x3 unit của
`vision.0`, mà mỗi unit đó đã nhìn 3x3 pixel và cách nhau 2 pixel, nên tổng cộng nó
thấy 7x7 pixel. Công thức tổng quát:

```
r_mới = r_cũ + (kernel − 1) × j_cũ          r: vùng nhìn (pixel), bắt đầu từ 1
j_mới = j_cũ × stride                        j: bước nhảy tính theo pixel, bắt đầu từ 1

vision.0: r = 1 + 2×1  = 3,   j = 2
vision.2: r = 3 + 2×2  = 7,   j = 4
vision.4: r = 7 + 2×4  = 15,  j = 8
vision.6: r = 15 + 2×8 = 31,  j = 16
vision.8: r = 31 + 2×16 = 63, j = 32
```

**Ý nghĩa với việc lái xe:** ở layer cuối, mỗi unit thấy một vùng 63x63 trong ảnh
96x128, đủ lớn để thấy cả một đoạn vạch kẻ đường đang cong. Nhưng không unit nào thấy
được **cả ảnh**. Việc ghép "vạch trái ở đây, vạch phải ở kia" thành một quyết định là
việc của `head.0`, layer nối mọi vị trí lại với nhau. Đó cũng là lý do layer này cần
nhiều tham số đến vậy.

## 2.6 Chạy model trên một frame

```bash
python3 - <<'EOF'
import numpy as np
import torch
from drive.learn.dataset import shrink
from drive.learn.model import load, predict

model, info = load("runs/hoc/models/v1.pt")
print("checkpoint holds:", sorted(info), "| plan:", info["steps"], "steps every", info["plan_dt"], "s")
print(model)

ep = "runs/hoc/episodes/sim-036"
data, labels = np.load(f"{ep}/episode.npz"), np.load(f"{ep}/labels.npz")
i = 254
image = shrink(data["images"][i])                  # 240x320 -> 96x128, as in training
speed, steer = data["states"][i]
plan = predict(model, image, speed, steer)          # (16, 2)
print(f"\nframe {i}: speed {speed:.2f} m/s, wheels at {steer:+.2f}; image {data['images'][i].shape} -> {image.shape}")
print("step   model steer  label steer   model throttle  label throttle")
for k in (0, 3, 6, 9, 12, 15):
    print(f"{k:4d}   {plan[k, 0]:+11.3f}  {labels['plans'][i][k, 0]:+11.3f}   {plan[k, 1]:+14.3f}  {labels['plans'][i][k, 1]:+14.3f}")

# Look at what flows through the network: a forward hook on every layer prints its output.
x = torch.from_numpy(image)[None]
s = torch.tensor([[speed, steer]])
hooks = [m.register_forward_hook(lambda m, inp, out, n=n: print(f"  {n:10s} {type(m).__name__:8s} -> {tuple(out.shape)}"))
         for n, m in model.named_modules() if n.count(".") == 1]
print("\nshapes on the way through:")
with torch.no_grad():
    out = model(x, s)
print("  output", tuple(out.shape))
for h in hooks:
    h.remove()
EOF
```

```
checkpoint holds: ['image_hw', 'plan_dt', 'report', 'steps'] | plan: 16 steps every 0.1 s
ChunkNet(
  (vision): Sequential(
    (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(24, 36, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): ReLU()
    (4): Conv2d(36, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (5): ReLU()
    (6): Conv2d(48, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (7): ReLU()
    (8): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (9): ReLU()
    (10): Flatten(start_dim=1, end_dim=-1)
  )
  (head): Sequential(
    (0): Linear(in_features=770, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=32, bias=True)
    (5): Tanh()
  )
)

frame 254: speed 1.44 m/s, wheels at +0.41; image (240, 320, 3) -> (96, 128, 3)
step   model steer  label steer   model throttle  label throttle
   0        +0.841       +0.845           +0.162          +0.171
   3        +0.630       +0.604           +0.172          +0.267
   6        +0.399       +0.344           +0.182          +0.344
   9        +0.307       +0.255           +0.181          +0.215
  12        +0.303       +0.281           +0.179          +0.202
  15        +0.330       +0.300           +0.189          +0.200

shapes on the way through:
  vision.0   Conv2d   -> (1, 24, 48, 64)
  vision.1   ReLU     -> (1, 24, 48, 64)
  vision.2   Conv2d   -> (1, 36, 24, 32)
  vision.3   ReLU     -> (1, 36, 24, 32)
  vision.4   Conv2d   -> (1, 48, 12, 16)
  vision.5   ReLU     -> (1, 48, 12, 16)
  vision.6   Conv2d   -> (1, 64, 6, 8)
  vision.7   ReLU     -> (1, 64, 6, 8)
  vision.8   Conv2d   -> (1, 64, 3, 4)
  vision.9   ReLU     -> (1, 64, 3, 4)
  vision.10  Flatten  -> (1, 768)
  head.0     Linear   -> (1, 256)
  head.1     ReLU     -> (1, 256)
  head.2     Linear   -> (1, 128)
  head.3     ReLU     -> (1, 128)
  head.4     Linear   -> (1, 32)
  head.5     Tanh     -> (1, 32)
  output (1, 16, 2)
```

**Đọc output:**

- **Checkpoint** (`v1.pt`) chứa trọng số cùng `steps`, `image_hw`, `plan_dt` và `report`
  (log train). Nhờ vậy lúc chạy, model tự biết ảnh phải thu về kích thước nào và mỗi
  bước kế hoạch cách nhau bao lâu.
- **Steer** ở bước 0 gần như trùng nhãn (`+0.841` so với `+0.845`). Xe đang ở giữa một
  khúc cua trái gắt (bánh xe đang ở `+0.41`) và model muốn bẻ lái mạnh hơn nữa.
- **Càng xa thì sai càng nhiều.** Ở bước 6 (0,6 s tới), model nói `+0.399` còn nhãn là
  `+0.344`. Dự đoán tương lai xa luôn khó hơn. Bài 4 đo điều này trên 1 200 frame.
- **Throttle phẳng hơn nhãn.** Teacher tăng ga lên `+0.344` khi sắp ra khỏi cua, còn
  model giữ quanh `+0.18`. Model đã học được "mức trung bình" của throttle nhưng chưa
  học được *thời điểm* tăng ga.
- **Forward hook** là cách chuẩn để nhìn vào bên trong một model PyTorch mà không phải
  sửa code của nó. `probe layers` và `probe activations` đều dùng hook.

> **Lỗi kinh điển: tiền xử lý không khớp.** Nếu lúc chạy bạn đưa ảnh 240x320 vào thay
> vì ảnh đã shrink, Flatten sẽ ra 64×8×10 = 5 120 số, không khớp với 768, và PyTorch
> báo lỗi shape. Lỗi này *dễ* thấy. Lỗi *khó* thấy là thu nhỏ ảnh bằng cách khác
> (ví dụ dùng `INTER_LINEAR` thay vì `INTER_AREA`): shape vẫn đúng, model vẫn chạy,
> nhưng ảnh nó nhận khác một chút so với ảnh lúc train, và nó có thể lái kém đi mà
> không có thông báo lỗi nào. Vì vậy dự án chỉ có **một** hàm `shrink`, dùng chung cho
> cả train lẫn lúc chạy.

## 2.7 Bài tập

1. Tính số tham số của `vision.4` (Conv2d 36→48, 3x3) và `head.2` (Linear 256→128).
   So với bảng ở mục 2.2.
2. Nếu ảnh đầu vào là 192x256 thay vì 96x128, shape sau `vision.8` là bao nhiêu? `head.0`
   có bao nhiêu tham số? Tổng số tham số của model là bao nhiêu?
3. Bỏ `Tanh` ở cuối thì chuyện gì có thể xảy ra với steer và throttle?
4. Receptive field của `vision.8` là 63 pixel. Nếu thêm một Conv2d thứ sáu (3x3,
   stride 2) thì receptive field mới là bao nhiêu? Shape sau layer đó là gì?
5. Sửa đoạn code ở mục 2.6 để in steer của cả 16 bước cho frame 130 của `sim-038`.
   Frame này thuộc nhóm "tệ nhất" ở bài 4.

<details><summary>Đáp án</summary>

1. `vision.4`: 3·3·36·48 + 48 = 15 600. `head.2`: 256·128 + 128 = 32 896.
2. Mỗi chiều gấp đôi, nên sau 5 lần chia đôi ta được 6x8: shape `(1, 64, 6, 8)`,
   Flatten ra 3 072 số, cộng trạng thái thành 3 074. `head.0` có 3 074·256 + 256 =
   787 200 tham số. Tổng: 323 124 − 197 376 + 787 200 = 912 948. Chỉ riêng việc tăng
   độ phân giải đã làm model to gần gấp 3, và toàn bộ phần tăng nằm ở `head.0`.
3. Đầu ra không còn bị chặn. Model có thể ra steer bằng 1,7 chẳng hạn. Bộ giám sát
   an toàn vẫn sẽ cắt về [−1, 1], nhưng lúc train, loss sẽ phạt cả những giá trị vô
   nghĩa đó, và train kém ổn định hơn.
4. r = 63 + 2·32 = 127 pixel (lớn hơn cả ảnh), shape `(1, 64, 2, 2)`, Flatten ra 256.
5. Đặt `ep = "runs/hoc/episodes/sim-038"`, `i = 130`, và đổi vòng lặp thành
   `for k in range(16):`.
</details>

**Tiếp theo:** [Bài 3 · Train và đọc log](03-train-va-doc-log.md)
