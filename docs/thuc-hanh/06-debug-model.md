# Bài 6 · Debug model

**Mục tiêu.** Học xong bài này, bạn sẽ:

- khoanh vùng lỗi của một model lái tồi theo 4 tầng: **dữ liệu → train → model offline →
  vòng kín**;
- chạy được bài kiểm tra kinh điển "model có học thuộc được một batch nhỏ không";
- tra bảng **triệu chứng → nguyên nhân → lệnh kiểm tra → cách sửa**;
- dừng chương trình giữa chừng để xem tensor bằng `breakpoint()`;
- rút được bài học từ **ba lỗi thật** đã gặp khi viết chính bộ tài liệu này.

**Lý thuyết sâu hơn:** [chương 17 · Debugging the model](../study/17-debugging-the-model.md)
(tiếng Anh, có phần debug VLA và debug bằng VS Code).

---

## 6.1 Bốn tầng, đi từ dưới lên

Khi model lái tồi, đừng đoán. Đi từ tầng thấp nhất lên, và chỉ đi tiếp khi tầng dưới
đã sạch:

| Tầng | Câu hỏi | Lệnh | Sạch khi |
|---|---|---|---|
| **1. Dữ liệu** | Dữ liệu có đúng, đủ, đa dạng không? Nhãn có bám đường không? | `validate`, `browse`, đoạn thống kê ở bài 1.5 | không có FAIL; WARN đã được hiểu; nhãn cyan nằm trên mặt đường |
| **2. Train** | Model có học được không? Có học thuộc không? | đoạn code ở 6.2, `probe history` | học thuộc được một batch nhỏ; history `HEALTHY` |
| **3. Model offline** | Model sai ở đâu, sai kiểu gì? | `probe errors`, `behaviour`, `saliency` | tốt hơn baseline `keep` > 80 %; không có bias; đối xứng; nhìn đúng chỗ |
| **4. Vòng kín** | Khi tự lái thì sao? Bộ giám sát có can thiệp không? | `evaluate`, `sim --log`, `sim --show` | 6/6; tốc độ gần bằng teacher; `cmd_*` ≈ đầu ra động cơ |

**Vì sao đi từ dưới lên?** Lỗi ở tầng dưới **giả dạng** thành lỗi ở tầng trên. Dữ liệu
thiếu cảnh xuất phát (tầng 1) trông giống hệt "model chạy chậm" (tầng 4). Một phép đo
validation bị sai (tầng 2) trông giống hệt "model học thuộc". Mục 6.4 kể lại cả hai trường
hợp này.

## 6.2 Bài kiểm tra kinh điển: học thuộc một batch nhỏ

Trước khi train cả tiếng đồng hồ, hãy hỏi: **model có học được gì không?** Lấy 64
frame và train lặp đi lặp lại chỉ trên đúng 64 frame đó. Một model khỏe phải đưa được
loss về gần 0, tức là học thuộc lòng. Nếu không làm được, chắc chắn có lỗi trong đường
ống: nhãn lệch khỏi ảnh, tiền xử lý sai, learning rate vô lý, hoặc gradient không chảy
được.

```bash
python3 - <<'EOF'
import numpy as np
import torch
from drive.learn.dataset import shrink
from drive.learn.model import ChunkNet

torch.manual_seed(0)
ep = "runs/hoc/episodes/sim-000"
data, labels = np.load(f"{ep}/episode.npz"), np.load(f"{ep}/labels.npz")
pick = np.linspace(0, 299, 64).astype(int)                     # 64 frames spread over the drive
x = torch.from_numpy(np.stack([shrink(image) for image in data["images"][pick]]))
s = torch.from_numpy(data["states"][pick])
y = torch.from_numpy(labels["plans"][pick])

model = ChunkNet(steps=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.SmoothL1Loss(beta=0.05)
for step in range(301):
    loss = loss_fn(model(x, s), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        with torch.no_grad():
            steer_error = (model(x, s)[..., 0] - y[..., 0]).abs().mean().item()
        print(f"step {step:3d}  loss {loss.item():.5f}  steer error {steer_error:.4f}")
EOF
```

```
step   0  loss 0.14492  steer error 0.2381
step  50  loss 0.08190  steer error 0.1774
step 100  loss 0.02713  steer error 0.0684
step 150  loss 0.00538  steer error 0.0202
step 200  loss 0.00350  steer error 0.0127
step 250  loss 0.00214  steer error 0.0064
step 300  loss 0.00178  steer error 0.0047
```

Chạy trên CPU, mất khoảng 1–2 phút. Loss giảm 80 lần, steer error xuống 0,005: đường
ống thông suốt. **Hãy tự làm hỏng nó** để thấy bài kiểm tra bắt lỗi thế nào:

- **Xáo trộn nhãn**: thêm dòng `y = y[torch.randperm(64)]` ngay sau dòng tạo `y`. Mỗi
  ảnh giờ đi với nhãn của một ảnh khác, tức là vô nghĩa. Vậy mà model **vẫn học thuộc
  được**: chậm hơn ở giữa chừng (bước 150: steer error 0,075 so với 0,020), nhưng tới bước
  300 còn xuống tới 0,0037. 323 124 tham số thừa sức nhớ 64 cặp số ngẫu nhiên. Vì vậy bài
  kiểm tra này chỉ chứng minh được "đường ống chảy", chứ **không** chứng minh được "nhãn
  đúng". Để biết nhãn đúng, phải dùng `browse` và `validate`.
- **`lr=1.0`** (bước nhảy khổng lồ):
  ```
  step   0  loss 0.14492  steer error 0.8625
  step  50  loss 0.85002  steer error 0.8625
  ...
  step 300  loss 0.85002  steer error 0.8625
  ```
  Chỉ sau **một bước**, loss nhảy lên 0,85 rồi đứng im mãi mãi. Bước nhảy đầu tiên đã
  đẩy mọi unit sang vùng âm của ReLU (hoặc ép Tanh dính ở ±1), nên gradient bằng 0 và mạng
  "chết" hẳn: mọi ảnh đều cho ra cùng một kế hoạch. Không có NaN, không có thông báo lỗi
  nào. Loss **tăng** ngay từ bước đầu là dấu hiệu duy nhất.
- **`lr=1e-6`** (bước nhảy bé xíu):
  ```
  step   0  loss 0.14492  steer error 0.2402
  ...
  step 300  loss 0.14307  steer error 0.2395
  ```
  Loss chỉ nhích 1 % sau 300 bước. Model không hỏng gì cả, chỉ là đi quá chậm.

## 6.3 Bảng triệu chứng

| Triệu chứng | Nguyên nhân hay gặp | Kiểm tra bằng | Sửa |
|---|---|---|---|
| Steer error validation **tốt**, nhưng xe **văng khỏi đường** | dữ liệu chỉ có lái hoàn hảo (covariate shift) | `validate` có WARN "perfect driving only" | thu lại với `--steer-noise 0.25` (bài 5, E2) |
| Loss **kẹt** ở khoảng 0,08 từ epoch 2, steer error khoảng 0,18 | learning rate quá cao, model chỉ học được mức trung bình | `probe history`: `NOT LEARNING`; `probe errors`: không tốt hơn `keep` | `--lr 0.001` (bài 5, E5) |
| Loss giảm **rất chậm**, đứng yên vài epoch đầu, hoặc vẫn đang giảm thì hết epoch | learning rate quá thấp, quá ít epoch, hoặc quá ít bước vì dữ liệu ít | `probe history`: `UNDERFITTING` hoặc `STILL LEARNING` | tăng `--lr` hoặc `--epochs` (bài 5, E3b, E6, E7) |
| Validation loss **tăng** (hoặc đứng yên) trong khi train loss vẫn giảm | học thuộc (overfitting) | `probe history`: `OVERFITTING` hoặc `GAP` | ít epoch hơn, thêm dữ liệu đa dạng hơn, giữ augmentation (bài 3, mục d) |
| Validation đẹp nhưng lái tệ ở **một chiều** hoặc **một tình huống** | model học một sự trùng hợp của dữ liệu train mà validation cũng có | `evaluate` (chạy cả hai chiều, mọi track) | dữ liệu đa dạng hơn; augmentation (bài 3, mục d) |
| Validation loss **gấp nhiều lần** train loss ngay từ đầu | dữ liệu validation khác hẳn train, **hoặc phép đo sai** | so cùng checkpoint trên CPU và GPU (mục 6.4a) | sửa dữ liệu, hoặc sửa phép đo |
| Xe **xuất phát rất chậm**, tốc độ trung bình thấp | nhãn phụ thuộc tốc độ mục tiêu mà model không thấy được, cộng với ít frame xuất phát (1 %) | `probe errors` theo tốc độ; `probe behaviour` speed (`HOLDS`); `sim --log` | bỏ sự mơ hồ trong dữ liệu, hoặc cho model thấy tốc độ mục tiêu (bài 5, E10, E12) |
| Rẽ một phía **tốt hơn** phía kia | dữ liệu lệch trái/phải, hoặc tắt augmentation lật ảnh | `validate` dòng `steering`; `probe behaviour` mirror | cân bằng dữ liệu; bật augmentation |
| Model **chép góc lái hiện tại** (copycat); bước 0 sai nhiều hơn hẳn các bước sau | nhãn là action vừa áp lên bánh xe, gần trùng với `state` (bài 1, bài tập 3) | `probe behaviour`: `COPIES THE STATE`; `status`: nhãn `from actions` | dùng nhãn của teacher (bài 5, E11) |
| Ôm cua **hụt** (hoặc **lố**) một cách có hệ thống | dữ liệu lệch, hoặc nhãn lệch | `probe errors` cột `bias` > 0,05 | xem lại dữ liệu bằng `browse` |
| Lái **giật**, nhiều `steer_rate` | output dao động giữa các chunk | `sim --log`: cột `overrides`; `sim --show` | thêm dữ liệu; ở mức hệ thống, tăng `blend_s` trong `config.py` |
| Nhiều `stale` | độ trễ suy luận > `max_plan_age` | `sim --log` cột `plan_age_ms` | model nhanh hơn, hoặc đổi giới hạn (chương 05, 07) |
| Model **nhạy** với độ sáng | không có augmentation, hoặc augmentation hẹp hơn điều kiện thật | `probe behaviour` brightness | bật augmentation (bài 5, E8); mở rộng `augment` trong `train.py` |
| Steer **không đổi** dù ảnh khác nhau | cả mạng đã chết (ví dụ `lr=1.0` ở mục 6.2), hoặc model chỉ chép `state` | `probe activations`; `probe behaviour` steering state | giảm lr; xem lại đầu vào |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x5122 and 770x256)` | ảnh 240x320 chưa được shrink về 96x128: 64×8×10 + 2 = 5 122, không khớp với 770 | đọc hai shape trong thông báo lỗi; bài 2.6 | luôn dùng `dataset.shrink` |
| Có `nan` trong loss | lr quá cao, hoặc có NaN trong dữ liệu | `validate` (FAIL NaN); giảm lr | sửa dữ liệu / giảm lr |
| `status` báo model "not evaluated yet" dù đã chạy `evaluate` | (bug cũ, đã sửa) tên version có dấu chấm | mục 6.4b | cập nhật code |

## 6.4 Ba lỗi thật, và cách chúng bị bắt

Cả ba lỗi dưới đây đều đã xảy ra khi viết bộ tài liệu này, và đã được sửa. Chúng được
kể lại vì cách tìm ra chúng có giá trị hơn bản thân lỗi.

### a) Phép đo nói dối: steer error 0,155 trên GPU Mac

**Triệu chứng.** Train trên GPU Mac (MPS) cho steer error validation là **0,155**. Cùng
dữ liệu đó, train trên CPU chỉ cho **0,026**. Train loss ở cả hai lần gần như trùng nhau
(0,0081 và 0,0084).

**Khoanh vùng.**

1. Train loss bằng nhau nghĩa là quá trình **học** giống nhau. Vậy chỉ có phép **đo**
   validation là khác.
2. Đem checkpoint train trên MPS **đánh giá trên CPU**: steer error ra 0,0246, tức model
   hoàn toàn tốt. Vậy lỗi nằm ở phép đo trên MPS, không nằm ở model.
3. Phép đo validation đưa ảnh vào theo từng lát 512 frame: `held[0][i:i + 512]`. Đưa
   lát đầu tiên (i = 0) thì kết quả đúng, đưa lát thứ hai (i = 512) thì sai.
4. Thu nhỏ tới mức một dòng lệnh:
   `x_mps[512:1024].permute(0, 3, 1, 2).float()` khác với kết quả trên CPU tới 254 đơn
   vị. Hóa ra MPS (PyTorch 2.5) **bỏ qua offset** của một lát `uint8` đã permute khi ép
   kiểu sang float, và đọc dữ liệu từ frame 0.

**Sửa.** Trong `ChunkNet.forward`, ép kiểu trước rồi mới permute:
`images.float().permute(0, 3, 1, 2)`. Thêm một test chạy model trên một lát có offset
trên MPS và so với CPU.

**Bài học.** Khi một con số trông vô lý, **nghi ngờ phép đo trước khi nghi ngờ model**.
Và hãy đánh giá checkpoint bằng một đường đi khác (thiết bị khác, code khác) để xác nhận.

### b) Report bị ghi đè: `v1.2` thành `v1`

**Triệu chứng.** Train một model tên `v1.2`, chạy `evaluate v1.2`, rồi `status` vẫn báo
`v1.2: not evaluated yet`. Còn report của `v1` thì bị thay bằng số liệu của `v1.2`.

**Nguyên nhân.** `evaluate` đặt tên report bằng `Path("v1.2").stem`. Mà `stem` cắt bỏ
phần sau dấu chấm cuối cùng, vì nghĩ đó là đuôi file, nên `v1.2` thành `v1`.

**Sửa.** Lấy tên từ chính file checkpoint: `Path("models/v1.2.pt").stem` cho ra `v1.2`.
Thêm test với tên có dấu chấm.

**Bài học.** Lỗi kiểu này không bao giờ lộ ra với tên "đẹp" như `v1`, `v2`. Hãy **thử
với dữ liệu xấu xí** (tên có dấu chấm, dấu cách, chữ có dấu), và **kiểm tra file thật
trên đĩa** chứ không chỉ đọc output trên màn hình.

### c) Bài kiểm tra sai: "ga phải giảm khi tốc độ tăng"

**Triệu chứng.** Phiên bản đầu của `probe behaviour` báo model `v1` là
`NOT MONOTONIC`, vì ga tăng theo tốc độ.

**Nguyên nhân.** Bài kiểm tra sai, không phải model sai. Teacher giữ một tốc độ cố định
cho mỗi lượt lái. Xe nhanh hơn chịu lực cản lớn hơn (`0,3 × v`), nên cần nhiều ga hơn để
**giữ** tốc độ: khoảng 0,12 mỗi m/s. Model ra 0,10 mỗi m/s, tức là gần đúng.

**Bài học.** Một bài kiểm tra cũng là code, và cũng có thể sai. Trước khi tin nó, hãy
kiểm tra giả thuyết của nó bằng **vật lý của bài toán**.

## 6.5 Dừng chương trình để xem tensor: `breakpoint()`

Khi cần xem một tensor ở giữa vòng train, cách nhanh nhất là chèn `breakpoint()` vào
code rồi chạy lại. Ví dụ, trong [`drive/learn/train.py`](../../drive/learn/train.py), ngay
sau dòng `loss = loss_fn(model(x, s), y)`, thêm hai dòng (cùng mức thụt lề với dòng đó):

```python
            if epoch == 1 and i == 0:          # dừng một lần thôi: batch đầu của epoch đầu
                breakpoint()
```

Rồi chạy một epoch. Chương trình dừng ở dấu nhắc `(Pdb)`; gõ lần lượt các lệnh `p …`
dưới đây, cuối cùng là `c` để chạy tiếp:

```bash
./drive.sh train dbg --project runs/hoc --epochs 1
```

```
p x.shape, x.dtype, s.shape, y.shape
p float(x.min()), float(x.max())
p s[:2]
p y[0, :3]
p loss
c
```

Phiên làm việc thật:

```
10800 training frames, 1200 validation frames, on mps
  the average plan, the same for every frame, scores validation 0.12114: the bar to clear
> /Users/fanning/workspaces/vla-drive/drive/learn/train.py(107)train()
-> optimizer.zero_grad()
(Pdb) (torch.Size([128, 96, 128, 3]), torch.float32, torch.Size([128, 2]), torch.Size([128, 16, 2]))
(Pdb) (32.22325134277344, 255.0)
(Pdb) tensor([[ 0.4867, -0.5428],
        [ 1.4385, -0.5371]], device='mps:0')
(Pdb) tensor([[-0.3134,  1.0000],
        [-0.2388,  1.0000],
        [-0.1788,  1.0000]], device='mps:0')
(Pdb) tensor(0.1799, device='mps:0', grad_fn=<SmoothL1LossBackward0>)
(Pdb)   epoch  1/1: train 0.10918   validation 0.09088   steer error 0.1844
323,124 parameters trained in 21.7 s; validation steer error 0.1844
  model: runs/hoc/models/dbg.pt
```

**Đọc phiên này:**

- `x` là 128 ảnh 96×128×3 kiểu `float32`, giá trị từ 32 đến 255: augmentation đã đổi
  sang float và nhân độ sáng, nhưng **chưa** chia cho 255. Việc chia nằm trong
  `ChunkNet.forward` (bài 2.3). Nếu thấy giá trị 0–1 ở đây thì tức là đã chuẩn hóa hai lần.
- `s[:2]` là `[tốc độ, góc lái]` của hai frame đầu. `device='mps:0'` cho biết tensor đang
  ở trên GPU của Mac (trên máy NVIDIA sẽ là `cuda:0`).
- `y[0, :3]` là 3 bước đầu trong nhãn của frame đầu tiên: throttle `1.0`, tức teacher đang
  đạp ga hết cỡ. Đây là một trong số ít frame lúc xe xuất phát (bài 1.5).
- `loss` là 0,18 ở batch đầu tiên, **cao hơn** cả mốc "kế hoạch trung bình" (0,121), vì
  trọng số vẫn còn ngẫu nhiên. Sau một epoch, nó đã xuống 0,109 (dòng `epoch 1/1`).

Lệnh pdb hay dùng: `p <biểu thức>` để in giá trị, `n` để chạy dòng tiếp theo, `c` để
chạy tiếp tới lần dừng sau, `q` để thoát, `l` để xem code quanh chỗ đang dừng. **Nhớ xóa
hai dòng `breakpoint()` sau khi debug xong** (`git diff` sẽ nhắc bạn), và xóa model thử:
`rm runs/hoc/models/dbg.*`.

## 6.6 Checklist trước khi nói "model tốt"

- [ ] `validate`: không có FAIL, và mọi WARN đều đã được giải thích.
- [ ] `probe history`: `HEALTHY`, không `OVERFITTING`.
- [ ] `probe errors`: tốt hơn `keep` > 80 %; \|bias\| < 0,02; không có mức tốc độ hay
      loại đường nào sai gấp nhiều lần phần còn lại.
- [ ] `probe behaviour`: mirror < 10 %; độ sáng ×0,8/×1,2 < 5 %; không chép `state`.
- [ ] `evaluate`: 6/6, RMS < 2 cm, **tốc độ gần bằng teacher**.
- [ ] `sim --log`: `cmd_*` khớp với đầu ra động cơ, ngoại trừ vài tick đầu.
- [ ] Đã so sánh với ít nhất **2 seed** khác (bài 5, E9), để chắc rằng chênh lệch không
      phải do may rủi.

## 6.7 Bài tập

1. Một bạn báo: "model của mình có steer error 0,012, tốt gấp đôi `v1`, nhưng
   `evaluate` ra 0/6". Bạn sẽ chạy lệnh nào đầu tiên, và đoán nguyên nhân là gì?
2. Ở mục 6.2, vì sao việc học thuộc được 64 frame **với nhãn bị xáo trộn** lại là một
   bằng chứng rằng model đủ sức chứa?
3. `sim --log` cho thấy `cmd_steer` dao động từ +0,4 đến −0,4 sau mỗi 0,1 s, và cột
   `overrides` toàn là `steer_rate`. Lỗi nằm ở tầng nào? Bạn sẽ kiểm tra gì tiếp theo?

<details><summary>Đáp án</summary>

1. Chạy `./drive.sh validate` trên dữ liệu đã dùng để train. Rất có thể nó in ra
   `WARN perfect driving only`: model được train trên dữ liệu lái hoàn hảo. Đúng như thí
   nghiệm E2 ở bài 5 (steer error 0,0136, 0/6).
2. Model có 323 124 tham số mà chỉ phải nhớ 64 × 32 = 2 048 con số. Nó đủ sức nhớ
   ngay cả những con số vô nghĩa. Vì vậy học thuộc được một batch nhỏ là điều kiện
   **cần** (đường ống chảy) nhưng không **đủ** (nhãn đúng).
3. Tầng 4 cho thấy triệu chứng, nhưng nguyên nhân nằm ở tầng 3: mỗi chunk mới mâu thuẫn
   với chunk trước. Tiếp theo nên chạy `probe errors` để xem sai số ở bước 0, và
   `probe behaviour` steering state để xem model có đang phản ứng quá mạnh với góc lái
   hiện tại không. Đây là một vòng phản hồi: model đổi lái, góc lái hiện tại đổi theo,
   rồi model lại đổi lái ngược lại.
</details>

**Tiếp theo:** [Bài 7 · Debug và tối ưu như dân chuyên nghiệp](07-debug-chuyen-nghiep.md)
