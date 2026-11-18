# Bài 3 · Train và đọc log

**Mục tiêu.** Học xong bài này, bạn sẽ:

- train model `v1` và đọc được **từng dòng** log;
- biết loss, steer error và **mốc "kế hoạch trung bình"** nói lên điều gì;
- hiểu một bước train làm gì: batch, Adam, learning rate giảm dần, augmentation;
- dùng `probe history` để nhận ra 4 dạng đường cong: **khỏe, không học, học chưa tới,
  học thuộc**. Mỗi dạng đều có một ví dụ thật.

**Code:** [`drive/learn/train.py`](../../drive/learn/train.py) (khoảng 150 dòng).
**Lý thuyết sâu hơn:** [chương 10 · Model training](../study/10-model-training.md).

---

## 3.1 Train

```bash
./drive.sh train v1 --project runs/hoc
```

```
10800 training frames, 1200 validation frames, on mps
  the average plan, the same for every frame, scores validation 0.12114: the bar to clear
  epoch  1/15: train 0.10918   validation 0.09088   steer error 0.1844
  epoch  2/15: train 0.05806   validation 0.04448   steer error 0.0890
  epoch  3/15: train 0.02449   validation 0.02848   steer error 0.0575
  epoch  4/15: train 0.01820   validation 0.02196   steer error 0.0457
  epoch  5/15: train 0.01420   validation 0.01892   steer error 0.0408
  epoch  6/15: train 0.01261   validation 0.01733   steer error 0.0366
  epoch  7/15: train 0.01170   validation 0.01667   steer error 0.0355
  epoch  8/15: train 0.01098   validation 0.01558   steer error 0.0328
  epoch  9/15: train 0.01016   validation 0.01492   steer error 0.0310
  epoch 10/15: train 0.00949   validation 0.01451   steer error 0.0297
  epoch 11/15: train 0.00901   validation 0.01324   steer error 0.0265
  epoch 12/15: train 0.00858   validation 0.01290   steer error 0.0253
  epoch 13/15: train 0.00835   validation 0.01268   steer error 0.0248
  epoch 14/15: train 0.00820   validation 0.01266   steer error 0.0248
  epoch 15/15: train 0.00813   validation 0.01261   steer error 0.0246
323,124 parameters trained in 197.4 s; validation steer error 0.0246
  model: runs/hoc/models/v1.pt
```

Khoảng 2 phút trên GPU của Mac M3 khi máy rảnh: lần chạy đầu tiên mất 117,5 s, còn lần
chạy ở trên mất 197 s vì máy đang chạy song song một thí nghiệm khác. Trọng số của hai lần
**giống hệt nhau từng bit**: cùng seed, cùng dữ liệu thì cùng model, kể cả trên GPU của
Mac. Trên CPU, cùng lệnh này mất khoảng 10 phút
(`--device cpu`). Trên RTX 4060 khoảng 20 giây. Muốn train lại thì phải đặt một tên
khác: `train` **không bao giờ ghi đè** một model đã có.

## 3.2 Đọc từng dòng

| Dòng | Nghĩa |
|---|---|
| `10800 training frames, 1200 validation frames` | 36 episode đầu dùng để train. 4 episode cuối (`sim-036` đến `sim-039`) được **giữ riêng** để đo; model không bao giờ được train trên chúng (bài 1, bài tập 2). |
| `on mps` | Thiết bị được chọn tự động theo thứ tự: `cuda` (GPU NVIDIA), `mps` (GPU Mac), `cpu`. |
| `the average plan ... scores validation 0.12114: the bar to clear` | **Mốc so sánh.** Một "model" không nhìn ảnh, lúc nào cũng trả ra kế hoạch trung bình của dữ liệu train, sẽ có validation loss là 0,121. Model thật phải xuống **thấp hơn rất nhiều**. Không có mốc này thì bạn không biết 0,0126 là tốt hay xấu. |
| `train 0.00813` | **Train loss**: trung bình loss trên mọi batch của epoch đó, tính trên ảnh **đã augmentation**, và trong lúc trọng số vẫn đang thay đổi. |
| `validation 0.01261` | **Validation loss**: đo ở **cuối** epoch, trên 4 episode giữ riêng, không augmentation. Đây mới là con số đáng tin. |
| `steer error 0.0246` | Trung bình \|steer của model − steer của nhãn\| trên cả 16 bước, tính trên tập validation. Steer nằm trong [−1, 1], nên 0,0246 tương đương khoảng 1 % hành trình vô lăng. |
| `323,124 parameters trained in 197.4 s` | Số tham số (bài 2) và thời gian train. |
| `model: runs/hoc/models/v1.pt` | Checkpoint. Cạnh nó là `v1.json`: report gồm siêu tham số, mốc so sánh và loss của từng epoch. |

## 3.3 Loss là gì

Loss là **SmoothL1** (còn gọi là Huber) với `beta = 0.05`, tính trên cả 16 × 2 = 32 con
số của kế hoạch rồi lấy trung bình:

```
với mỗi con số:  e = |dự đoán − nhãn|
                 loss = 0.5 · e² / 0.05     nếu e < 0.05   (sai nhỏ: phạt nhẹ, theo bình phương)
                 loss = e − 0.025           nếu e ≥ 0.05   (sai lớn: phạt tuyến tính)
```

**Vì sao không dùng bình phương sai số (MSE)?** Với MSE, một frame sai 0,5 bị phạt nặng
gấp 100 lần một frame sai 0,05. Vài frame kỳ quặc (ví dụ lúc xe vừa bị nhiễu đẩy mạnh)
sẽ kéo cả model về phía chúng. SmoothL1 phạt những sai số lớn một cách "bình tĩnh" hơn.

**Vì sao in thêm steer error?** Loss trộn cả steer lẫn throttle, theo một công thức
không trực quan. Steer error thì có đơn vị dễ hiểu (một phần của vô lăng), và steer cũng
là thứ quyết định xe có nằm trên đường hay không.

## 3.4 Một bước train làm gì

Lõi của `train.py`, rút gọn:

```python
for epoch in range(1, epochs + 1):
    order = torch.randperm(len(images))              # xáo trộn thứ tự frame mỗi epoch
    for i in range(0, len(order), batch):            # batch = 128 frame, nên khoảng 85 bước mỗi epoch
        pick = order[i:i + batch]
        x, s, y = augment(images[pick], states[pick], plans[pick])   # lật ảnh, đổi độ sáng
        loss = loss_fn(model(x, s), y)               # 1. dự đoán, 2. so với nhãn
        optimizer.zero_grad()
        loss.backward()                              # 3. tính gradient: mỗi tham số nên đổi theo hướng nào
        optimizer.step()                             # 4. Adam đổi tham số một bước nhỏ, cỡ bằng learning rate
    schedule.step()                                  # learning rate giảm dần theo hình cosin: 0.001 -> 0
    val_loss, steer_error = check()                  # đo trên validation
```

- **Adam** với learning rate (lr) 0,001. Learning rate là "cỡ bước" mỗi lần đổi tham số.
  Bước quá lớn thì nhảy qua lại quanh đáy mà không xuống được. Bước quá nhỏ thì đi mãi
  không tới (mục 3.7).
- **Cosine schedule**: lr giảm dần về 0 ở epoch cuối, nên đường cong luôn **phẳng ra**
  ở cuối. Đừng nhầm "phẳng vì lr đã về 0" với "đã học xong".
- **Augmentation** ([`augment`](../../drive/learn/train.py)):
  - **Lật ảnh trái–phải** với một nửa batch, đồng thời **đảo dấu** steer trong cả
    trạng thái lẫn nhãn. Cua trái biến thành cua phải, nên lượng dữ liệu "miễn phí" tăng
    gấp đôi và model học được tính đối xứng (bài 4, kiểm tra mirror).
  - **Nhân độ sáng** với một số ngẫu nhiên trong 0,8–1,2. Bài 5 (E8) cho thấy ở dự án
    này, đây mới là phần quan trọng: dữ liệu vốn đã cân bằng trái/phải, nhưng không có
    augmentation thì model rất nhạy với độ sáng.

  Augmentation chỉ dùng lúc train. Không bao giờ dùng lúc đo hay lúc chạy.

## 3.5 Vì sao train loss và validation loss khác nhau

Nhìn lại log `v1`:

- **Epoch 1–2: validation *thấp hơn* train.** Train loss là trung bình của cả epoch,
  trong đó các batch đầu được tính lúc model còn rất kém. Validation thì được đo ở cuối
  epoch, khi model đã giỏi hơn. Thêm vào đó, augmentation làm dữ liệu train khó hơn.
- **Từ epoch 3: validation *cao hơn* train khoảng 1,5 lần.** Đây là điều bình thường:
  model làm tốt hơn trên những gì nó đã thấy. Tỷ lệ này chỉ đáng lo khi nó **tăng mãi**
  (học thuộc) hoặc **quá lớn** (trên 3 lần).

## 3.6 `probe history`: đọc đường cong

```bash
./drive.sh probe v1 history --project runs/hoc
```

```
10800 training frames, 1200 validation frames; 15 epochs, batch 128, lr 0.001, augmentation on, seed 0; 117.5 s on mps
  epoch train loss  val loss val/train steer err   validation loss
      1    0.10918   0.09088      0.83    0.1844   #########################
      2    0.05806   0.04448      0.77    0.0890   ############
      3    0.02449   0.02848      1.16    0.0575   ########
      4    0.01820   0.02196      1.21    0.0457   ######
      5    0.01420   0.01892      1.33    0.0408   #####
      6    0.01261   0.01733      1.37    0.0366   #####
      7    0.01170   0.01667      1.42    0.0355   #####
      8    0.01098   0.01558      1.42    0.0328   ####
      9    0.01016   0.01492      1.47    0.0310   ####
     10    0.00949   0.01451      1.53    0.0297   ####
     11    0.00901   0.01324      1.47    0.0265   ####
     12    0.00858   0.01290      1.50    0.0253   ####
     13    0.00835   0.01268      1.52    0.0248   ###
     14    0.00820   0.01266      1.54    0.0248   ###
     15    0.00813   0.01261      1.55    0.0246   ###
  the average plan, the same for every frame, scores validation 0.12114; this model ends at 0.01261, 10 % of that
  HEALTHY: both losses fell and flattened together; validation ended near its best and far below the average plan's
  Low validation error does not prove the model drives: ./drive.sh evaluate decides
```

`probe history` vẽ validation loss thành một thanh `###`, rồi đưa ra một hoặc nhiều nhận
định theo các quy tắc sau:

| Nhận định | Điều kiện | Nghĩa là | Thường sửa bằng |
|---|---|---|---|
| `HEALTHY` | không dính quy tắc nào bên dưới | cả hai loss cùng giảm rồi phẳng ra; validation kết thúc gần mức tốt nhất của nó, và rất thấp so với mốc | |
| `NOT LEARNING` | validation cuối > 50 % mốc "kế hoạch trung bình" | model gần như chỉ học được mức trung bình, không học được cách đọc ảnh | sửa learning rate; kiểm tra nhãn (bài 6.2) |
| `UNDERFITTING` | 25–50 % mốc | có học, nhưng học chưa tới | lr gần 0,001 hơn, thêm epoch, thêm dữ liệu |
| `STILL LEARNING` | validation giảm hơn 10 % trong 3 epoch cuối | vẫn đang tiến bộ thì bị dừng | thêm epoch |
| `OVERFITTING` | validation cuối cao hơn mức thấp nhất của nó trên 10 % | model bắt đầu **học thuộc** episode train | ít epoch hơn, thêm dữ liệu, giữ augmentation |
| `GAP` | validation > 3 × train | model biết dữ liệu train tốt hơn hẳn dữ liệu mới | dữ liệu đa dạng hơn |

Các ngưỡng 50 %, 25 %, 10 %, 3 lần là **quy tắc kinh nghiệm** cho dự án này, không phải
định luật.

> **`HEALTHY` chỉ nói về quá trình train, không nói về khả năng lái.** Model `p1` ở bài 5,
> train trên dữ liệu lái hoàn hảo, có validation chỉ bằng 8 % mốc và được chấm `HEALTHY`,
> vậy mà văng khỏi đường ở cả 6 lượt. Dòng cuối của `probe history` luôn nhắc điều này.

## 3.7 Bốn dạng đường cong, bốn ví dụ thật

Tất cả đều train trên **cùng dữ liệu** `runs/hoc` (trừ ví dụ d), mỗi lần chỉ đổi đúng
**một** tham số. Bài 5 sẽ cho các model này lái thật.

### a) Không học: learning rate quá lớn (`--lr 0.01`)

```bash
./drive.sh train v1-lr1e-2 --project runs/hoc --lr 0.01
./drive.sh probe v1-lr1e-2 history --project runs/hoc
```

```
10800 training frames, 1200 validation frames; 15 epochs, batch 128, lr 0.01, augmentation on, seed 0; 124.9 s on mps
  epoch train loss  val loss val/train steer err   validation loss
      1    0.09466   0.09229      0.97    0.1920   #############################
      2    0.07905   0.08690      1.10    0.1823   ############################
      3    0.07882   0.08729      1.11    0.1836   ############################
      4    0.07841   0.08626      1.10    0.1816   ###########################
      5    0.07837   0.08743      1.12    0.1842   ############################
      6    0.07827   0.08680      1.11    0.1830   ############################
      7    0.07822   0.08596      1.10    0.1816   ###########################
      8    0.07773   0.08685      1.12    0.1834   ############################
      9    0.07749   0.08581      1.11    0.1815   ###########################
     10    0.07732   0.08562      1.11    0.1810   ###########################
     11    0.07724   0.08576      1.11    0.1816   ###########################
     12    0.07711   0.08515      1.10    0.1803   ###########################
     13    0.07702   0.08502      1.10    0.1802   ###########################
     14    0.07689   0.08520      1.11    0.1807   ###########################
     15    0.07680   0.08503      1.11    0.1803   ###########################
  the average plan, the same for every frame, scores validation 0.12114; this model ends at 0.08503, 70 % of that
  NOT LEARNING: still 70 % of the average plan's loss; it learned little more than the average, not to read the picture. The learning rate (too high or too low), or the labels
  Low validation error does not prove the model drives: ./drive.sh evaluate decides
```

**Đọc:** loss có giảm (0,092 → 0,085), nên nếu chỉ nhìn loss bạn sẽ tưởng model đang
học. Nhưng so với mốc thì nó vẫn ở mức 70 %: model gần như chỉ trả ra kế hoạch trung
bình. Steer error 0,18 cũng xác nhận điều đó, vì baseline `average` ở bài 4 là 0,25.
Bước nhảy 0,01 quá lớn nên model không bao giờ "rơi" được vào vùng tốt. **Một đường
cong đang giảm chưa phải bằng chứng là model đang học. Hãy luôn so với mốc.**

### b) Học chưa tới: learning rate quá nhỏ (`--lr 0.0001`)

```
10800 training frames, 1200 validation frames; 15 epochs, batch 128, lr 0.0001, augmentation on, seed 0; 122.8 s on mps
  epoch train loss  val loss val/train steer err   validation loss
      1    0.13519   0.12206      0.90    0.2447   ###########################
      2    0.11552   0.12076      1.05    0.2425   ###########################
      3    0.11355   0.11578      1.02    0.2340   ##########################
      4    0.07280   0.06436      0.88    0.1260   ##############
      5    0.04493   0.05354      1.19    0.1060   ############
      6    0.03689   0.04676      1.27    0.0937   ##########
      7    0.03347   0.04525      1.35    0.0906   ##########
      8    0.03207   0.04443      1.39    0.0890   ##########
      9    0.03119   0.04284      1.37    0.0858   ##########
     10    0.03031   0.04332      1.43    0.0849   ##########
     11    0.02972   0.04201      1.41    0.0834   #########
     12    0.02928   0.04214      1.44    0.0828   #########
     13    0.02897   0.04154      1.43    0.0821   #########
     14    0.02883   0.04163      1.44    0.0818   #########
     15    0.02878   0.04144      1.44    0.0818   #########
  the average plan, the same for every frame, scores validation 0.12114; this model ends at 0.04144, 34 % of that
  UNDERFITTING: still 34 % of the average plan's loss; it learned a little. A learning rate nearer 0.001, more epochs, or more data
  Low validation error does not prove the model drives: ./drive.sh evaluate decides
```

**Đọc:** 3 epoch đầu gần như đứng yên (0,122 → 0,116), sau đó mới bắt đầu học. Tới lúc
learning rate về 0 thì validation mới chỉ xuống 34 % mốc, gấp 3 lần `v1`. Model *có*
học được (bài 5: nó vẫn lái được), nhưng học quá chậm so với số epoch được cho.

### c) Đang học thì dừng: quá ít epoch (`--epochs 5`)

```
10800 training frames, 1200 validation frames; 5 epochs, batch 128, lr 0.001, augmentation on, seed 0; 50.1 s on mps
  epoch train loss  val loss val/train steer err   validation loss
      1    0.10918   0.09088      0.83    0.1844   #########################
      2    0.05868   0.04546      0.77    0.0904   ############
      3    0.02577   0.03027      1.17    0.0617   ########
      4    0.01997   0.02641      1.32    0.0545   #######
      5    0.01841   0.02561      1.39    0.0519   #######
  the average plan, the same for every frame, scores validation 0.12114; this model ends at 0.02561, 21 % of that
  STILL LEARNING: validation loss fell more than 10 % over the last 3 epochs. More epochs may help (the learning rate decays to 0 at the last epoch, so curves flatten there anyway)
  Low validation error does not prove the model drives: ./drive.sh evaluate decides
```

**Đọc:** validation vẫn đang giảm mạnh, từ 0,045 xuống 0,026 trong 3 epoch cuối, thì
learning rate đã về 0. So với `v1` (15 epoch): steer error 0,052 so với 0,025.

### d) Học thuộc: khó gây ra hơn bạn nghĩ, và khó thấy hơn bạn nghĩ

**Overfitting** (học thuộc) là khi model nhớ những đặc điểm riêng của dữ liệu train thay
vì học quy luật chung. Dấu hiệu sách giáo khoa là validation loss đi xuống rồi **đi lên
lại** trong khi train loss vẫn giảm. Ta thử gây ra nó.

**Lần thử 1: ít dữ liệu, nhiều epoch.** 10 episode (2 700 frame train), 80 epoch thay vì
15 (bài 5, E3b). Kết quả: **không** học thuộc. Validation giảm đều tới epoch cuối, model
lái 6/6 với RMS 1,0 cm, ngang `v1`.

**Lần thử 2: cực ít dữ liệu, không augmentation, 200 epoch.** Chỉ 3 episode: 2 để train
(600 frame), 1 để validation.

```bash
./drive.sh collect --project runs/hoc-tiny --episodes 3
./drive.sh train t1 --project runs/hoc-tiny --epochs 200 --no-augmentation
./drive.sh probe t1 history --project runs/hoc-tiny
```

```
600 training frames, 300 validation frames; 200 epochs, batch 128, lr 0.001, augmentation off, seed 0; 151.2 s on mps
  epoch train loss  val loss val/train steer err   validation loss
      1    0.16204   0.13839      0.85    0.2594   ##########################
      2    0.13490   0.12188      0.90    0.2576   #######################
      3    0.12813   0.12032      0.94    0.2569   ######################
      4    0.12669   0.11845      0.93    0.2534   ######################
      5    0.12575   0.12071      0.96    0.2584   ######################
    ...
    198    0.00675   0.00864      1.28    0.0290   ##
    199    0.00675   0.00864      1.28    0.0290   ##
    200    0.00675   0.00864      1.28    0.0290   ##
  the average plan, the same for every frame, scores validation 0.12455; this model ends at 0.00864, 7 % of that
  HEALTHY: both losses fell and flattened together; validation ended near its best and far below the average plan's
  Low validation error does not prove the model drives: ./drive.sh evaluate decides
```

Vẫn `HEALTHY`: validation chỉ bằng 7 % mốc. Mô phỏng này rất "đều": cùng track, cùng
ánh sáng, cùng kiểu nhiễu, nên một episode mới gần như không có gì lạ với model. Vậy
model có học thuộc không? Cho nó lái thật:

```bash
./drive.sh evaluate t1 --project runs/hoc-tiny
```

```
t1, 100 ms per inference, 40 s per drive
track               laps   m/s  rms cm  max cm  off road  result
peanut              1.46  0.80     1.4     3.4     0.0 %  pass
peanut reversed     2.77  1.53     4.1     8.6     0.0 %  pass
oval                1.90  0.97     1.6     3.7     0.0 %  pass
oval reversed       2.78  1.42     2.6     6.7     0.0 %  pass
trefoil             1.59  0.95     1.8     4.2     0.0 %  pass
trefoil reversed    2.52  1.51     3.2     6.7     0.0 %  pass
6/6 drives passed
  report: runs/hoc-tiny/reports/t1.json

```

6/6, nhưng hãy nhìn cột `m/s`: chạy **theo chiều thuận** thì 0,80–0,97 m/s, còn **chiều
ngược** (`reversed`) thì 1,42–1,53 m/s. Model đã học rằng **chạy chiều ngược thì phải
nhanh**. Vì sao? Hai episode train của nó tình cờ là `peanut, 0.9 m/s` và
`peanut reversed, 1.8 m/s` (xem output `collect` ở bài 1). Với chỉ hai ví dụ, "chiều chạy
quyết định tốc độ" là một quy luật hoàn toàn khớp với dữ liệu. Đó chính là học thuộc:
nhớ một **sự trùng hợp** của dữ liệu train như thể nó là quy luật.

Còn validation thì không phát hiện ra, vì episode validation (`peanut, 1.0 m/s`, chiều
thuận, chạy chậm) **cũng khớp với sự trùng hợp đó**.

> **Bài học:** đường cong validation chỉ phát hiện được kiểu học thuộc nào làm validation
> tệ đi. Nếu tập validation cũng chung "sự trùng hợp" với tập train, nó sẽ không bao giờ
> thấy được. Muốn thấy thì phải kiểm tra trên những tình huống **khác** tập train: ở đây
> là chạy cả hai chiều (`evaluate`), hoặc một track khác.

**Lần thử 3: tập validation khác hẳn tập train.** 3 episode trên 3 track khác nhau. Hai
episode đầu (`peanut`, `oval`) dùng để train; episode cuối (`trefoil`) để validation.
Không augmentation, 200 epoch.

```bash
./drive.sh collect --project runs/hoc-3tracks --episodes 3 --tracks peanut,oval,trefoil
./drive.sh train o1 --project runs/hoc-3tracks --epochs 200 --no-augmentation
./drive.sh probe o1 history --project runs/hoc-3tracks
```

```
recording 3 drives of 30 s to runs/hoc-3tracks/episodes
  episode 1/3: peanut, 0.9 m/s
  episode 2/3: oval, 1.8 m/s
  episode 3/3: trefoil, 1.0 m/s
900 frames in 3 episodes; the disturbed car strayed up to 45 cm from the centerline
  1 frames (0.1 %) were recorded with the car off the road
```

```
600 training frames, 300 validation frames; 200 epochs, batch 128, lr 0.001, augmentation off, seed 0; 96.1 s on mps
  epoch train loss  val loss val/train steer err   validation loss
      1    0.15029   0.10011      0.67    0.1848   ####################
      2    0.10475   0.07410      0.71    0.1577   ###############
     50    0.01173   0.01591      1.36    0.0469   ###
    ...
     95    0.00508   0.01174      2.31    0.0374   ##
    100    0.00462   0.01267      2.74    0.0396   ###
    ...
    200    0.00367   0.01238      3.37    0.0386   ##
  the average plan, the same for every frame, scores validation 0.07051; this model ends at 0.01238, 18 % of that
  GAP: validation loss is 3.4x the training loss: the model knows the training episodes much better than new ones. More varied data
  Low validation error does not prove the model drives: ./drive.sh evaluate decides
```

**Đọc:** validation thấp nhất ở epoch 95 (0,01174), sau đó **đứng yên** (0,01238 ở epoch
200), trong khi train loss vẫn tiếp tục giảm (0,0051 → 0,0037). Tỷ số validation/train
tăng từ 1,2 lên 3,4 lần, và `probe history` báo `GAP`. Sau epoch 95, mọi tiến bộ của
model chỉ là học thêm chi tiết riêng của `peanut` và `oval`, không giúp gì cho `trefoil`.
Đây là hình ảnh gần với sách giáo khoa nhất mà dự án này tạo ra được. Validation không
tăng hẳn lên (chưa tới ngưỡng 10 % của `OVERFITTING`), nhưng 105 epoch cuối hoàn toàn vô
ích với dữ liệu mới.

Lái thật cho thấy thêm một điều:

```bash
./drive.sh evaluate o1 --project runs/hoc-3tracks
```

```
o1, 100 ms per inference, 40 s per drive
track               laps   m/s  rms cm  max cm  off road  result
peanut              0.87  0.48     2.0     3.9     0.0 %  pass
peanut reversed     1.76  0.93    12.4    28.9     0.0 %  pass
oval                1.33  0.68     2.6     5.5     0.0 %  pass
oval reversed       1.69  0.85     6.6    18.7     0.0 %  pass
trefoil             1.24  0.74     1.9     3.7     0.0 %  pass
trefoil reversed    1.40  0.83     6.2    12.8     0.0 %  pass
6/6 drives passed
  report: runs/hoc-3tracks/reports/o1.json

```

Theo chiều thuận, RMS là 1,9–2,6 cm. Theo **chiều ngược**, RMS lên tới 6,2–12,4 cm, và
có lúc xe lệch 28,9 cm (mép đường ở 45 cm). Cả 3 episode đều chạy chiều thuận: `collect`
chỉ đổi chiều sau mỗi vòng qua đủ các track (bài 1, output `collect`). Lại thêm đã tắt
augmentation, nên model **chưa từng thấy** một khúc cua theo chiều ngược. Episode
validation cũng chạy chiều thuận, nên không thể phát hiện ra điều này. Đây là lý do
augmentation lật ảnh vẫn quan trọng: ở bộ dữ liệu chính, vốn đã cân bằng trái/phải, nó gần
như không đem lại gì thêm (bài 5, E8); nhưng khi dữ liệu lệch như ở đây, chính nó bù lại
phần còn thiếu.

## 3.8 Các tham số của `train`

| Tham số | Mặc định | Tác dụng |
|---|---|---|
| `--epochs` | 15 | số lần đi qua toàn bộ dữ liệu |
| `--lr` | 0.001 | learning rate lúc bắt đầu; giảm dần về 0 theo hình cosin |
| `--batch` | 128 | số frame mỗi bước |
| `--no-augmentation` | (bật) | tắt lật ảnh và đổi độ sáng |
| `--seed` | 0 | khởi tạo trọng số và thứ tự xáo trộn. Đổi seed để đo xem kết quả **dao động** bao nhiêu (bài 5, E9) |
| `--device` | tự chọn | `cuda`, `mps` hoặc `cpu` |

Report `runs/hoc/models/<version>.json` ghi lại mọi tham số ở trên, nên bạn luôn biết
một model được train như thế nào:

```bash
python3 -c "import json; r = json.load(open('runs/hoc/models/v1.json')); print({k: v for k, v in r.items() if k != 'history'})"
```

```
{'frames': 10800, 'validation_frames': 1200, 'epochs': 15, 'batch': 128, 'lr': 0.001, 'augmentation': True, 'seed': 0, 'average_plan_val_loss': 0.12114, 'parameters': 323124, 'device': 'mps', 'seconds': 197.4}
```

## 3.9 Bài tập

1. Train `v1` lần thứ hai với `--seed 1`. Steer error có giống hệt không? Chênh lệch bao
   nhiêu? Điều đó nói gì về việc so sánh hai model chỉ khác nhau 0,002?
2. Ở ví dụ (a), train loss thấp hơn validation loss rất ít (tỷ lệ 1,1). Vì sao đây
   **không** phải là dấu hiệu tốt?
3. Vì sao không nên chọn số epoch bằng cách nhìn validation loss rồi dừng ở epoch tốt
   nhất, *rồi* báo cáo đúng con số validation đó như kết quả cuối cùng?
4. Tắt augmentation (`--no-augmentation`). Validation loss thay đổi thế nào? Sau đó chạy
   `probe … behaviour` và so dòng `mirror` với `v1`.

<details><summary>Đáp án</summary>

1. Không. Ở bài 5 (E9), ba seed cho ra 0,0246; 0,0212; 0,0226. Chênh lệch giữa hai model
   phải **lớn hơn hẳn** độ dao động này (ở đây khoảng ±0,002) thì mới chắc là thật.
2. Cả hai loss đều cao (70 % mốc). Model kém đều trên cả train lẫn validation: nó chưa
   học được gì để mà học thuộc. Khoảng cách train–validation nhỏ chỉ có ý nghĩa khi bản
   thân hai loss đã thấp.
3. Bạn đã dùng tập validation để **chọn** một thứ (số epoch). Con số đó vì vậy được tối
   ưu cho chính tập validation, và sẽ đẹp hơn thực tế. Muốn có con số trung thực, cần một
   tập thứ ba chưa từng dùng để chọn gì cả (tập test). Trong dự án này, `evaluate` (lái
   thật trên 3 track) đóng vai trò đó.
4. Validation loss cao hơn (0,0147 so với 0,0126; steer error 0,0307 so với 0,0246).
   `mirror` gần như không đổi (7 % so với 5 %), vì dữ liệu đã luân phiên chiều chạy. Khác
   biệt lớn nằm ở độ sáng: ảnh tối đi 20 % thì steer lệch 31 %, so với 2 % của `v1`. Xem
   bài 5, E8.
</details>

**Tiếp theo:** [Bài 4 · Phân tích output](04-phan-tich-output.md)
