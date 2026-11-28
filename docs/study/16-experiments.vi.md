# 16 · Thí nghiệm và phân tích kỹ thuật

*Bản tiếng Việt · [English](16-experiments.md)*

*Bản tiếng Việt · [English](16-experiments.vi.md)*

Các lab của [chương 15](15-hands-on-labs.vi.md) cho thấy hệ thống hành xử thế nào. Các thí nghiệm ở
đây hỏi *vì sao*, theo một khuôn bạn dùng lại được: một giả thuyết có thể hoá ra sai, mỗi lần chỉ
đổi một biến, một lệnh, một phép đo, và kết quả sẽ có ý nghĩa gì với một quyết định kỹ thuật.

## Luật chơi

1. **Viết giả thuyết và dự đoán trước.** Một thí nghiệm mà kết quả không thể làm bạn bất ngờ thì
   không dạy được gì.
2. **Chỉ đổi một biến.** Mọi thứ khác giữ ở giá trị mặc định, và bạn nói rõ điều đó.
3. **Không bao giờ ghi đè.** Dùng `--project runs/<thí-nghiệm>`; lưu ý `evaluate` thay
   `reports/<model>.json` ở mỗi lần chạy.
4. **Ghi lệnh cùng với kết quả.** Một con số không kèm lệnh chỉ là tin đồn.
5. **Một lần train là một giai thoại.** Mọi thứ dính tới `train` cần nhiều seed trước khi một khác
   biệt nhỏ mang ý nghĩa.

### Cái gì đã và chưa được đo

| Ký hiệu | Nghĩa |
|---|---|
| **Measured** | đã chạy cho giáo trình này ngày 2026-11-16; số liệu nằm trong bảng |
| **Repo** | được báo trong `docs/driving.md` hoặc lưu trong `runs/` |
| **Open** | chưa chạy; bảng là để bạn điền. Cố ý không đưa số kỳ vọng |

Một bảng kết quả cho các lần chạy của chính bạn nằm trong [bảng theo dõi tiến độ học](STUDY-TRACKER.vi.md).

---

## E1 · Độ trễ của policy

| | |
|---|---|
| **Giả thuyết** | Đọc chunk theo thời điểm observation che được độ trễ suy luận tới một giới hạn, quá giới hạn đó khả năng bám đường suy giảm đột ngột chứ không từ từ. |
| **Biến độc lập** | độ trễ: 0–400 ms; bù bật / tắt |
| **Giữ cố định** | teacher, 1,5 m/s, peanut, 30 s, watchdog nâng lên 1500 ms |
| **Lệnh** | `./drive.sh sweep --max-plan-age-ms 1500 --latencies 0,50,100,200,250,300,400` |
| **Phép đo** | rms và max của độ lệch ngang (cm) |

**Measured.**

| ms | có bù | không bù |
|---:|---:|---:|
| 0 | 1.8 / 3.5 | 1.6 / 3.0 |
| 50 | 1.8 / 3.5 | 1.0 / 2.4 |
| 100 | 1.8 / 3.4 | 0.7 / 2.2 |
| 200 | 2.0 / 3.6 | 3.8 / 8.8 |
| 250 | 3.1 / 6.6 | 10.0 / 23.3 |
| 300 | 20.2 / 31.0 | 27.2 / 54.4 |
| 400 | 65.2 / 123.9 | 49.0 / 89.0 |

**Diễn giải.** Được ủng hộ: phẳng tới 200 ms, một vách đứng giữa 250 và 300 ms. Hình dạng này
quan trọng cho thiết kế: không có sự suy giảm êm ái nào để dựa vào, nên ngân sách độ trễ cần có
biên, và một model có độ trễ sát vách sẽ rơi xuống khi board giảm xung. Lợi thế của việc không bù
ở 50–100 ms là đặc tính của teacher này ([chương 5](05-action-chunking-and-latency.vi.md)).

**Tiếp theo (open).** Tinh chỉnh vị trí vách: `--latencies 250,260,270,280,290,300`.

---

## E2 · Tốc độ xe so với độ trễ

| | |
|---|---|
| **Giả thuyết** | Độ trễ chịu được không phải một khoảng thời gian cố định. Nó tỷ lệ với $\ell/v$ (khoảng nhìn trước chia tốc độ), nên thứ làm hỏng vòng lặp xấp xỉ là một quãng đường đi trên thông tin cũ. |
| **Biến độc lập** | tốc độ và độ trễ |
| **Lệnh** | `./drive.sh sweep --speed S --max-plan-age-ms 1500 --latencies …` |
| **Phép đo** | rms khi có bù (cm) |

**Measured.**

| Tốc độ | 100 ms | 250 ms | 400 ms | 650 ms | $v \cdot L$ ở điểm ổn định cuối → mất ổn định đầu |
|---:|---:|---:|---:|---:|---|
| 0.6 m/s | 0.7 | 0.8 | 1.6 | 35.1 | 0.24 → 0.39 m |
| 1.0 m/s | 1.1 | 1.6 | 27.7 | | 0.25 → 0.40 m |
| 1.5 m/s | 1.8 | 3.1 | 65.2 | | 0.375 → 0.45 m (300 ms: 20,2 cm) |

**Diễn giải.** Phù hợp với một ngưỡng gần $v \cdot L \approx 0.38$ m cho teacher này, và phù hợp
về xu hướng với ước lượng $\tau^* \approx 0.52\,\ell/v$ ở [chương 6](06-control-theory.vi.md). Các
lần quét quá thô để phân biệt một $v \cdot L$ không đổi với tỷ lệ theo $\ell/v$ (chúng khác nhau
vì $\ell$ tăng theo $v$).

**Tiếp theo (open).** Lưới đo sẽ tách được hai mô hình:

```bash
for s in 0.6 0.8 1.0 1.2 1.5; do
  echo "== $s"; ./drive.sh sweep --speed $s --max-plan-age-ms 1500 --latencies 200,250,300,350,400,450,500,550,600
done
```

**Dùng vào kỹ thuật.** Một luật để chọn giới hạn tốc độ từ độ trễ đo được của bất kỳ model nào
được triển khai.

---

## E3 · Tuổi kế hoạch tối đa

| | |
|---|---|
| **Giả thuyết** | Xe bị phanh mỗi khi tuổi kế hoạch vượt giới hạn; vì tuổi quét trong $[L, 2L)$, tỷ lệ thời gian bị phanh là $(2L - A)/L$ khi $L < A < 2L$, bằng 0 khi dưới, bằng 1 khi trên. |
| **Biến độc lập** | độ trễ, ở giới hạn mặc định $A = 0.6$ s; sau đó là giới hạn, ở một độ trễ cố định |
| **Lệnh** | `./drive.sh sweep --latencies 0,100,200,250,400,650` và `./drive.sh sim --latency-ms 400 --max-plan-age-ms A` |
| **Phép đo** | số giây `stale` trên 30; số vòng |

**Measured** (giới hạn mặc định).

| Độ trễ | Tỷ lệ bị phanh dự đoán | `stale` (s trên 30) | Số vòng |
|---:|---:|---:|---:|
| 250 ms | 0 | 0,3 (lúc khởi động) | 1.97 |
| 400 ms | 0,5 (0,45 trên lưới 20 ms) | 13.9 | 0.14 |
| 650 ms | 1 | 30.0 | 0.00 |

**Diễn giải.** Được ủng hộ. Để ý sự đánh đổi mà giới hạn thể hiện: ở 400 ms với 1500 ms, xe không
bị phanh và ra khỏi đường (65 cm rms); với 600 ms nó bị phanh một nửa thời gian và ở lại trên
đường (14,3 cm rms trong lần quét mặc định là trên 0,14 vòng chạy-dừng). Watchdog biến một kiểu
hỏng không an toàn thành một kiểu hỏng an toàn. Nâng nó lên để tắt cảnh báo là đảo ngược điều đó.

**Tiếp theo (open).** Ở 400 ms, quét `--max-plan-age-ms 600,700,800,900` và ghi cả `stale` lẫn
`off_track_pct`. Có thiết lập nào vừa chạy được vừa ở trên đường ở 1,5 m/s không?

---

## E4 · Độ dài chunk (cạn chunk)

| | |
|---|---|
| **Giả thuyết** | Một policy có horizon chunk $H$ ngắn hơn $2L$ sẽ bị phanh vì thiếu kế hoạch, độc lập với watchdog. |
| **Biến độc lập** | độ trễ, cho một policy có $H = 0.47$ s |
| **Lệnh** | `./drive.sh sim --policy lane-camera --latency-ms L --max-plan-age-ms 1500` |
| **Phép đo** | số giây `stale` khi watchdog đã được dẹp sang bên |

**Measured.**

| Độ trễ | $2L$ so với $H = 0.47$ | `stale` | Tốc độ trung bình |
|---:|---|---:|---:|
| 100 ms | 0.2 < 0.47 | 0.1 s | 0.86 m/s |
| 300 ms | 0.6 > 0.47 | 12.2 s | 0.18 m/s |

**Diễn giải.** Được ủng hộ. Yêu cầu $H \ge 2L$ là một ràng buộc cứng khi ghép một model với một
board, tách biệt với tính ổn định.

**Tiếp theo (open).** Dựng `LaneCameraPolicy(…, chunk_size=30)` trong một script và lặp lại ở
300 ms. Việc phanh phải biến mất. Khả năng bám đường có giữ được không?

---

## E5 · Nhiễu tay lái khi thu dữ liệu

| | |
|---|---|
| **Giả thuyết** | Thành công trong vòng kín cần dữ liệu hồi phục. Sai số validation offline sẽ *không* đi theo thành công trong vòng kín, và có thể đi ngược lại. |
| **Biến độc lập** | `--steer-noise` |
| **Giữ cố định** | 40 episode, peanut, seed 0, 15 epoch |
| **Lệnh** | bên dưới |
| **Phép đo** | `val_steer_error` cuối cùng; số lần lái đạt; rms; `max_lateral_m` do `collect` báo |

```bash
for n in 0 0.05 0.1 0.25 0.5; do
  p=runs/noise-$n
  make collect  ARGS="--project $p --steer-noise $n"
  make validate ARGS="--project $p"
  make train    V=v1 ARGS="--project $p"
  make evaluate V=v1 ARGS="--project $p"
done
```

**Repo** (hai trong năm điểm).

| `--steer-noise` | Sai số steer validation | Số lần lái đạt | Peanut |
|---:|---:|---:|---|
| 0 | 0.015 | 0/6 | ra khỏi đường 75 % thời gian |
| 0.25 | 0.024 | 6/6 | 1,5 cm rms |

**Open:** 0,05, 0,1, 0,5.

**Diễn giải hai điểm đã biết.** Được ủng hộ mạnh: thứ tự theo sai số offline ngược hẳn thứ tự
theo khả năng lái.

**Những câu hỏi các điểm còn mở sẽ trả lời.** Có một ngưỡng nhiễu mà dưới đó model không lái
được, hay là cải thiện dần dần? Quá nhiều nhiễu có gây hại không (xe ra khỏi đường trong lúc thu
dữ liệu; các khung cảnh trở nên khác hẳn mọi thứ nó sẽ thấy khi lái tốt)? Mức nhiễu nhỏ nhất đủ
dùng là bao nhiêu, điều quan trọng khi "teacher" là một người phải chịu đựng nhiễu đó?

---

## E6 · Kích thước bộ dữ liệu, độ đa dạng và seed

| | |
|---|---|
| **Giả thuyết** | (a) Thêm episode trên cùng đường đua cải thiện khả năng bám đường nhưng lợi ích giảm dần. (b) Thêm các đường đua khác giúp nhiều hơn thêm cùng lượng dữ liệu trên cùng đường đua. (c) Khác biệt giữa `v1` và `v2` nằm trong mức dao động giữa các seed. |
| **Biến độc lập** | `--episodes`; `--tracks`; `--seed` của `train` |
| **Phép đo** | số lần lái đạt; rms trung bình và max tệ nhất trên sáu lần lái |

**Repo** (mỗi thứ một lần chạy).

| Model | Dữ liệu train | Đạt | rms | Tệ nhất |
|---|---|---:|---:|---:|
| `v1` | 40 episode mô phỏng, peanut | 6/6 | 1.3 cm | 4.0 cm |
| `v2` | như trên + 2 bản ghi ROS (oval, trefoil) | 6/6 | 1.2 cm | 3.8 cm |

**Open.** Các lần chạy sẽ làm những phép so sánh này có ý nghĩa:

```bash
# (c) dao động theo seed trên dữ liệu giống hệt nhau
for s in 0 1 2 3 4; do make train V=seed$s ARGS="--project runs/lab10 --seed $s"; done
make evaluate V="seed0 seed1 seed2 seed3 seed4" ARGS='--project runs/lab10'

# (a) kích thước
for e in 5 10 20 40 80; do
  make collect ARGS="--project runs/size-$e --episodes $e"; make train V=v1 ARGS="--project runs/size-$e"
  make evaluate V=v1 ARGS="--project runs/size-$e"
done

# (b) độ đa dạng với cùng kích thước
make collect ARGS='--project runs/div-1 --episodes 42'
make collect ARGS='--project runs/div-3 --episodes 42 --tracks peanut,oval,trefoil'
```

**Lưu ý cho (b).** `evaluate` lái trên đúng ba hình dạng đó. Một model train trên cả ba không còn
được test trên đường đua chưa thấy; để giữ một bài test riêng, train trên hai và đọc kết quả trên
đường thứ ba.

**Diễn giải.** Cho tới khi làm xong (c), "v2 tốt hơn v1" vẫn chưa được xác lập
([chương 11](11-model-evaluation.vi.md)).

---

## E7 · Tổng quát hoá qua các đường đua

| | |
|---|---|
| **Giả thuyết** | Một mạng train trên một đường đua đã học được việc bám đường cục bộ, không phải học thuộc vòng đua, nên lái được các hình dạng và chiều chạy chưa từng thấy. |
| **Lệnh** | `make evaluate V=v1` |
| **Phép đo** | đạt và rms theo từng đường đua và chiều chạy |

**Repo.** `v1` (chỉ peanut): rms 1,5 / 1,4 cm trên peanut (xuôi / ngược), 1,7 / 1,3 trên oval,
1,0 / 0,8 trên trefoil. 6/6.

**Diễn giải.** Được ủng hộ, trong bộ mô phỏng này. Các đường đua chưa thấy lại *dễ hơn* đường đua
train (cua thoải hơn: bán kính hẹp nhất 1,44 và 1,83 m so với 1,05 m), nên điều này cho thấy tổng
quát hoá sang hình học nhẹ nhàng hơn, không phải khó hơn.

**Tiếp theo (open).** Thêm một hình dạng khó hơn vào `SHAPES` (bán kính hẹp nhất vừa trên 1 m, có
khúc cua chữ S) trong một bản sao nháp và đánh giá `v1` trên đó. Và train chỉ trên oval, rồi đánh
giá trên peanut: tổng quát hoá có đi từ dễ sang khó không?

---

## E8 · Độ trễ với một policy học được

| | |
|---|---|
| **Giả thuyết** | Một kế hoạch học được chịu độ trễ kém hơn kế hoạch của teacher, vì dự đoán của nó càng về cuối chunk càng kém. |
| **Lệnh** | trong container: `./drive.sh sim --policy cnn --checkpoint v1 --latency-ms L --max-plan-age-ms 1500 --duration-s 40` |
| **Phép đo** | rms / max, % ra khỏi đường, ở tốc độ của chính mạng (khoảng 0,84 m/s) |

**Measured.**

| Độ trễ | rms / max | Ra khỏi đường |
|---:|---:|---:|
| 0 ms | 1.3 / 2.8 cm | 0 % |
| 100 ms | 1.5 / 4.0 cm | 0 % |
| 200 ms | 2.5 / 5.9 cm | 0 % |
| 300 ms | 22.6 / 54.5 cm | 6.2 % |
| 400 ms | 259.8 / 825.5 cm | 75.2 % |

**Diễn giải.** Được ủng hộ về kết cục: ở 0,84 m/s mạng hỏng ở 300 ms, trong khi teacher ở 1,0 m/s
giữ 1,6 cm ở 250 ms và ở 0,6 m/s giữ 1,6 cm ở 400 ms. Nguyên nhân chưa được tách riêng: chất
lượng dự đoán và bước kế hoạch thô hơn (0,1 s so với 1/30 s) đang lẫn vào nhau.

**Dùng vào kỹ thuật.** Ngân sách độ trễ phải được đo với chính policy sẽ triển khai. Với một VLA,
đây là thí nghiệm cần chạy trong mô phỏng, ở độ trễ đo được trên Jetson, trước mọi bài test phần
cứng.

**Tiếp theo (open).** Tách hai nguyên nhân: cho teacher một chunk 16 × 0,1 s
(`PurePursuitPolicy(…, chunk_size=16, chunk_dt_s=0.1)`) và lặp lại lần quét của nó ở 0,84 m/s.

---

## E9 · Hành vi an toàn

| | |
|---|---|
| **Giả thuyết** | Supervisor dừng xe tại biên an toàn bất kể tốc độ tiếp cận, vì tốc độ cho phép được suy ra từ khoảng trống còn lại. |
| **Biến độc lập** | tốc độ tiếp cận |
| **Lệnh** | `./drive.sh sim --obstacle 8 --speed S` |
| **Phép đo** | `min_range_m`, `collided`, số giây cho mỗi luật vật cản |

**Measured.**

| Tốc độ | Gần nhất | Đâm | `slow` / `brake` |
|---:|---:|---|---:|
| 0.8 m/s | 0.30 m | không | 0.12 / 0.12 s |
| 1.5 m/s | 0.30 m | không | 0.22 / 0.22 s |
| 1.5 m/s, `--no-safety` | 0.00 m | **có** | |

**Diễn giải.** Được ủng hộ ở hai tốc độ này.

**Tiếp theo (open).** Những gì bộ mô phỏng không cho bạn biết được, mà một chiếc xe thật sẽ cho:
(1) hạ mức phanh *thật* xuống dưới mức 4 m/s² đã lập kế hoạch bằng cách dựng
`World(vehicle=VehicleParams(max_brake_mps2=3.0))` rồi lặp lại: giờ xe dừng ở đâu? (2) Đặt cọc
tiêu lệch tâm bằng `track.obstacle(8.0, lateral_m=…)`: ở độ lệch nào thì nón ±15° của cảm biến
bỏ sót nó trong khi thân xe vẫn đâm vào? (3) Đưa vào một số đo khoảng cách NaN
([chương 7](07-safety-supervisor.vi.md)).

---

## E10 · Hồi phục từ một độ lệch

| | |
|---|---|
| **Giả thuyết** | Một model train với tay lái bị làm nhiễu quay về được đường tâm từ một điểm xuất phát lệch tâm và lệch hướng; một model train trên dữ liệu lái hoàn hảo thì không. |
| **Biến độc lập** | độ lệch ngang và độ lệch hướng lúc xuất phát; model |
| **Lệnh** | script ở [chương 11](11-model-evaluation.vi.md), mục "Thí nghiệm thực hành", 10 s bắt đầu từ 5 m dọc peanut |
| **Phép đo** | rms, max, % ra khỏi đường trong 10 s |

**Measured** (trong container).

| Model | Xuất phát: độ lệch, hướng | rms | max | Ra khỏi đường |
|---|---|---:|---:|---:|
| `v1` (dữ liệu bị làm nhiễu) | 25 cm, 0.3 rad | 19.3 cm | 31.3 cm | 0 % |
| `v1` | 35 cm, 0.5 rad | 33.6 cm | 50.6 cm | 24 % |
| chỉ lái hoàn hảo | 25 cm, 0.3 rad | 91.4 cm | 302 cm | 61.5 % |
| chỉ lái hoàn hảo | 35 cm, 0.5 rad | 132 cm | 429 cm | 57.5 % |

**Diễn giải.** Được ủng hộ, với một lưu ý mà bài đánh giá chuẩn che mất: khả năng hồi phục của
`v1` là thật nhưng **có giới hạn**. Từ 25 cm và 0,3 rad nó ở lại trên đường; từ 35 cm và 0,5 rad
nó ra khỏi đường. Dữ liệu train của nó đặt xe trong khoảng ±15 cm và ±0,2 rad lúc xuất phát, cộng
thêm nhiễu. "6/6" không nói gì về điều này.

**Tiếp theo (open).** Vẽ ranh giới: một lưới các độ lệch và hướng, đạt hay trượt. Sau đó collect
với khoảng xuất phát rộng hơn (nó được viết cứng trong `collect`; đổi trong một bản sao nháp) hoặc
nhiều nhiễu hơn, train lại, và xem ranh giới có dịch chuyển không.

---

## E11 · Dao động độ trễ và thread thật

| | |
|---|---|
| **Giả thuyết** | Vì executor đánh chỉ số chunk theo thời điểm observation, độ trễ thay đổi không cần xử lý đặc biệt, cho tới khi đuôi chậm của phân phối vượt qua watchdog. |
| **Lệnh** | `./drive.sh sim --jitter-ms 40`; `./drive.sh sim --latency-ms 250 --jitter-ms 40`; `./drive.sh sim --duration-s 5 --realtime` |
| **Phép đo** | rms; `stale`; số lượt suy luận và độ trễ trung bình |

**Measured.**

| Lần chạy | Kết quả |
|---|---|
| 100 ms ± 40 ms | 1,8 / 3,5 cm, `stale 0.12 s`: không phân biệt được với khi không có jitter |
| 250 ms ± 40 ms | 3,6 / 7,9 cm, `stale 0.46 s` (0,3 s khi không có jitter): các lượt suy luận chậm thỉnh thoảng đẩy tuổi vượt 0,6 s |
| `--realtime`, 5 s | 42 lượt suy luận với trung bình 105,6 ms, so với 50 lượt đúng 100 ms trong mô phỏng |

**Diễn giải.** Được ủng hộ. Jitter là miễn phí khi $2 \times$ (độ trễ xấu nhất) còn dưới giới
hạn; gần giới hạn, nó lộ ra thành những lần phanh thỉnh thoảng, một cảnh báo sớm hữu ích trên
board thật.

**Tiếp theo (open).** Trên máy tính đích: chạy node driver với node bộ mô phỏng dùng policy thật và
ghi log phân bố tuổi kế hoạch. Đại lượng cần lập ngân sách là phân vị thứ 99, không phải trung bình.

---

## E12 · Bộ phát hiện "lái hoàn hảo"

| | |
|---|---|
| **Giả thuyết** | Cảnh báo của `validate` (chênh lệch trung vị giữa steer đã áp vào và steer định làm dưới 0,02) tách được các bộ dữ liệu cho ra model lái được với các bộ dữ liệu không cho ra. |
| **Lệnh** | vòng lặp của E5; đọc cảnh báo và kết quả đánh giá cùng nhau |
| **Phép đo** | có hay không có cảnh báo, so với số lần lái đạt |

**Measured** (chỉ phần cảnh báo): có cảnh báo ở `--steer-noise 0` (3 episode dài 10 s), không có ở
mặc định 0,25. **Open:** mức nhiễu mà ở đó cảnh báo biến mất, và các model train ngay trên mức đó
có lái được không.

**Dùng vào kỹ thuật.** Nếu ngưỡng của bộ phát hiện và ngưỡng lái được không khớp nhau, phép kiểm
tra rẻ tiền sẽ cho một cảm giác yên tâm sai; khi đó ngưỡng trong
[`validate`](../../drive/learn/validate.py#L62) nên được dời đi.

---

## Các thí nghiệm kỹ thuật nên làm tiếp

Theo thứ tự giá trị với mục tiêu đã nêu của dự án.

| # | Thí nghiệm | Vì sao làm trước |
|---|---|---|
| 1 | **E8 với một SmolVLA đã fine-tune trong bộ mô phỏng** (backend PyTorch, chỉ cần PC) | *đã bắt đầu*: một lần fine-tune 2000 bước lái được và đạt ở 125 ms ([Lab 15](15-hands-on-labs.vi.md#lab-15-fine-tune-smolvla-trên-dữ-liệu-lái-xe-ngay-trên-laptop-này)). Còn mở: lần quét độ trễ của nó, `--action applied` so với `intended`, nhiều bước hơn, nhiều seed hơn |
| 2 | **E6 (c): phương sai theo seed** | mọi phép so sánh model sau này đều phụ thuộc vào việc biết mức nhiễu nền; tốn vài phút |
| 3 | **E10: ranh giới hồi phục**, thêm vào `evaluate` | quy trình chuẩn không phân biệt được một model bền vững với một model mong manh tình cờ xuất phát ở giữa |
| 4 | **E5: quét nhiễu** | xác định cỡ nhiễu cho trường hợp người làm mẫu là một người |
| 5 | **E2: lưới mịn** | biến "khoảng 250 ms ở 1,5 m/s" thành một luật tốc độ supervisor dùng được |
| 6 | **E11 trên Jetson** | phép đo đầu tiên dính tới máy tính thật mà chưa cần xe |
| 7 | **E9 với phanh yếu đi và cọc tiêu lệch tâm** | tìm ra giới hạn thật của supervisor trước khi một chiếc xe thật tìm ra |
| 8 | **Real-time chunking** so với E1/E8 ở 300–650 ms | chỉ khi 1, 5 và 6 cho thấy tích độ trễ × tốc độ quá lớn |

## Mẫu cho thí nghiệm của riêng bạn

```markdown
### E<n> · <tiêu đề>
- Ngày, commit, máy:
- Giả thuyết (có thể bác bỏ được):
- Dự đoán (con số, viết trước khi chạy):
- Biến độc lập và các giá trị:
- Giữ cố định:
- Lệnh:
- Kết quả thô (dán vào):
- Được ủng hộ / bị bác bỏ / chưa kết luận được, và vì sao:
- Nó thay đổi gì trong một quyết định thiết kế:
- Thí nghiệm tiếp theo nó gợi ý:
```
