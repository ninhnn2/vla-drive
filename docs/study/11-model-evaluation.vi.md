# 11 · Đánh giá model

*Bản tiếng Việt · [English](11-model-evaluation.md)*

*Bản tiếng Việt · [English](11-model-evaluation.vi.md)*

Mọi thứ ở đây đều **[SIM-ONLY]**: các metric cần dữ liệu thật tuyệt đối mà chỉ bộ mô phỏng
có. **[TESTED]** bởi `EvaluateTests` và `DriveTests`.

## Ta đang giải bài toán gì?

Quyết định một policy có tốt hay không. [Chương 8](08-imitation-learning.vi.md) đã cho thấy thước
đo hiển nhiên, sai số trên các khung hình giữ lại, có thể xếp một model không lái được lên trên
một model lái được. Thước đo có giá trị là **vòng kín**: đặt policy vào vòng lặp, để chính các
action của nó quyết định nó thấy gì tiếp theo, và xem chuyện gì xảy ra.

| | Offline (vòng hở) | Vòng kín |
|---|---|---|
| Câu hỏi | Model có đồng ý với chuyên gia trên các trạng thái của chuyên gia không? | Model có ở lại trong những trạng thái nó xử lý được không? |
| Dữ liệu | các khung hình đã ghi | một lần lái thật sự |
| Sai số | độc lập | tích luỹ |
| Chi phí | vài mili giây | một lần lái mô phỏng 40 s cho mỗi điều kiện |
| Trong repository này | `val_steer_error` trong `models/<v>.json` | `evaluate` → `reports/<v>.json` |

## Các metric

Được [`Metrics`](../../drive/episode.py#L37) tính ở mỗi nhịp điều khiển từ trạng thái thật
của bộ mô phỏng, và tóm tắt bởi [`Metrics.summary`](../../drive/episode.py#L60). Gọi $e_k$ là
độ lệch ngang so với đường tâm tại nhịp $k$ ([`Track.project`](../../drive/sim/track.py#L69)),
$n$ là số nhịp.

| Khoá | Định nghĩa | Nghĩa |
|---|---|---|
| `lateral_rms_m` | $\sqrt{\frac1n\sum_k e_k^2}$ | khoảng cách điển hình tới đường tâm |
| `lateral_max_m` | $\max_k \lvert e_k \rvert$ | thời điểm tệ nhất |
| `off_track_pct` | $100 \cdot \frac1n \cdot$ (số nhịp có $\lvert e_k \rvert > 0.45$ m) | tỷ lệ thời gian ở ngoài mặt đường 0,9 m |
| `mean_speed_mps` | $\frac1n\sum_k v_k$ | lái nhanh nhẹn tới đâu |
| `laps` | quãng tiến dọc đường đua ÷ chiều dài đường đua | quãng đường đã đi |
| `collided` | true nếu thân xe từng chồng lên một vật cản | |
| `min_range_m` | số đo khoảng cách nhỏ nhất | đã tới gần tới mức nào |
| `safety_s` | số giây mỗi luật của supervisor hoạt động | policy phải bị gạt đi bao nhiêu lần |
| `inferences`, `latency_ms_mean` | từ worker | policy tốn bao nhiêu |

**Quãng tiến** được cộng dồn từ mức thay đổi độ dài cung (có quay vòng) ở mỗi nhịp,
$\Delta s = \big((s_k - s_{k-1} + \tfrac{\ell_t}{2}) \bmod \ell_t\big) - \tfrac{\ell_t}{2}$
với $\ell_t$ là chiều dài đường đua, để việc đi qua vạch xuất phát không bị tính là một cú nhảy
lùi, và lái ngược chiều sẽ bị tính âm.

**Vì sao dùng cả RMS và max?** RMS cho biết về cả lần lái và che giấu một khúc cua tồi duy
nhất; max cho thấy khúc cua đó và nhạy với một lần trục trặc. Một policy có RMS thấp và max cao
có một chỗ cụ thể mà nó vật lộn.

## Đạt hay trượt

[`evaluate`](../../drive/learn/evaluate.py#L22), một dòng:

```python
ok = r["off_track_pct"] == 0 and not r["collided"] and r["mean_speed_mps"] > 0.3
```

Một lần lái đạt khi xe **không bao giờ ra khỏi đường, không đâm vào gì, và vẫn tiếp tục chạy**.

Điều kiện thứ ba quan trọng. Một chiếc xe đứng yên không bao giờ ra khỏi đường. Không có tốc độ
tối thiểu, policy an toàn nhất sẽ là policy tốt nhất.
(`EvaluateTests.test_a_drive_that_hits_something_fails` phủ điều kiện thứ hai.)

## Quy trình đánh giá

Mỗi model được cho lái trên **mọi đường đua, chiều xuôi và chiều ngược**: mặc định sáu lần lái
40 s ở độ trễ 100 ms.

```bash
make evaluate V="v1 v2 pure-pursuit"
```

- **Đường đua** `peanut`, `oval`, `trefoil` ([chương 3](03-vehicle-simulation.vi.md)). Các model
  chỉ train trên peanut chưa từng thấy hai đường còn lại.
- **Ngược chiều** đổi chỗ cua trái và cua phải: một phép kiểm tra thiên lệch theo hướng.
- **Độ trễ** do `SimulatedWorker` áp, nên phép so sánh không bị ảnh hưởng bởi PC nhanh hay chậm.
- **Tất định**: cùng một model cho cùng một báo cáo.

Output: một bảng cho mỗi model, `reports/<model>.json`, một bảng so sánh, và exit code 0 chỉ
khi mọi lần lái của mọi model đều đạt, để lệnh có thể làm cửa chặn trong một script.

## Kết quả

*Repo* (`runs/drive/reports/*.json`; lần lái 40 s, độ trễ 100 ms). `v1` được train trên 40
episode mô phỏng, tất cả trên peanut. `v2` trên các episode đó cộng hai bản ghi ROS 2, một trên
oval và một trên trefoil.

| Model | Số lần lái đạt | Tốc độ trung bình | rms lệch đường tâm | Tệ nhất |
|---|---:|---:|---:|---:|
| `v1` | 6/6 | 0.84 m/s | 1.3 cm | 4.0 cm |
| `v2` | 6/6 | 0.90 m/s | 1.2 cm | 3.8 cm |
| `pure-pursuit` (teacher) | 6/6 | 1.47 m/s | 1.5 cm | 3.4 cm |

Theo từng lần lái (rms / max tính bằng cm, tốc độ trung bình tính bằng m/s):

| Đường đua | `v1` | `v2` | `pure-pursuit` |
|---|---|---|---|
| peanut | 1.5 / 4.0 @ 0.84 | 1.1 / 2.7 @ 0.89 | 1.8 / 3.4 @ 1.46 |
| peanut ngược | 1.4 / 3.9 @ 0.83 | 1.3 / 3.5 @ 0.90 | 1.8 / 3.4 @ 1.46 |
| oval | 1.7 / 3.5 @ 0.86 | 0.8 / 2.0 @ 0.89 | 1.0 / 2.4 @ 1.47 |
| oval ngược | 1.3 / 3.5 @ 0.83 | 1.6 / 3.3 @ 0.92 | 1.0 / 2.4 @ 1.47 |
| trefoil | 1.0 / 2.9 @ 0.84 | 1.2 / 3.8 @ 0.91 | 1.7 / 2.8 @ 1.47 |
| trefoil ngược | 0.8 / 2.0 @ 0.82 | 1.0 / 3.2 @ 0.90 | 1.7 / 2.8 @ 1.47 |

Trong mọi lần lái, supervisor chỉ can thiệp lúc khởi động (`stale 0.1 s`, `steer_rate`
0,02–0,06 s).

### Đọc kết quả

**1. `v1` tổng quát hoá qua các đường đua.** Train trên một hình dạng, nó đạt trên ba hình
dạng, theo cả hai chiều, và khả năng bám đường trên các hình dạng chưa thấy (0,8–1,7 cm rms)
chỉ chênh vài milimet so với trên hình dạng đã train (1,4–1,5 cm). Các con số của teacher giống
hệt nhau khi chạy xuôi và ngược, đúng như phải thế với một thuật toán đối xứng; của các mạng thì
không, cho thấy một chút bất đối xứng theo hướng mà lật gương chưa xoá hết.

**2. Các mạng bám đường tâm sát bằng hoặc sát hơn teacher, trong khi lái chậm hơn 40 %.** Hai
điều này là một. Sai số của pure pursuit chủ yếu là cắt góc, thứ tăng theo tốc độ. Xe chậm hơn
thì cắt ít hơn. So rms ở những tốc độ khác nhau không phải so cùng loại: teacher ở tốc độ của
các mạng mới là mốc so sánh công bằng (*measured:* teacher ở 0,6 m/s giữ được 0,7 cm rms).

**3. Vì sao chậm hơn?** `docs/driving.md`: "its target speed changed per episode and is not in
the image, so they settle on a cautious average." Xem [chương 8](08-imitation-learning.vi.md),
mục "thứ ảnh không cho thấy".

**4. `v2` không rõ ràng tốt hơn `v1`.** 1,2 so với 1,3 cm rms và 3,8 so với 4,0 cm tệ nhất, mỗi
model một lần train. `v2` tốt hơn ở vài lần lái và tệ hơn ở những lần khác. Với một seed cho mỗi
model, một khác biệt nhỏ như vậy không phải bằng chứng rằng hai bản ghi thêm vào đã giúp được.
Muốn khẳng định, hãy train mỗi cấu hình với nhiều seed và so độ phân tán
([thí nghiệm E6](16-experiments.vi.md)).

**5. Còn model train trên dữ liệu lái hoàn hảo** (`runs/perfect/reports/v1.json`): 0/6, ra khỏi
đường 62 tới 85 % thời gian, ở 1,7 m/s. Sai số steer validation của nó là tốt nhất trong ba.

| | `v1` | `v2` | chỉ lái hoàn hảo |
|---|---:|---:|---:|
| Sai số steer validation (offline) | 0.0242 | 0.0223 | **0.0149** |
| Số lần lái đạt (vòng kín) | 6/6 | 6/6 | **0/6** |

## Độ trễ, với một policy học được

*Measured* cho giáo trình này: `./drive.sh sim --policy cnn --checkpoint v1 --latency-ms L
--max-plan-age-ms 1500 --duration-s 40` trên peanut, trong container.

| Độ trễ | rms / max | Ra khỏi đường | Tốc độ trung bình |
|---:|---:|---:|---:|
| 0 ms | 1.3 / 2.8 cm | 0 % | 0.84 m/s |
| 100 ms | 1.5 / 4.0 cm | 0 % | 0.84 m/s |
| 200 ms | 2.5 / 5.9 cm | 0 % | 0.84 m/s |
| 300 ms | 22.6 / 54.5 cm | 6.2 % | 0.89 m/s |
| 400 ms | 259.8 / 825.5 cm | 75.2 % | 0.83 m/s |

So với teacher ở tốc độ tương tự ([chương 5](05-action-chunking-and-latency.vi.md)): ở 1,0 m/s nó
vẫn giữ 1,6 cm ở 250 ms, và ở 0,6 m/s nó giữ 1,6 cm ở 400 ms. Mạng hỏng ở 300 ms trong khi chỉ
lái 0,84 m/s. **Một kế hoạch học được chịu độ trễ kém hơn kế hoạch của teacher.** Các lý do khả
dĩ, chưa được tách riêng ở đây: các dự đoán càng về cuối chunk của nó càng kém chính xác hơn một
lần rollout mô hình, và kế hoạch của nó có bước thời gian thô hơn (0,1 s so với 1/30 s). Hệ quả
thực tế: đừng lấy một ngân sách độ trễ đo với teacher rồi mặc định nó cho một policy học được.
Hãy đo chính policy bạn sẽ chạy.

`evaluate` cung cấp điều này qua `--latency-ms`.

> **Cẩn thận.** `evaluate` ghi `reports/<model>.json` mỗi lần chạy. Đánh giá lại `v1` với một
> `--latency-ms` hay `--duration-s` khác sẽ **ghi đè** báo cáo trước đó. Dùng
> `./drive.sh sim --policy cnn --checkpoint v1 …` cho các lần chạy thăm dò, như ở trên, hoặc
> dùng một bản sao của project.

## Những gì việc đánh giá không cho thấy

| Chưa được test | Vì sao quan trọng |
|---|---|
| Vật cản | không lần lái đánh giá nào có vật cản; các mạng chưa bao giờ được cho thấy lý do để dừng |
| Xuất phát lệch tâm, hoặc hồi phục sau một cú đẩy | mọi lần lái đều xuất phát trên đường tâm từ trạng thái đứng yên |
| Lâu hơn 40 s | một sự trôi chậm sẽ không lộ ra |
| Nhiều seed | mỗi model một lần train; không có ước lượng phương sai |
| Bất kỳ vẻ ngoài nào khác | cùng bộ mô phỏng, cùng màu sắc, cùng camera như lúc train |
| Độ trễ dao động, thread thật | `SimulatedWorker`, độ trễ cố định |
| Một chiếc xe thật | mọi thứ trong bảng giới hạn của [chương 3](03-vehicle-simulation.vi.md) |

"6/6" nghĩa là: trên ba hình dạng này, trong bộ mô phỏng này, xuất phát từ đứng yên, trong 40 s,
ở độ trễ cố định 100 ms.

## Những lỗi và hiểu lầm thường gặp

- **"Các mạng thắng teacher về rms."** Ở tốc độ thấp hơn.
- **"6/6 nghĩa là an toàn."** Nó nghĩa là đã thoả ba điều kiện trong sáu lần lái ngắn.
- **"`v2` cải thiện so với `v1`."** Chưa chứng minh được với một seed.
- **"Nếu loss validation đã giảm thì bỏ qua đánh giá."** Hãy nhớ model lái hoàn hảo.
- **"Đạt ở 100 ms nghĩa là sẽ đạt trên Jetson."** Chỉ khi độ trễ trên Jetson là 100 ms và model
  là cùng một model.

## Thí nghiệm thực hành

Thêm thứ quy trình chuẩn còn thiếu: một bài test hồi phục.

```python
# recover.py: xuất phát lệch đường tâm 25 cm, lệch hướng 0,3 rad, và xem policy có quay về không
from drive.episode import EpisodeConfig, Metrics, build
from drive.learn.project import Project

config = EpisodeConfig(policy="cnn", checkpoint=str(Project().checkpoint("v1")), duration_s=10.0)
world, loop = build(config)
world.place(5.0, lateral_m=0.25, yaw_offset_rad=0.3)
metrics = Metrics(world)
while world.t < config.duration_s:
    metrics.add(loop.step())
loop.worker.close()
print(metrics.summary(loop.config.dt))
```

Chạy nó trong container (`make shell`, rồi `python3 recover.py`). Lặp lại với
`Project("runs/perfect")`. Sau đó thay đổi độ lệch cho tới khi `v1` trượt: độ lệch đó mô tả độ
bền vững của nó trung thực hơn "6/6".

## Kiểm tra kiến thức

1. Vì sao tốc độ trung bình tối thiểu là một phần của luật đạt?
2. Một policy có rms 1,0 cm và max 30 cm. Điều đó gợi ý gì?
3. Vì sao so rms của các mạng với rms của teacher là gây hiểu lầm?
4. Bằng chứng nào sẽ cho thấy `v2` thực sự tốt hơn `v1`?
5. Việc đánh giá nói gì về vật cản?

<details><summary>Đáp án</summary>

1. Nếu không, một chiếc xe đứng yên, thứ không bao giờ ra khỏi đường, sẽ đạt.
2. Bám đường tốt gần như ở mọi nơi, và có một chỗ, hoặc một thời điểm, suýt hỏng.
3. Chúng lái ở các tốc độ khác nhau, và sai số của pure pursuit tăng theo tốc độ.
4. Nhiều lần train mỗi cấu hình với seed khác nhau, và một khác biệt lớn hơn độ phân tán giữa các lần chạy.
5. Không nói gì: không lần lái đánh giá nào có vật cản.
</details>

## Đọc thêm

- Codevilla và cộng sự 2019, về khoảng cách giữa metric offline và hiệu năng lái.
- Ross và Bagnell 2010, về vì sao khoảng cách đó tồn tại.
- Các metric của bảng xếp hạng CARLA (route completion, infractions), làm ví dụ về đánh giá vòng kín ở quy mô lớn hơn.
