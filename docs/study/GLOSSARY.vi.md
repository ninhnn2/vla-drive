# Thuật ngữ

*Bản tiếng Việt · [English](GLOSSARY.md)*

*Bản tiếng Việt · [English](GLOSSARY.vi.md)*

Các thuật ngữ theo đúng nghĩa được dùng trong repository này. Mỗi mục có định nghĩa, chỗ ý
tưởng đó nằm trong code, và chương dạy nó. Tên thuật ngữ giữ tiếng Anh như trong code và
tài liệu gốc; nghĩa tiếng Việt ghi trong ngoặc khi cần.

## Hệ thống lái

**Action** (hành động) · Một cặp `[steer, throttle]`, cả hai trong $[-1, 1]$. Steer $+1$ là
đánh lái hết cỡ sang trái; throttle $+1$ là hết ga và $-1$ là phanh hết cỡ; không có số lùi.
[`drive/types.py`](../../drive/types.py) · [Ch. 1](01-system-architecture.vi.md)

**Action chunk** (chunk hành động) · Câu trả lời của một policy: $N$ action tương lai, một
bước thời gian `dt`, và thời điểm của observation đã dùng để tính ra chúng. `actions[i]`
dành cho thời điểm `t_obs + i * dt`. [`ActionChunk`](../../drive/types.py#L41) · [Ch. 5](05-action-chunking-and-latency.vi.md)

**Action chunking** · Cho policy dự đoán một chuỗi ngắn các action tương lai mỗi lần suy
luận thay vì chỉ một action. Cách này chia một lần suy luận chậm ra cho nhiều nhịp điều
khiển, và làm giảm sai số tích luỹ trong học bắt chước. Tên gọi này do ACT đặt ra
(Zhao và cộng sự, 2023). [Ch. 5](05-action-chunking-and-latency.vi.md)

**Chunk executor** (bộ thực thi chunk) · Thành phần mà mỗi nhịp điều khiển trả về action
cho *thời điểm hiện tại* từ chunk mới nhất, chuyển dần (fade) từ chunk trước sang chunk mới.
[`ChunkExecutor`](../../drive/control/executor.py#L11) · [Ch. 5](05-action-chunking-and-latency.vi.md)

**Chunk exhaustion** (cạn chunk) · Đi tới cuối một chunk trước khi chunk mới hơn tới nơi.
`sample` trả về `None` và supervisor phanh. Tránh được khi horizon của chunk ít nhất gấp
đôi độ trễ. [Ch. 5](05-action-chunking-and-latency.vi.md)

**Chunk horizon** · Khoảng thời gian một chunk bao phủ: $(N - 1) \cdot dt$.
[`ActionChunk.horizon_s`](../../drive/types.py#L55)

**Control loop** (vòng điều khiển) · Đoạn code chạy với tần số cố định (ở đây 50 Hz) và
mỗi lần gửi một lệnh tới động cơ. [`DriveLoop.step`](../../drive/loop.py#L32) · [Ch. 2](02-data-flow-and-lifecycle.vi.md)

**Control tick** (nhịp điều khiển) · Một lần lặp của vòng điều khiển: 20 ms.

**Decision** (quyết định) · Output của supervisor: action đi tới động cơ, và các lý do
(nếu có) khiến nó khác với lệnh. [`Decision`](../../drive/safety.py#L19)

**Episode** · Một lần lái được ghi lại: một thư mục chứa `episode.npz`, `meta.json` và,
sau khi gán nhãn, `labels.npz`. [Ch. 9](09-dataset-pipeline.vi.md)

**Inference worker** · Thứ chạy policy tách khỏi vòng điều khiển, mỗi lần một lượt suy
luận. `SimulatedWorker` tạo ra một độ trễ tất định; `ThreadedWorker` dùng một thread thật.
[`drive/control/inference.py`](../../drive/control/inference.py) · [Ch. 2](02-data-flow-and-lifecycle.vi.md)

**Latency** (độ trễ) · Thời gian từ lúc có một observation tới lúc chunk tính từ nó tới nơi.

**Latency compensation** (bù độ trễ) · Đọc một chunk tại vị trí `now - t_obs` thay vì từ
action đầu tiên, để bỏ qua các action dành cho khoảng thời gian đã mất vào việc tính toán,
và mọi action được phát đúng lúc nó được dự định. Nó sửa thời điểm, không sửa độ cũ của
thông tin. [`ChunkExecutor.update`](../../drive/control/executor.py#L28) · [Ch. 5](05-action-chunking-and-latency.vi.md)

**Observation** (quan sát) · Những gì policy thấy tại một thời điểm: ảnh, tốc độ, vị trí
tay lái, một câu mô tả nhiệm vụ và một mốc thời gian (và, chỉ trong bộ mô phỏng, pose).
[`Observation`](../../drive/types.py#L27)

**Observation timestamp** · `obs.t`, thời điểm cảm biến được đọc; được chép vào chunk dưới
tên `t_obs`. Qua ROS 2, đó là stamp trong header của message ảnh, nên thời gian truyền
cũng tính vào độ trễ. [Ch. 2](02-data-flow-and-lifecycle.vi.md)

**Plan** (kế hoạch) · Trong dây chuyền dữ liệu, nhãn của một khung hình: 16 action tương
lai, cách nhau 0,1 s. [`drive/learn/label.py`](../../drive/learn/label.py#L18) · [Ch. 9](09-dataset-pipeline.vi.md)

**Plan age** (tuổi kế hoạch) · `now - t_obs`: thông tin đứng sau action đang được phát đã
cũ bao lâu. Khi policy chạy liên tục lượt này nối lượt kia, nó quét trong khoảng $[L, 2L)$.
[`ChunkExecutor.age_s`](../../drive/control/executor.py#L56) · [Ch. 5](05-action-chunking-and-latency.vi.md)

**Platform** · Những gì vòng điều khiển cần từ một chiếc xe: một đồng hồ, một trạng thái,
`observe`, `range_m` và `apply`. Được hiện thực bởi bộ mô phỏng và bởi adapter ROS 2.
[`Platform`](../../drive/platform.py#L14) · [Ch. 1](01-system-architecture.vi.md)

**Policy** · Bất cứ thứ gì biến một observation thành một action chunk.
[`Policy`](../../drive/policy/base.py#L6) · [Ch. 4](04-driving-policies.vi.md)

**Project** · Một thư mục chứa `episodes/`, `models/` và `reports/`; mặc định là
`runs/drive`. [`Project`](../../drive/learn/project.py#L18)

**Safety supervisor** (bộ giám sát an toàn) · Thành phần cuối cùng trước động cơ. Nó chỉ
thấy lệnh, tốc độ đo được, cảm biến khoảng cách của riêng nó và tuổi của kế hoạch, và có
thể giới hạn lệnh hoặc thay nó bằng lệnh phanh.
[`SafetySupervisor`](../../drive/safety.py#L24) · [Ch. 7](07-safety-supervisor.vi.md)

**Stale plan** (kế hoạch quá cũ) · Một kế hoạch cũ hơn `max_plan_age_s` (0,6 s), hoặc không
có kế hoạch nào. Supervisor phanh, với lý do `stale`.

**Stale-plan watchdog** · Luật của supervisor thực thi điều trên. Ngưỡng của nó phải lớn
hơn hai lần độ trễ của model đang dùng. [Ch. 5](05-action-chunking-and-latency.vi.md)

**Teacher** (giáo viên) · Một policy có thông tin đặc quyền (ở đây là bản đồ và pose chính
xác của xe) dùng để tạo ra các lần lái mẫu: policy pure pursuit. [Ch. 4](04-driving-policies.vi.md)

## Xe và mô phỏng

**Car frame** (hệ toạ độ xe) · Toạ độ tính theo xe: `forward` (phía trước) và `left` (bên
trái), đơn vị mét, tính từ trục sau. [Ch. 3](03-vehicle-simulation.vi.md)

**Curvature** (độ cong) · $\kappa = 1/R$, nghịch đảo của bán kính quay, đơn vị m⁻¹; ở đây
dương nghĩa là quay trái.

**Inverse perspective map** (bản đồ phối cảnh ngược) · Với mỗi pixel của ảnh, điểm trên mặt
đất mà pixel đó nhìn vào. Là hai mảng `forward_m` và `left_m` của camera.
[`Camera`](../../drive/sim/render.py#L48) · [Ch. 3](03-vehicle-simulation.vi.md)

**Kinematic bicycle model** (mô hình xe đạp động học) · Mô hình xe gộp mỗi trục bánh thành
một bánh và giả định lốp không trượt, nên quỹ đạo suy ra chỉ từ hình học:
$\dot\psi = \frac{v}{L}\tan\delta$. Chính xác khi gia tốc ngang nhỏ.
[`step_vehicle`](../../drive/sim/vehicle.py#L10) · [Ch. 3](03-vehicle-simulation.vi.md)

**Lateral offset** (độ lệch ngang) · Khoảng cách có dấu tới đường tâm của đường đua, dương
khi lệch sang trái theo hướng đi. [`Track.project`](../../drive/sim/track.py#L69)

**Look-ahead distance** (khoảng nhìn trước) · Pure pursuit nhắm tới điểm cách bao xa dọc
theo quỹ đạo; ở đây là $\text{clip}(0.4 + 0.35v, 0.5, 1.4)$ m.

**Pose** · Vị trí và hướng, $(x, y, \psi)$. Chỉ biết chính xác trong bộ mô phỏng.

**Pure pursuit** · Một phương pháp bám quỹ đạo: lái theo đường tròn đi qua trục sau, tiếp
tuyến với hướng xe, và đi qua một điểm nằm trên quỹ đạo cách một khoảng nhìn trước.
$\kappa = 2y/d^2$.
[`pursuit_steer`](../../drive/policy/pure_pursuit.py#L19) · [Ch. 4](04-driving-policies.vi.md)

**Range sensor** (cảm biến khoảng cách) · Cảm biến đo khoảng cách phía trước, đầu vào riêng
của supervisor. [`World.range_m`](../../drive/sim/world.py#L53)

**Servo lag** (độ trễ servo) · Tay lái không tới ngay vị trí được ra lệnh. Mô hình hoá thành
một khâu trễ bậc nhất với hằng số thời gian 0,08 s; vị trí thực của bánh xe là một phần của
trạng thái. [Ch. 3](03-vehicle-simulation.vi.md), [Ch. 6](06-control-theory.vi.md)

**Wheelbase** (chiều dài cơ sở) · Khoảng cách giữa trục trước và trục sau, $L = 0.26$ m.

**Yaw** (góc hướng) · Góc hướng $\psi$, đơn vị radian, ngược chiều kim đồng hồ tính từ trục
$+x$ của thế giới.

## Điều khiển

**Closed-loop control** (điều khiển vòng kín) · Điều khiển có đo kết quả và sửa phần chênh
lệch so với mong muốn. Ở đây vòng được đóng ở tần số suy luận (mỗi observation cho một kế
hoạch mới) và, với supervisor, ở 50 Hz. [Ch. 6](06-control-theory.vi.md)

**Open-loop control** (điều khiển vòng hở) · Áp các đầu vào đã quyết định trước mà không đo
kết quả. Phát một chunk là vòng hở cho tới khi chunk tiếp theo tới.

**Dead time (delay)** (thời gian chết, độ trễ) · Khoảng trễ giữa một lần đo và hành động dựa
trên lần đo đó. Nó thêm độ trễ pha $\omega\tau$ và giới hạn mức "mạnh tay" của một vòng
phản hồi.

**Delay margin** (dự trữ độ trễ) · Độ trễ lớn nhất một vòng phản hồi chịu được trước khi mất
ổn định: dự trữ pha chia cho tần số cắt. [Ch. 6](06-control-theory.vi.md)

**Feed-forward** (truyền thẳng) · Một thành phần điều khiển tính từ mục tiêu và mô hình của
đối tượng thay vì từ sai số; chính là `0.1 * speed` trong luật ga.

**Overshoot** (vọt lố) · Vượt quá mục tiêu trước khi ổn định.

**RMS tracking error** (sai số bám RMS) · $\sqrt{\frac1n\sum e_k^2}$ của độ lệch ngang trong
một lần lái; `lateral_rms_m`. [`Metrics.summary`](../../drive/episode.py#L60) · [Ch. 11](11-model-evaluation.vi.md)

**Stability** (tính ổn định) · Một nhiễu tắt dần thay vì lớn lên.

**Zero-order hold** (giữ bậc không) · Giữ lệnh không đổi giữa hai lần lấy mẫu; đó là việc
`apply(action, dt)` làm.

## Học máy

**Behavior cloning (BC)** · Học bắt chước bằng học có giám sát: fit một model để ánh xạ
observation của chuyên gia sang action của chuyên gia. [Ch. 8](08-imitation-learning.vi.md)

**Closed-loop evaluation** (đánh giá vòng kín) · Đánh giá một policy bằng cách để nó tự lái,
nên chính action của nó quyết định nó sẽ thấy gì tiếp theo. Lệnh `evaluate`. [Ch. 11](11-model-evaluation.vi.md)

**Compounding error** (sai số tích luỹ) · Trong các quyết định nối tiếp, một sai số làm đổi
đầu vào tiếp theo, khiến các sai số sau dễ xảy ra hơn; chi phí có thể tăng theo bình phương
của horizon.

**Covariate shift** · Phân phối *đầu vào* của model thay đổi giữa lúc train và lúc dùng,
trong khi output đúng cho mỗi đầu vào vẫn giữ nguyên. Trong lái xe: model đi vào những
trạng thái mà chuyên gia chưa bao giờ đi qua. [Ch. 8](08-imitation-learning.vi.md)

**DAgger** · Dataset Aggregation: chạy model đang học, nhờ chuyên gia gán nhãn cho các
trạng thái nó đi qua, thêm vào dữ liệu, train lại, lặp lại. Chưa hiện thực ở đây.

**DART** · Disturbances for Augmenting Robot Trajectories: chèn nhiễu vào action mà chuyên
gia *thực sự thực hiện* trong lúc ghi, và giữ action chuyên gia *định làm* làm nhãn. Đó là
việc `collect` làm. [`collect`](../../drive/learn/collect.py#L120) · [Ch. 8](08-imitation-learning.vi.md)

**Data augmentation** (tăng cường dữ liệu) · Biến đổi các mẫu đã ghi để mở rộng bộ dữ liệu:
ở đây là lật gương (đảo dấu steer) và đổi độ sáng.
[`augment`](../../drive/learn/train.py#L20)

**Distribution shift** (lệch phân phối) · Bất kỳ khác biệt nào giữa dữ liệu model được train
và dữ liệu nó gặp khi dùng.

**Expert demonstration** (lần làm mẫu của chuyên gia) · Một ví dụ được ghi lại về hành vi
mong muốn.

**Imitation learning** (học bắt chước) · Học một policy từ các lần làm mẫu thay vì từ một
hàm thưởng (reward). [Ch. 8](08-imitation-learning.vi.md)

**Offline (open-loop) evaluation** (đánh giá offline, vòng hở) · So output của model với của
chuyên gia trên các khung hình đã ghi. `val_steer_error`. Nó không dự đoán được khả năng lái.

**Ornstein–Uhlenbeck process** (quá trình Ornstein–Uhlenbeck) · Nhiễu ngẫu nhiên có tương
quan theo thời gian và luôn kéo về 0; dùng để làm nhiễu tay lái một cách mượt.
[`Disturbance`](../../drive/learn/collect.py#L31)

**Recovery behaviour** (hành vi hồi phục) · Phải làm gì khi đã lệch khỏi quỹ đạo mong muốn.
Không có trong các lần làm mẫu lái hoàn hảo.

**Smooth L1 loss** · Bậc hai với sai số nhỏ, tuyến tính với sai số lớn; là loss dùng khi
train, với $\beta = 0.05$.

## VLA và triển khai

**Action expert** · Trong các VLA có đầu ra action liên tục, mạng sinh ra action chunk từ
các đặc trưng của backbone, thường bằng cách khử nhiễu lặp nhiều bước.

**Attention sink** · Một token nhận nhiều attention mà không mang thông tin về đầu vào, ví
dụ token đệm (padding) hoặc một token ở góc giống hệt nhau ở mọi khung hình. Transformer
"đỗ" attention ở đó. Đó là lý do attention không phải là mức độ quan trọng.
[Ch. 17](17-debugging-the-model.vi.md#nó-nhìn-vào-đâu-toolssee_vlapy)

**Backend** · Trong jetson-orin-nano-vla, một object chạy suy luận của một model:
`infer(observation)` trả về một kết quả có `.chunk`. `BackendPolicy` bọc một backend.
[`drive/policy/vla.py`](../../drive/policy/vla.py) · [Ch. 13](13-vla-and-smolvla.vi.md)

**Fine-tuning** · Train tiếp một model đã pre-train trên dữ liệu của nhiệm vụ đích.

**Flow matching** · Một cách train model sinh, biến nhiễu thành dữ liệu bằng cách tích phân
một trường vận tốc học được qua vài bước; là đầu ra action của SmolVLA.

**Image token** (token ảnh) · Một vector mà bộ mã hoá thị giác tạo ra từ một vùng của ảnh.
Trong SmolVLA, một đầu vào 512×512 thành 8×8 = 64 token. Với camera này, 16 token trong số
đó là padding. [Ch. 17](17-debugging-the-model.vi.md#ảnh-trở-thành-token-thế-nào)

**Jetson Orin Nano** · Máy tính nhúng nhỏ của NVIDIA có GPU tích hợp và 8 GB bộ nhớ dùng
chung giữa CPU và GPU; là máy tính dự định đặt trên xe. [Ch. 14](14-jetson-deployment.vi.md)

**LeRobot** · Thư viện robot mã nguồn mở của Hugging Face: bộ dữ liệu, policy (gồm cả
SmolVLA), train và công cụ cho robot.

**LeRobotDataset** · Định dạng bộ dữ liệu của LeRobot: bảng Parquet, video cho camera, và
metadata. `./drive.sh export` ghi các episode theo định dạng này.
[`drive/learn/export.py`](../../drive/learn/export.py) · [Ch. 13](13-vla-and-smolvla.vi.md)

**Occlusion sensitivity** (độ nhạy khi che) · Che lần lượt từng vùng của đầu vào và đo output
dịch đi bao nhiêu. Nó cho thấy câu trả lời phụ thuộc vào đâu, điều attention không cho thấy.
[`tools/see_vla.py`](../../tools/see_vla.py) · [Ch. 17](17-debugging-the-model.vi.md#hai-cách-hỏi-ở-đâu)

**ONNX** · Một định dạng file mở cho mạng nơ-ron, dùng để chuyển giao giữa framework train
và engine suy luận.

**Real-time chunking (RTC)** · Sinh mỗi chunk mới có điều kiện theo các action đã cam kết
cho khoảng thời gian lượt suy luận sẽ mất, để các chunk liên tiếp khớp nhau (Black,
Galliker và Levine, 2025). Chưa hiện thực ở đây.

**SmolVLA** · Một VLA mở, gọn (khoảng 450 triệu tham số) của nhóm LeRobot, xây trên backbone
SmolVLM2 với một action expert dùng flow matching, sinh ra chunk 50 action.
[Ch. 13](13-vla-and-smolvla.vi.md)

**Temporal ensembling** · Lấy trung bình các dự đoán chồng lên nhau của các chunk liên tiếp
(ACT). Chưa hiện thực ở đây; cross-fade 0,1 s là một họ hàng đơn giản hơn.

**TensorRT** · Bộ tối ưu và runtime suy luận của NVIDIA. Nó biên dịch một mạng thành một
engine chuyên cho một GPU, một độ chính xác số và một kích thước đầu vào.
[Ch. 14](14-jetson-deployment.vi.md)

**VLA (vision-language-action model)** · Model nhận ảnh camera, một nhiệm vụ bằng ngôn ngữ
tự nhiên và trạng thái của robot, rồi xuất ra action của robot; thường là một model thị
giác–ngôn ngữ đã pre-train được thêm đầu ra action. [Ch. 13](13-vla-and-smolvla.vi.md)

**VLM (vision-language model)** · Model nhận ảnh và văn bản, sinh ra văn bản.

## ROS 2

**ROS 2** · Robot Operating System 2: middleware và công cụ để xây một robot thành các tiến
trình giao tiếp với nhau. Repository dùng bản phân phối Humble.
[Ch. 12](12-ros2-integration.vi.md)

**Node** · Một thành viên có tên trong đồ thị ROS, làm một việc.

**Topic** · Một kênh có tên và có kiểu, trên đó các node publish và subscribe.

**Message** · Dữ liệu có kiểu gửi trên một topic, ví dụ `sensor_msgs/Image`.

**Publisher / subscriber** · Các handle của một node để gửi tới và nhận từ một topic.

**Parameter** (tham số) · Một giá trị cấu hình có tên của một node.

**Launch file** · Một script khởi động một nhóm node cùng tham số của chúng.

**QoS (quality of service)** · Thiết lập truyền nhận của một publisher hoặc subscription; ở
đây chỉ có độ sâu hàng đợi bằng 1.

**Command watchdog** · Phía xe: phanh nếu 0,5 s không nhận được lệnh nào.
[`sim_node.py`](../../ros/src/drive_ros/drive_ros/sim_node.py#L20) · [Ch. 7](07-safety-supervisor.vi.md), [Ch. 12](12-ros2-integration.vi.md)

**`RosPlatform`** · Adapter trình bày các topic `/drive/*` thành một `Platform`.
[`platform.py`](../../ros/src/drive_ros/drive_ros/platform.py#L13)
