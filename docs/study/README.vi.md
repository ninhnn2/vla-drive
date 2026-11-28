# Giáo trình: hệ thống lái vla-drive

*Bản tiếng Việt · [English](README.md)*

*Bản tiếng Việt · [English](README.vi.md)*

Một giáo trình dựng từ chính code của repository này. Mỗi chương dạy một ý (lý thuyết
trước, rồi tới những dòng code hiện thực nó), và kết thúc bằng một thí nghiệm bạn chạy
được cùng những câu hỏi bạn phải trả lời được.

Giáo trình bổ sung cho, chứ không thay thế, [`README.vi.md`](../../README.vi.md) và
[`docs/driving.vi.md`](../driving.vi.md): hai tài liệu đó nói dự án là gì và chạy nó thế nào;
giáo trình này nói vì sao nó được xây như vậy.

## Các chương

| # | Chương | Học xong bạn sẽ |
|---|---|---|
| 00 | [Lộ trình học](00-learning-roadmap.vi.md) | chọn được đường đi qua tài liệu hợp với nền tảng của mình |
| 01 | [Kiến trúc hệ thống](01-system-architecture.vi.md) | gọi tên mọi module và nói được module đó được làm gì, không được làm gì |
| 02 | [Luồng dữ liệu và vòng đời thực thi](02-data-flow-and-lifecycle.vi.md) | lần theo một observation từ camera tới động cơ |
| 03 | [Mô phỏng xe](03-vehicle-simulation.vi.md) | tự tay bước mô hình xe và nói được nó khác xe thật ở đâu |
| 04 | [Các policy lái xe](04-driving-policies.vi.md) | giải thích bốn policy và chọn được giữa chúng |
| 05 | [Action chunking và bù độ trễ](05-action-chunking-and-latency.vi.md) | tính được action nào của một chunk được phát ở mỗi thời điểm, và vì sao vòng điều khiển hỏng ở khoảng 250 ms |
| 06 | [Lý thuyết điều khiển](06-control-theory.vi.md) | đọc hệ thống như một tập các vòng phản hồi có độ trễ |
| 07 | [Bộ giám sát an toàn](07-safety-supervisor.vi.md) | đoán trước quyết định của supervisor với mọi đầu vào |
| 08 | [Nền tảng học bắt chước](08-imitation-learning.vi.md) | giải thích covariate shift và vì sao tay lái bị cố tình làm nhiễu khi ghi dữ liệu |
| 09 | [Dây chuyền dữ liệu](09-dataset-pipeline.vi.md) | đọc một episode từ đĩa và chạy collect → evaluate |
| 10 | [Train model](10-model-training.vi.md) | suy ra kích thước các tensor và số tham số của mạng |
| 11 | [Đánh giá model](11-model-evaluation.vi.md) | đọc một báo cáo đánh giá và nói được nó không cho thấy điều gì |
| 12 | [Tích hợp ROS 2](12-ros2-integration.vi.md) | vẽ được đồ thị các node và thay bộ mô phỏng bằng một chiếc xe khác |
| 13 | [VLA và SmolVLA](13-vla-and-smolvla.vi.md) | tách được cái đã xây với cái còn trong kế hoạch trên đường tới một VLA |
| 14 | [Triển khai lên Jetson Orin Nano](14-jetson-deployment.vi.md) | liệt kê những gì còn thiếu trước khi hệ thống này lái một chiếc xe thật |
| 15 | [Lab thực hành](15-hands-on-labs.vi.md) | chạy 14 bài thực hành |
| 16 | [Thí nghiệm và phân tích kỹ thuật](16-experiments.vi.md) | tự thiết kế và diễn giải thí nghiệm của mình |
| 17 | [Debug model và thuật toán](17-debugging-the-model.vi.md) | khoanh vùng một lỗi: dữ liệu, quá trình train, model khi chạy offline (nó xuất ra gì, nó nhìn vào đâu trong ảnh), hay vòng điều khiển |
| | [Thuật ngữ](GLOSSARY.vi.md) · [Tài liệu tham khảo](REFERENCES.vi.md) · [Theo dõi tiến độ học](STUDY-TRACKER.vi.md) | |
| | [Thực hành (tiếng Việt)](../thuc-hanh/README.md): 8 bài | thu thập, đọc và đánh giá dữ liệu; gọi tên mọi layer; train và đọc đường cong; phân tích output của model offline và trong vòng kín bằng `./drive.sh probe`; chạy 12 thí nghiệm tối ưu; debug; learning-rate range test, gradient và tỷ lệ cập nhật, ablation đầu vào, sai số on-policy và DAgger; chạy VLA trên Mac |
| | [Sơ đồ kiến trúc](diagrams/vla-drive-architecture.drawio) (draw.io, 5 trang) | nhìn một lượt dây chuyền dữ liệu, vòng chạy thời gian thực và model SmolVLA; tìm đường trong mã nguồn; lên kế hoạch cải thiện model |

## Mỗi nhận định được gắn nhãn thế nào

Dự án mô tả một đích đến (một VLA đã fine-tune, chạy trên Jetson, lái một chiếc xe) mà
nó chưa tới. Để hai thứ đó không bao giờ bị lẫn, các nhận định mang một trong các nhãn sau:

| Nhãn | Nghĩa |
|---|---|
| **[TESTED]** | Đã hiện thực, và có một test trong [`tests/`](../../tests) chạy qua nó |
| **[UNTESTED]** | Đã hiện thực, nhưng chưa có test chạy qua, hoặc chỉ test với một bản thay thế |
| **[SIM-ONLY]** | Đúng với bộ mô phỏng; không có gì cho thấy nó đúng trên xe thật |
| **[PROPOSED]** | Được nêu là bước tiếp theo trong tài liệu của dự án; chưa có code |
| **[FUTURE]** | Một hướng do giáo trình này hoặc tài liệu nghiên cứu gợi ý; repo chưa có kế hoạch |

## Quy ước

- **Liên kết tới mã nguồn** tính từ thư mục này, có anchor tới dòng bắt đầu class hoặc
  hàm: [`DriveLoop.step`](../../drive/loop.py#L32). Số dòng lấy từ working tree ngày
  2026-11-16 và sẽ lệch dần khi code đổi; tên class và tên hàm mới là mốc ổn định.
- **Số liệu.** Một con số ghi *measured* (đo) được tạo cho giáo trình này bằng cách chạy
  đúng lệnh đã nêu vào ngày 2026-11-16 (Python trên máy host, numpy 2.2.6, OpenCV 4.11).
  Một con số ghi *repo* được trích từ [`docs/driving.vi.md`](../driving.vi.md) hoặc từ các báo
  cáo trong `runs/`. Ngoài ra không có gì là số đo. Các lần chạy pure pursuit trong bộ mô
  phỏng là tất định, nên bạn phải lặp lại được chính xác; các lần chạy có render camera
  (`lane-camera`, `cnn`) có thể hơi khác nhau giữa các phiên bản OpenCV.
- **Đơn vị** theo hệ SI. Một action là `[steer, throttle]`, cả hai trong $[-1, 1]$: steer
  $+1$ là đánh lái hết cỡ sang trái, throttle $+1$ là hết ga, $-1$ là phanh hết cỡ. Không
  có số lùi.
- **Lệnh.** `make …` chạy trong container Docker của dự án; trên Mac, nó chạy `drive/`
  trực tiếp trong `.venv` (`make setup`) và chỉ ROS 2 chạy trong container.
  `./drive.sh …` cũng chạy thẳng trên một PC có numpy và OpenCV; chỉ `train` và policy
  `cnn` cần PyTorch, thứ mà container và `.venv` đều có.
- **Sơ đồ** viết bằng Mermaid và **phương trình** bằng LaTeX; cả hai hiển thị được trên
  GitHub và trong Markdown preview của VS Code (cần extension Mermaid).

## Trạng thái working tree mà giáo trình này mô tả

Nhánh `smolvla-sim`: commit đầu tiên (`34e83ed`) cộng thêm `collect --show/--video`,
`browse --match`, bước `export`, policy `vla` (`LeRobotBackend`), các checkpoint
`vla/<version>` của `Project`, bảng debug của HUD và `sim --log`.
82 test đều qua trong container (`make test`); trên Python của máy host, 5 trong số đó bị
bỏ qua (PyTorch, ROS 2, LeRobot).
