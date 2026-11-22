# Thực hành: từ dữ liệu đến một model lái xe tốt

Bộ bài này dạy bằng cách **làm**: bạn chạy lệnh, đọc output, rồi hiểu vì sao output
như vậy. Mọi thứ diễn ra trên chính dự án này, gồm một chiếc xe 1:10 mô phỏng, một
"giáo viên" (teacher) lái giỏi và một mạng CNN 0,3 triệu tham số học lái theo giáo viên.

Mỗi lệnh trong tài liệu đều đã được chạy thật, và output in ra là output thật. Bạn
copy cả khối lệnh, dán vào terminal rồi so kết quả của mình với tài liệu.

## Học xong bạn làm được gì

- Thu thập dữ liệu lái xe, mở từng file ra xem, và **biết một bộ dữ liệu tốt hay xấu**
  dựa trên con số cụ thể.
- Gọi tên **từng layer** của model, tự tính shape và số tham số bằng tay, và biết mỗi
  layer "nhìn thấy" bao nhiêu phần của ảnh.
- Train model, đọc log, và nhận ra model **đang học tốt, học thuộc (overfitting) hay
  không học được gì**.
- **Phân tích output** của model theo hai cách: offline (so với nhãn, so với baseline,
  theo từng tình huống) và khi lái thật trong mô phỏng. Từ đó biết model tốt hay xấu
  và xấu ở chỗ nào.
- Làm thí nghiệm **tối ưu model** đúng cách: mỗi lần chỉ đổi một biến, và biết khi nào
  một chênh lệch là thật, khi nào chỉ là nhiễu.
- **Debug** một model lái tồi: khoanh vùng lỗi nằm ở dữ liệu, ở quá trình train, ở
  model hay ở vòng điều khiển.

## Các bài

| Bài | Nội dung | Thời gian |
|---|---|---|
| [1 · Thu thập và hiểu dữ liệu](01-thu-thap-du-lieu.md) | teacher, nhiễu DART, cấu trúc một episode, `validate`, `browse`, dữ liệu tốt/xấu | 45 phút |
| [2 · Giải phẫu model](02-giai-phau-model.md) | từng layer, shape, tham số, vùng nhìn (receptive field), chạy model trên một frame | 40 phút |
| [3 · Train và đọc log](03-train-va-doc-log.md) | loss, steer error, đường cong train/validation, overfitting, augmentation | 40 phút |
| [4 · Phân tích output của model](04-phan-tich-output.md) | sai số theo bước, so với baseline, frame tệ nhất, activation, saliency, kiểm tra hành vi, lái thật, log CSV | 60 phút |
| [5 · Tối ưu model bằng thí nghiệm](05-toi-uu-model.md) | 12 thí nghiệm thật với kết quả: dữ liệu, nhiễu, nhãn, tốc độ mục tiêu, learning rate, epoch, augmentation, seed | 60 phút |
| [6 · Debug model](06-debug-model.md) | khoanh vùng lỗi, bảng triệu chứng → nguyên nhân → lệnh, các bug thật đã gặp trong dự án | 40 phút |
| [7 · Debug và tối ưu như dân chuyên nghiệp](07-debug-chuyen-nghiep.md) | bản đồ các phương pháp; LR range test; gradient, update/weight, unit chết; ablation; sai số on-policy; DAgger; báo cáo HTML | 60 phút |
| [8 · Chạy và debug VLA (SmolVLA) trên MacBook](08-vla-tren-mac.md) | môi trường LeRobot; fine-tune trên GPU Mac; `probe_vla`, `see_vla`, `vla_app`; so sánh MPS với CPU; lỗi riêng của Mac | 45 phút, cộng khoảng 2 giờ train |
| [9 · Thu dữ liệu cho SmolVLA trong CARLA](09-thu-du-lieu-carla.md) | server CARLA trong Docker; mô hình xe và độ chính xác nhãn; câu lệnh theo từng frame; `--show`, `--resume`; export, train; kiểm model offline (`probe_vla_carla`) và lái thật (`carla drive`); tìm lỗi server bằng cách đổi từng biến; hai điểm yếu của dữ liệu | 90 phút, cộng khoảng 1 giờ thu và 1 giờ train |

Nên học theo thứ tự. Bài 4 và 5 dùng model `v1` train ở bài 3.

## Chuẩn bị (làm một lần)

Mọi lệnh đều chạy từ **thư mục gốc của repo**.

**MacBook (Apple Silicon).** Dự án chạy trực tiếp trên máy, trong một môi trường ảo
`.venv`, và train trên GPU của Mac (MPS). Cần có [uv](https://docs.astral.sh/uv/)
(`brew install uv`).

```bash
make setup                       # tạo .venv: Python 3.10, numpy, OpenCV, PyTorch 2.5.1
source .venv/bin/activate        # từ giờ, trong terminal này, python3 là Python của dự án
```

**Linux có GPU NVIDIA.** Dự án chạy trong container Docker.

```bash
make build                       # một lần, khoảng 9 GB
make shell                       # mở shell trong container; mọi lệnh bên dưới gõ ở đây
```

**Kiểm tra môi trường:**

```bash
./drive.sh --help | head -3
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"
```

Output trên máy dùng để viết tài liệu này:

```
usage: drive.sh [-h]
                {sim,sweep,collect,label,validate,browse,lrfind,train,evaluate,probe,status,export}
                ...
torch 2.5.1 | cuda False | mps True
```

`./drive.sh` là lệnh chính của dự án. Trên Mac, nó tự dùng `.venv`; trong container,
nó dùng Python của container.

## Quy ước trong tài liệu

- **Khối lệnh** không có dấu `$` ở đầu, nên bạn copy nguyên khối là chạy được. Ngay
  sau đó là khối **output thật**.
- **Máy đo.** MacBook Apple M3 (CPU 8 nhân, GPU 10 nhân, RAM 16 GB), macOS 14.7,
  Python 3.10, PyTorch 2.5.1, train trên MPS. Ngày đo: 2026-11-17.
- **Có lặp lại được không?**
  - `collect` và teacher `pure-pursuit` cho kết quả **giống hệt** trên mọi máy.
  - `train` cho kết quả lệch nhau ở vài chữ số cuối giữa MPS, CUDA và CPU, nên model
    của bạn sẽ hơi khác một chút. Những gì quan trọng (model lái được hay không, xu
    hướng giữa các thí nghiệm) thì phải giống.
  - Thời gian chạy phụ thuộc vào máy. Trên GPU RTX 4060, train nhanh hơn khoảng 6 lần.
- **Thư mục thí nghiệm.** Mọi bài đều dùng `--project runs/hoc` (và `runs/hoc-*` cho các
  thí nghiệm ở bài 3 và 5), không đụng tới `runs/drive` mặc định. Thư mục `runs/` không nằm trong git, nên
  muốn làm lại từ đầu thì cứ xóa nó đi.
- **Thuật ngữ.** Tài liệu giữ nguyên thuật ngữ tiếng Anh, vì đó là chữ bạn sẽ gặp trong
  code, log và sách báo, và giải thích nghĩa ở lần đầu xuất hiện. Danh sách đầy đủ ở
  [GLOSSARY](../study/GLOSSARY.md) (tiếng Anh).
- **Bài tập** có đáp án ẩn trong `<details>`. Hãy tự trả lời trước khi mở xem.

## Bộ tài liệu này nằm ở đâu so với các tài liệu khác

- [`README.md`](../../README.md) và [`docs/driving.md`](../driving.md): dự án là gì và
  chạy thế nào.
- [`docs/study/`](../study/README.md): giáo trình tiếng Anh 18 chương, đi sâu vào lý
  thuyết (điều khiển, độ trễ, an toàn, ROS 2, VLA).
- **Bộ này** chỉ tập trung vào dữ liệu và model, theo lối thực hành: lệnh nào, output
  gì, đọc ra sao, tốt hay xấu. Khi cần lý thuyết sâu hơn, mỗi bài đều có link sang
  chương tương ứng của giáo trình.

## Công cụ chính

| Lệnh | Dùng để |
|---|---|
| `./drive.sh collect` | thu dữ liệu: teacher lái 40 lượt, có nhiễu; hoặc model lái và teacher gán nhãn (DAgger, bài 7) |
| `./drive.sh validate` | kiểm tra dữ liệu trước khi train |
| `./drive.sh browse` | xem từng frame, có vẽ nhãn lên đường |
| `./drive.sh train` | train CNN |
| `./drive.sh probe` | **soi model**: layers, history, training, errors, ablation, activations, saliency, behaviour, onpolicy, report |
| `./drive.sh lrfind` | thử nhanh các learning rate trước khi train lâu (bài 7) |
| `./drive.sh evaluate` | cho model lái thật trên mọi track, cả hai chiều: bài kiểm tra cuối cùng |
| `./drive.sh sim --log` | ghi từng tick (50 lần/giây) của một lượt lái ra file CSV để phân tích |
| `./drive.sh status` | dự án đang có gì, nên làm gì tiếp |
