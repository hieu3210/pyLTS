# NEW — Project scaffold for implementing forecast methods

Mục tiêu
- Tạo khuôn khổ (scaffold) để triển khai và so sánh các phương pháp dự báo được mô tả trong tài liệu PDF đính kèm.

Trạng thái hiện tại
- Thư mục chứa file mẫu: `src/` (loader, splitter, placeholder models) và `examples/`.

Tiếp theo cần làm
1. Cung cấp tệp PDF chứa mô tả phương pháp để tôi nghiên cứu chi tiết.
2. Tôi sẽ triển khai các mô-đun tính toán theo phương pháp trong PDF (không thay đổi nguyên lý hiện có trong pyLTS trừ khi bạn yêu cầu).

Files
- `src/data_loader.py` — helper để đọc dataset (định dạng `.txt` giống folder `LTS/Datasets`).
- `src/split.py` — rolling-origin / expanding-window splitter tiện lợi.
- `src/models.py` — placeholder wrappers cho các thuật toán (sẽ triển khai sau khi có PDF).
- `examples/run_example.py` — ví dụ cách nạp dữ liệu và tạo các fold bằng rolling-origin.
- `requirements.txt` — gợi ý các thư viện cần thiết.

How to proceed
- Upload the PDF (or confirm the list of algorithms) and tôi sẽ implement các phương pháp tương ứng trong `NEW/src/models.py`, cùng với tests và notebook minh họa.
