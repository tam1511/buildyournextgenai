# Hướng dẫn triển khai — Vietnet Air Voice Agent

Tài liệu này hướng dẫn toàn bộ quá trình:
1. Mua và thiết lập VPS Hostinger với template n8n
2. Kết nối tên miền với n8n qua HTTPS (Traefik)
3. Deploy landing page tại domain chính
4. Kết nối Vapi với webhook n8n mới

---

## Phần 1 — Mua VPS Hostinger với template n8n

### 1.1 Chọn gói và thiết lập

1. Truy cập link Hostinger (kèm mã giảm giá 10%) → nhấn **Chọn Gói**.
2. Chọn **KVM 2** (2 vCPU, 8 GB RAM, 100 GB NVMe) — khuyến nghị cho n8n.
3. Thời hạn: chọn **24 tháng** để có giá tốt nhất (~241.900đ/tháng) và tên miền miễn phí.
4. Bật **sao lưu hàng ngày** nếu chạy hệ thống thật cho khách hàng.
5. Chọn vị trí máy chủ phù hợp với khách hàng của bạn.
6. Ở phần **Chọn hệ điều hành**, tìm kiếm `n8n` → chọn template **n8n One-Click**.
   - Template này đã bao gồm Docker + n8n + Traefik — không cần cài thêm gì.
7. Áp mã giảm giá → Tiếp tục → thanh toán.

---

## Phần 2 — Nhận tên miền miễn phí và thiết lập DNS

### 2.1 Nhận tên miền

Sau khi thanh toán, từ trang **Tổng quan**:

1. Nhấn **"Nhận tên miền"**.
2. Nhập tên muốn dùng (ví dụ: `vietnetair.tech`) → kiểm tra khả dụng → xác nhận.
3. Điền thông tin chủ sở hữu → xác nhận.
4. Mở email xác minh từ Hostinger → nhấn **Xác minh Email**.

### 2.2 Lấy địa chỉ IP của VPS

Vào **VPS → Tổng quan** → ghi lại địa chỉ **IP** (ví dụ: `187.127.118.119`).

### 2.3 Cấu hình DNS

1. Hostinger → **Quản lý tên miền** → **DNS / Máy chủ tên miền**.
2. Xoá toàn bộ bản ghi mặc định.
3. Thêm hai bản ghi A mới:

| Type | Name | Points to       | TTL |
|------|------|-----------------|-----|
| A    | @    | `<IP VPS của bạn>` | 300 |
| A    | n8n  | `<IP VPS của bạn>` | 300 |

- Bản ghi `@` → `vietnetair.tech` (landing page)
- Bản ghi `n8n` → `n8n.vietnetair.tech` (giao diện n8n)

4. Nhấn **Save** và đợi 2–5 phút để DNS lan truyền.

---

## Phần 3 — Thiết lập n8n trên VPS

### 3.1 Mở Terminal

Vào **Hostinger → Quản lý VPS** → nhấn nút **Terminal** (terminal ngay trong trình duyệt).

### 3.2 Cập nhật hệ thống

```bash
sudo apt update && sudo apt upgrade -y
```

### 3.3 Kiểm tra Docker containers

```bash
docker ps
```

Bạn sẽ thấy hai container đang chạy:
- **traefik** — xử lý HTTPS và routing tên miền
- **n8n** — ứng dụng automation

### 3.4 Cập nhật domain trong cấu hình n8n

Xem cấu hình hiện tại:

```bash
cat /docker/n8n-hvdv/.env
```

Thay thế subdomain mặc định của Hostinger bằng domain của bạn  
*(thay `srv1728555.hstgr.cloud` bằng subdomain thực tế bạn thấy trong file `.env`)*:

```bash
sed -i 's/TRAEFIK_HOST=srv1728555.hstgr.cloud/TRAEFIK_HOST=vietnetair.tech/' /docker/n8n-hvdv/.env
echo "COMPOSE_PROJECT_NAME=n8n" >> /docker/n8n-hvdv/.env
```

Kiểm tra lại:

```bash
cat /docker/n8n-hvdv/.env
```

Kết quả mong đợi:

```
TZ=Asia/Ho_Chi_Minh
TRAEFIK_HOST=vietnetair.tech
COMPOSE_PROJECT_NAME=n8n
```

### 3.5 Restart n8n và Traefik

```bash
# Restart n8n
cd /docker/n8n-hvdv
docker compose down
docker compose up -d

# Restart Traefik (tự xin chứng chỉ HTTPS cho domain mới)
cd /docker/traefik
docker compose down
docker compose up -d
```

Đợi 2–3 phút để Traefik xin chứng chỉ Let's Encrypt.

### 3.6 Truy cập n8n

Mở trình duyệt → `https://n8n.vietnetair.tech`

Bạn sẽ thấy trang đăng ký n8n với HTTPS hoạt động hoàn toàn tự động.

---

## Phần 4 — Thiết lập n8n và import workflow

### 4.1 Tạo tài khoản n8n

1. Điền email, tên, mật khẩu → **Next**.
2. Trả lời các câu hỏi onboarding → vào dashboard.
3. Nhấn vào popup **Free Licence Key** → mở email → kích hoạt → đăng nhập lại.

### 4.2 Thêm Credentials

Vào **Settings → Credentials → Add first credential**:

- **OpenAI**: dán API key từ [platform.openai.com](https://platform.openai.com)
- **Supabase**: dán API URL và Service Role Key từ Supabase dashboard

### 4.3 Import workflow từ n8n Cloud

1. Vào n8n Cloud → mở workflow **Voice Agent Backend** → **Download** (lưu file `.json`).
2. Vào n8n self-host → **Import** → chọn file vừa tải.
3. Đợi n8n nhận dạng lại credentials.

### 4.4 Publish và lấy Production URL

1. Nhấn **Publish**.
2. Ở node Webhook → copy **Production URL** (dạng `https://n8n.vietnetair.tech/webhook/...`).

### 4.5 Cập nhật URL trong Vapi

Vào **Vapi → Assistants → Tools → `tra_cuu_chuyen_bay`**:
- Tìm ô **Server URL** → xoá URL cũ → dán Production URL mới → **Save**.

---

## Phần 5 — Deploy Landing Page

### 5.1 Tạo thư mục

```bash
mkdir -p /docker/landing/html
```

### 5.2 Tạo docker-compose.yml

```bash
cat > /docker/landing/docker-compose.yml << 'EOF'
services:
  landing:
    image: nginx:alpine
    container_name: landing
    restart: unless-stopped
    volumes:
      - ./html:/usr/share/nginx/html:ro
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.landing.rule=Host(`vietnetair.tech`)"
      - "traefik.http.routers.landing.entrypoints=websecure"
      - "traefik.http.routers.landing.tls.certresolver=letsencrypt"
      - "traefik.http.services.landing.loadbalancer.server.port=80"
EOF
```

### 5.3 Upload file index.html

Copy nội dung file `index.html` từ repo này lên server:

```bash
nano /docker/landing/html/index.html
```

Dán toàn bộ nội dung HTML vào → `Ctrl+O` → `Enter` → `Ctrl+X`.

### 5.4 Điền thông tin Vapi vào file HTML

Tìm và sửa hai dòng này trong `index.html`:

```javascript
const VAPI_PUBLIC_KEY   = "public-key-của-bạn";         // Vapi → Account → Public Key
const VAPI_ASSISTANT_ID = "0ef42f8b-4956-448a-835b-386d748724a6"; // Vapi → Assistants → ID
```

Lấy thông tin từ:
- **Public Key**: Vapi → Account Settings → API Keys
- **Assistant ID**: Vapi → Assistants → chọn assistant → copy ID từ URL hoặc settings

### 5.5 Khởi động nginx container

```bash
cd /docker/landing
docker compose up -d
```

### 5.6 Kiểm tra

```bash
docker ps | grep landing
```

Mở trình duyệt → `https://vietnetair.tech`

Landing page xuất hiện với HTTPS. Nhấn **"Bắt đầu trò chuyện"** để test AI agent ngay trên trang web.

---

## Phần 6 — (Tuỳ chọn) Thêm số điện thoại thật

Nếu muốn khách hàng gọi qua số điện thoại:

1. **Vapi → Phone Numbers → Buy Number** — chọn số Mỹ trực tiếp từ Vapi.
2. Hoặc kết nối số Twilio nếu cần đầu số khác.
3. Sau khi có số: gán vào Assistant → cập nhật số hiển thị trong `index.html`.

> 📎 Hướng dẫn chi tiết: [docs.vapi.ai/phone-numbers](https://docs.vapi.ai/phone-numbers)

---

## Kiểm tra nhanh — Checklist

- [ ] VPS Hostinger đã chạy, terminal truy cập được
- [ ] `docker ps` thấy container `traefik` và `n8n`
- [ ] DNS đã cập nhật: `n8n.vietnetair.tech` và `vietnetair.tech` trỏ về IP VPS
- [ ] `https://n8n.vietnetair.tech` mở được với HTTPS
- [ ] Credentials OpenAI và Supabase đã thêm vào n8n
- [ ] Workflow Voice Agent Backend đã import và publish
- [ ] Production URL đã cập nhật trong Vapi Tool
- [ ] `https://vietnetair.tech` hiển thị landing page
- [ ] Nút "Bắt đầu trò chuyện" kết nối được với AI agent

---

## Cấu trúc thư mục trên server

```
/docker/
├── traefik/            ← HTTPS reverse proxy (Hostinger tạo sẵn)
├── n8n-hvdv/           ← n8n automation (Hostinger tạo sẵn)
│   └── .env            ← TRAEFIK_HOST, TZ, COMPOSE_PROJECT_NAME
└── landing/            ← Landing page (chúng ta tạo)
    ├── docker-compose.yml
    └── html/
        └── index.html
```

---

*Được xây dựng trong series **AI Agent với n8n + Vapi + Supabase** — tự host, toàn quyền kiểm soát, chi phí tối ưu.*
