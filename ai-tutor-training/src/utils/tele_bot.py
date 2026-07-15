import time
import threading
import requests
import traceback
import os
from dotenv import load_dotenv

# load_dotenv()

# TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    # """Hàm gửi tin nhắn qua Telegram"""
    # url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # payload = {
    #     "chat_id": TELEGRAM_CHAT_ID,
    #     "text": message,
    #     "parse_mode": "HTML" # Cho phép format chữ đậm, nghiêng
    # }
    # try:
    #     requests.post(url, json=payload, timeout=5)
    # except Exception as e:
    #     print(f"Không thể gửi Telegram alert: {e}")

    pass

class TrainingWatchdog:
    def __init__(self, accelerator, timeout_seconds=900):
        """
        timeout_seconds: Thời gian tối đa cho 1 step (mặc định 900s = 15 phút).
        """
        self.accelerator = accelerator
        self.timeout_seconds = timeout_seconds
        self.last_step_time = time.time()
        self.is_running = True
        self.alert_sent = False
        
        # Tạo luồng chạy ngầm
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def start(self):
        if self.accelerator.is_main_process:
            send_telegram_alert("🚀 <b>Bắt đầu quá trình Training!</b>")
            self.thread.start()

    def update(self):
        """Gọi hàm này ở cuối MỖI STEP training để reset đồng hồ đếm giờ"""
        if self.accelerator.is_main_process:
            self.last_step_time = time.time()
            self.alert_sent = False # Reset trạng thái cảnh báo

    def stop(self):
        self.is_running = False
        if self.accelerator.is_main_process:
            send_telegram_alert("✅ <b>Quá trình Training đã hoàn tất an toàn!</b>")

    def send_mess(self, mess):
        send_telegram_alert(message=mess)

    def _monitor(self):
        # """Hàm này chạy ngầm liên tục để kiểm tra thời gian"""
        # while self.is_running:
        #     time.sleep(30) # Kiểm tra mỗi 10 giây
        #     elapsed_time = time.time() - self.last_step_time
            
        #     # Nếu thời gian vượt mức cho phép và chưa gửi cảnh báo
        #     if elapsed_time > self.timeout_seconds and not self.alert_sent:
        #         msg = (f"⚠️ <b>CẢNH BÁO: TRAINING BỊ TREO!</b>\n"
        #                f"Đã <code>{int(elapsed_time)}</code> giây trôi qua mà chưa xong 1 step.\n"
        #                f"Vui lòng kiểm tra lại server/GPU ngay lập tức!")
        #         send_telegram_alert(msg)
        #         self.alert_sent = True
        pass
                