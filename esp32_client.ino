/*
  ESP32 CLIENT - Hệ thống phát hiện cháy/khói  (bản v2)
  ------------------------------------------------------
  Chức năng:
   1. Đọc DHT11 (nhiệt độ, độ ẩm), quang trở (analog), LM393 (digital), MQ-2 (analog)
   2. Gửi JSON lên server Python (/predict) và nhận {prediction, p_fire, alarm}
   3. Điều khiển OLED, LED RGB, buzzer, relay 2 kênh
   4. TỰ QUYẾT ĐỊNH BẰNG LUẬT NGƯỠNG KHI MẤT WIFI/SERVER (fail-safe)

  NHỮNG THỨ ĐÃ SỬA SO VỚI BẢN CŨ - đọc trước khi nạp code:
   1. FAIL-SAFE CỤC BỘ: bản cũ khi mất server chỉ hiện "Mat ket noi Server" rồi
      thôi - tức là đầu báo cháy ngừng báo cháy đúng lúc mạng chập chờn. Bản này
      có luật ngưỡng chạy ngay trên ESP32 (gas cao / quá nóng / nhiệt tăng nhanh).
   2. KHÔNG CÒN delay() TRONG LOOP: nhấp nháy LED, còi, OLED đều chạy bằng
      máy trạng thái theo millis(). delay(150) cũ làm cả vòng lặp bị chặn.
   3. WiFi KHÔNG CÒN while(...) VÔ HẠN: có timeout + thử lại theo chu kỳ, nên
      thiết bị vẫn đọc cảm biến và vẫn báo động khi chưa có mạng.
   4. analogWrite() KHÔNG tồn tại trên arduino-esp32 core 2.x -> dùng LEDC
      (có #if để chạy được cả core 2.x và 3.x).
   5. LED_B chuyển khỏi GPIO12: GPIO12 là chân strapping (MTDI), kéo mức cao
      lúc boot có thể đặt sai điện áp flash và làm board không boot được.
   6. RELAY_ACTIVE_LOW: phần lớn module relay là kích mức THẤP, nên
      digitalWrite(LOW) của bản cũ có thể đang BẬT relay ngay khi khởi động.
   7. CHỐT BÁO ĐỘNG (latch) + nút RESET/MUTE: đã xác nhận cháy thì 1 mẫu
      "an toàn" không được phép tắt bơm; phải có người bấm nút.
   8. Lọc ADC bằng trung vị 9 mẫu (ADC ESP32 rất nhiễu, nhiễu đó chảy thẳng
      vào đặc trưng tốc-độ-thay-đổi của mô hình).
   9. Lỗi đọc DHT không làm bỏ qua cả vòng lặp: giữ giá trị hợp lệ cuối cùng,
      đếm số lỗi và gửi cờ sensor_ok cho server.
  10. Gửi kèm device_id (từ MAC) + seq + uptime; HTTP có timeout để server
      chết không làm treo vòng lặp.

  Thư viện cần cài (Sketch > Include Library > Manage Libraries):
   - DHT sensor library (Adafruit) + Adafruit Unified Sensor
   - Adafruit SSD1306 + Adafruit GFX Library
   - ArduinoJson (v6 hoặc v7 đều chạy)
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ====== CẤU HÌNH WIFI & SERVER ======
// Nên tạo file secrets.h (và thêm vào .gitignore) chứa:
//   #define CFG_WIFI_SSID "..."
//   #define CFG_WIFI_PASSWORD "..."
//   #define CFG_SERVER_URL "http://192.168.1.100:5000/predict"
#if __has_include("secrets.h")
  #include "secrets.h"
#endif
#ifndef CFG_WIFI_SSID
  #define CFG_WIFI_SSID     "TEN_WIFI_CUA_BAN"
#endif
#ifndef CFG_WIFI_PASSWORD
  #define CFG_WIFI_PASSWORD "MAT_KHAU_WIFI"
#endif
#ifndef CFG_SERVER_URL
  #define CFG_SERVER_URL    "http://192.168.1.100:5000/predict"
#endif

const char* WIFI_SSID     = CFG_WIFI_SSID;
const char* WIFI_PASSWORD = CFG_WIFI_PASSWORD;
const char* SERVER_URL    = CFG_SERVER_URL;

// ====== CHÂN KẾT NỐI (chỉnh theo board thực tế) ======
#define DHTPIN        4
#define DHTTYPE       DHT11
#define LDR_PIN       34     // ADC1 - chỉ ADC1 dùng được khi WiFi bật
#define IR_PIN        35
#define MQ2_PIN       32
#define BUZZER_PIN    26
#define LED_R_PIN     27
#define LED_G_PIN     14
#define LED_B_PIN     13     // (cũ: 12 - chân strapping, KHÔNG nên dùng)
#define RELAY1_PIN    25
#define RELAY2_PIN    33
#define BUTTON_PIN    0      // nút BOOT trên hầu hết devkit: reset/mute báo động

// ====== CẤU HÌNH PHẦN CỨNG ======
#define RELAY_ACTIVE_LOW   1    // 1 = module relay kích mức THẤP (phổ biến nhất)
#define BUZZER_PASSIVE     0    // 1 = buzzer thụ động (cần phát tần số), 0 = còi chủ động
#define BUZZER_TONE_HZ     2400

#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
#define OLED_ADDR     0x3C

// ====== THAM SỐ THỜI GIAN ======
const unsigned long SEND_INTERVAL      = 1000;   // 1 mẫu / giây (khớp features.py)
const unsigned long DHT_INTERVAL       = 2000;   // DHT11 chỉ đọc được ~0.5Hz
const unsigned long DISPLAY_INTERVAL   = 250;
const unsigned long MQ2_WARMUP_MS      = 30000;  // heater MQ-2 (ổn định thật cần 24-48h burn-in)
const unsigned long WIFI_ATTEMPT_MS    = 15000;  // thử kết nối tối đa 15s mỗi lần
const unsigned long WIFI_RETRY_MIN_MS  = 5000;   // backoff tối thiểu
const unsigned long WIFI_RETRY_MAX_MS  = 60000;  // backoff tối đa
const unsigned long SERVER_STALE_MS    = 10000;  // quá lâu không có phản hồi = mất server
const unsigned long BLINK_MS           = 250;

// ====== LUẬT NGƯỠNG CỤC BỘ (fail-safe, phải khớp inference_server.py) ======
const int   GAS_HARD_LIMIT      = 2500;   // ADC
const float TEMP_HARD_LIMIT     = 60.0;   // độ C
const float TEMP_RISE_LIMIT_5S  = 5.0;    // độ C / 5 giây
const uint8_t LOCAL_CONFIRM      = 3;     // cần 3 mẫu liên tiếp mới báo (chống nhiễu)

// ====== LEDC (PWM) ======
#define LEDC_FREQ       5000
#define LEDC_BITS       8
#define CH_LED_R        0
#define CH_LED_G        1
#define CH_LED_B        2
#define CH_BUZZER       3

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
DHT dht(DHTPIN, DHTTYPE);

// ====== TRẠNG THÁI ======
struct Reading {
  float temp = NAN;
  float hum  = NAN;
  int   light = 0;
  int   ir    = 0;
  int   gas   = 0;
  bool  sensorOk = false;
};

Reading current;
bool     oledOk        = false;
uint32_t seq           = 0;
uint16_t dhtErrors     = 0;
unsigned long bootTime = 0;
unsigned long lastSend = 0, lastDht = 0, lastDisplay = 0, lastBlink = 0;
unsigned long lastServerOkMs = 0;
unsigned long wifiAttemptStart = 0, wifiNextRetry = 0, wifiBackoff = WIFI_RETRY_MIN_MS;
bool wifiConnecting = false;

int   serverPrediction = -1;   // -1 = chưa biết / mất kết nối
float serverPFire      = -1.0;
bool  serverOnline     = false;

bool  alarmLatched   = false;  // đã xác nhận cháy -> chốt đến khi bấm nút
bool  muted          = false;  // tạm tắt còi (relay vẫn giữ)
bool  localAlarm     = false;  // do luật ngưỡng trên ESP32 quyết định
uint8_t localHits    = 0;
bool  blinkOn        = false;

// Lịch sử nhiệt độ ~5 giây (6 ô, 1 ô/giây) để tính tốc độ tăng nhiệt cục bộ
const uint8_t TEMP_HIST = 6;
float  tempHist[TEMP_HIST];
uint8_t tempHistCount = 0, tempHistIdx = 0;

unsigned long lastButtonMs = 0;
bool lastButtonState = HIGH;

// ---------------------------------------------------------------- PWM helpers
void pwmAttach(uint8_t pin, uint8_t channel, uint32_t freq) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)channel;
  ledcAttach(pin, freq, LEDC_BITS);
#else
  ledcSetup(channel, freq, LEDC_BITS);
  ledcAttachPin(pin, channel);
#endif
}

void pwmWrite(uint8_t pin, uint8_t channel, uint32_t duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)channel;
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(channel, duty);
#endif
}

void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  pwmWrite(LED_R_PIN, CH_LED_R, r);
  pwmWrite(LED_G_PIN, CH_LED_G, g);
  pwmWrite(LED_B_PIN, CH_LED_B, b);
}

void setBuzzer(bool on) {
#if BUZZER_PASSIVE
  pwmWrite(BUZZER_PIN, CH_BUZZER, on ? 128 : 0);
#else
  digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
#endif
}

void setRelay(uint8_t pin, bool on) {
#if RELAY_ACTIVE_LOW
  digitalWrite(pin, on ? LOW : HIGH);
#else
  digitalWrite(pin, on ? HIGH : LOW);
#endif
}

String deviceId() {
  uint64_t mac = ESP.getEfuseMac();
  char buf[24];
  snprintf(buf, sizeof(buf), "esp32-%04X%08X",
           (uint16_t)(mac >> 32), (uint32_t)mac);
  return String(buf);
}

// ---------------------------------------------------------------- setup
void setup() {
  Serial.begin(115200);
  delay(50);

  pinMode(IR_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(RELAY1_PIN, OUTPUT);
  pinMode(RELAY2_PIN, OUTPUT);
  setRelay(RELAY1_PIN, false);   // TẮT thật sự, không phụ thuộc mức logic
  setRelay(RELAY2_PIN, false);

  pwmAttach(LED_R_PIN, CH_LED_R, LEDC_FREQ);
  pwmAttach(LED_G_PIN, CH_LED_G, LEDC_FREQ);
  pwmAttach(LED_B_PIN, CH_LED_B, LEDC_FREQ);
#if BUZZER_PASSIVE
  pwmAttach(BUZZER_PIN, CH_BUZZER, BUZZER_TONE_HZ);
#else
  pinMode(BUZZER_PIN, OUTPUT);
#endif
  setBuzzer(false);
  setRGB(0, 0, 0);

  analogReadResolution(12);
  analogSetPinAttenuation(LDR_PIN, ADC_11db);
  analogSetPinAttenuation(MQ2_PIN, ADC_11db);

  dht.begin();

  oledOk = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!oledOk) {
    Serial.println("Khong tim thay OLED - van chay tiep (khong co man hinh).");
  } else {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Dang khoi dong...");
    display.println(deviceId());
    display.display();
  }

  Serial.println("Device ID: " + deviceId());
  startWiFi();
  bootTime = millis();
}

// ---------------------------------------------------------------- WiFi (non-blocking)
void startWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  wifiConnecting = true;
  wifiAttemptStart = millis();
  Serial.println("Dang ket noi WiFi...");
}

void serviceWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    if (wifiConnecting) {
      wifiConnecting = false;
      wifiBackoff = WIFI_RETRY_MIN_MS;
      Serial.println("Da ket noi WiFi! IP: " + WiFi.localIP().toString());
    }
    return;
  }

  if (wifiConnecting) {
    // Có timeout, KHÔNG while(...) vô hạn như bản cũ
    if (millis() - wifiAttemptStart > WIFI_ATTEMPT_MS) {
      Serial.println("Ket noi WiFi that bai -> se thu lai (van do cam bien binh thuong).");
      WiFi.disconnect(true);
      wifiConnecting = false;
      wifiNextRetry = millis() + wifiBackoff;
      wifiBackoff = min(wifiBackoff * 2, WIFI_RETRY_MAX_MS);  // exponential backoff
    }
    return;
  }

  if (millis() >= wifiNextRetry) {
    startWiFi();
  }
}

// ---------------------------------------------------------------- đọc cảm biến
int readAdcMedian(uint8_t pin) {
  const uint8_t N = 9;
  int samples[N];
  for (uint8_t i = 0; i < N; i++) {
    samples[i] = analogRead(pin);
    delayMicroseconds(200);
  }
  for (uint8_t i = 1; i < N; i++) {          // insertion sort (N nhỏ)
    int key = samples[i];
    int8_t j = i - 1;
    while (j >= 0 && samples[j] > key) {
      samples[j + 1] = samples[j];
      j--;
    }
    samples[j + 1] = key;
  }
  return samples[N / 2];
}

void readSensors() {
  current.light = readAdcMedian(LDR_PIN);
  current.gas   = readAdcMedian(MQ2_PIN);
  current.ir    = digitalRead(IR_PIN);

  if (millis() - lastDht >= DHT_INTERVAL) {
    lastDht = millis();
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (isnan(t) || isnan(h)) {
      dhtErrors++;
      Serial.println("Loi doc DHT11 - giu lai gia tri hop le cuoi cung.");
    } else {
      current.temp = t;
      current.hum  = h;
      current.sensorOk = true;
    }
  }
}

void pushTempHistory(float temp) {
  tempHist[tempHistIdx] = temp;
  tempHistIdx = (tempHistIdx + 1) % TEMP_HIST;
  if (tempHistCount < TEMP_HIST) tempHistCount++;
}

float tempRise5s() {
  if (tempHistCount < TEMP_HIST) return 0.0;     // chưa đủ lịch sử -> không kết luận
  uint8_t oldest = tempHistIdx;                  // ô sắp bị ghi đè = cũ nhất
  uint8_t newest = (tempHistIdx + TEMP_HIST - 1) % TEMP_HIST;
  return tempHist[newest] - tempHist[oldest];
}

// ---------------------------------------------------------------- luật cục bộ
bool evaluateLocalRule() {
  bool danger = current.gas >= GAS_HARD_LIMIT;
  if (current.sensorOk) {
    danger = danger || current.temp >= TEMP_HARD_LIMIT
                    || tempRise5s() >= TEMP_RISE_LIMIT_5S;
  }
  if (danger) {
    if (localHits < 255) localHits++;
  } else {
    localHits = 0;
  }
  return localHits >= LOCAL_CONFIRM;
}

// ---------------------------------------------------------------- gửi server
int sendToServer() {
  if (WiFi.status() != WL_CONNECTED) return -1;

  HTTPClient http;
  http.setConnectTimeout(2000);
  http.setTimeout(3000);           // server chết cũng không treo vòng lặp
  http.setReuse(false);
  if (!http.begin(SERVER_URL)) {
    Serial.println("http.begin() that bai - kiem tra SERVER_URL.");
    return -1;
  }
  http.addHeader("Content-Type", "application/json");

#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  StaticJsonDocument<256> doc;
#endif
  doc["device_id"] = deviceId();
  doc["seq"]       = ++seq;
  doc["uptime_ms"] = millis();
  doc["temp"]      = current.sensorOk ? current.temp : 0;
  doc["hum"]       = current.sensorOk ? current.hum  : 0;
  doc["light"]     = current.light;
  doc["ir"]        = current.ir;
  doc["gas"]       = current.gas;
  doc["sensor_ok"] = current.sensorOk ? 1 : 0;

  String body;
  serializeJson(doc, body);

  int httpCode = http.POST(body);
  int prediction = -1;
  serverPFire = -1.0;

  if (httpCode == 200) {
    String response = http.getString();
#if ARDUINOJSON_VERSION_MAJOR >= 7
    JsonDocument res;
#else
    StaticJsonDocument<384> res;
#endif
    if (!deserializeJson(res, response)) {
      prediction  = res["prediction"] | -1;
      serverPFire = res["p_fire"] | -1.0f;
      if (res["alarm"].is<bool>()) {
        prediction = res["alarm"].as<bool>() ? 1 : 0;
      }
      lastServerOkMs = millis();
    } else {
      Serial.println("JSON tra ve khong doc duoc: " + response);
    }
  } else {
    Serial.println("Loi HTTP: " + String(httpCode));
  }

  http.end();
  return prediction;
}

// ---------------------------------------------------------------- nút bấm
void serviceButton() {
  bool state = digitalRead(BUTTON_PIN);
  if (state == lastButtonState) return;
  if (millis() - lastButtonMs < 50) return;     // chống dội phím
  lastButtonMs = millis();
  lastButtonState = state;

  if (state == LOW) {                           // nhấn (INPUT_PULLUP)
    if (alarmLatched && !muted) {
      muted = true;                             // lần 1: tắt còi, relay vẫn chạy
      Serial.println("Nut: TAT COI (relay van giu).");
    } else {
      alarmLatched = false;                     // lần 2: reset hẳn báo động
      muted = false;
      localHits = 0;
      Serial.println("Nut: RESET bao dong.");
    }
  }
}

// ---------------------------------------------------------------- đầu ra
void applyOutputs() {
  bool alarm = alarmLatched;

  setRelay(RELAY1_PIN, alarm);        // VD: quạt hút khói / bơm phun sương
  setRelay(RELAY2_PIN, alarm);        // VD: ngắt nguồn băng chuyền
  setBuzzer(alarm && !muted && blinkOn);

  if (alarm) {
    setRGB(blinkOn ? 255 : 0, 0, 0);            // đỏ nhấp nháy (không dùng delay)
  } else if (!serverOnline) {
    setRGB(0, 0, blinkOn ? 180 : 20);           // xanh dương = đang chạy offline
  } else {
    setRGB(0, 120, 0);                          // xanh lá = an toàn
  }
}

void renderDisplay(bool warmingUp, unsigned long warmupLeft) {
  if (!oledOk) return;
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);

  if (warmingUp) {
    display.println("MQ-2 dang lam nong...");
    display.print(warmupLeft / 1000);
    display.println(" giay con lai");
    display.println();
    display.print("Gas: "); display.println(current.gas);
    display.print("WiFi: ");
    display.println(WiFi.status() == WL_CONNECTED ? "OK" : "...");
    display.display();
    return;
  }

  if (current.sensorOk) {
    display.print("T:"); display.print(current.temp, 1); display.print("C ");
    display.print("H:"); display.print(current.hum, 0);  display.println("%");
  } else {
    display.println("DHT11: LOI DOC");
  }
  display.print("Gas:"); display.print(current.gas);
  display.print(" Lt:"); display.println(current.light);
  display.print("IR:"); display.print(current.ir);
  display.print(" dT5s:"); display.println(tempRise5s(), 1);
  display.print(WiFi.status() == WL_CONNECTED ? "WiFi OK " : "WiFi -- ");
  if (serverOnline && serverPFire >= 0) {
    display.print("P:"); display.println(serverPFire, 2);
  } else {
    display.println(serverOnline ? "SRV OK" : "OFFLINE");
  }
  display.println("--------------------");

  display.setTextSize(2);
  if (alarmLatched) {
    display.println("CANH BAO!");
    display.setTextSize(1);
    display.print(localAlarm && !serverOnline ? "Luat cuc bo" : "Mo hinh AI");
    if (muted) display.print(" (da tat coi)");
  } else if (!serverOnline) {
    display.println("Offline");
    display.setTextSize(1);
    display.println("Dang dung luat cuc bo");
  } else {
    display.println("An toan");
  }
  display.display();
}

// ---------------------------------------------------------------- loop
void loop() {
  serviceWiFi();
  serviceButton();

  if (millis() - lastBlink >= BLINK_MS) {
    lastBlink = millis();
    blinkOn = !blinkOn;
  }

  bool warmingUp = (millis() - bootTime) < MQ2_WARMUP_MS;
  unsigned long warmupLeft = warmingUp ? MQ2_WARMUP_MS - (millis() - bootTime) : 0;

  if (millis() - lastSend >= SEND_INTERVAL) {
    lastSend = millis();

    readSensors();
    if (current.sensorOk) pushTempHistory(current.temp);

    if (!warmingUp) {
      serverPrediction = sendToServer();
      serverOnline = (millis() - lastServerOkMs) < SERVER_STALE_MS && lastServerOkMs != 0;

      localAlarm = evaluateLocalRule();

      // Quyết định cuối: server nói cháy HOẶC luật cục bộ nói cháy.
      // Khi mất server, luật cục bộ là thứ duy nhất giữ cho thiết bị còn hữu ích.
      if (serverPrediction == 1 || localAlarm) {
        if (!alarmLatched) {
          Serial.println(localAlarm && serverPrediction != 1
                         ? "BAO DONG (luat cuc bo tren ESP32)"
                         : "BAO DONG (server/AI)");
        }
        alarmLatched = true;   // chốt: chỉ nút bấm mới xoá được
      }

      Serial.printf("#%lu T=%.1f H=%.0f L=%d IR=%d G=%d dT5=%.1f srv=%d p=%.2f alarm=%d\n",
                    (unsigned long)seq, current.temp, current.hum, current.light,
                    current.ir, current.gas, tempRise5s(), serverPrediction,
                    serverPFire, alarmLatched ? 1 : 0);
    }
  }

  applyOutputs();

  if (millis() - lastDisplay >= DISPLAY_INTERVAL) {
    lastDisplay = millis();
    renderDisplay(warmingUp, warmupLeft);
  }
}
