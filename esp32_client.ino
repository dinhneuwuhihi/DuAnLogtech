/*
  ESP32 CLIENT - Hệ thống phát hiện cháy/khói
  ---------------------------------------------
  Đọc DHT11 + quang trở + LM393 + MQ-2, gửi JSON lên server (Flask), nhận
  quyết định báo động và điều khiển OLED/LED RGB/Buzzer/Relay.

  Chưa làm (có thể bổ sung sau):
   - ArduinoOTA, esp_task_wdt tường minh, NTP time, ghi buffer ra flash khi
     mất mạng để không mất dữ liệu training.

  Thư viện cần cài (Library Manager):
   - DHT sensor library (Adafruit) + Adafruit Unified Sensor
   - Adafruit SSD1306 + Adafruit GFX Library
   - ArduinoJson (bản 7.x)
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "secrets.h"   // WIFI_SSID, WIFI_PASSWORD, SERVER_HOST, SERVER_PORT, DEVICE_ID

// ====== CHÂN KẾT NỐI (chỉnh lại theo board thực tế của bạn) ======
#define DHTPIN          4      // Chân data của DHT11
#define DHTTYPE         DHT11
#define LDR_PIN         34     // Analog - quang trở
#define IR_PIN          35     // Digital OUT - module LM393
#define MQ2_PIN         32     // Analog (A0) - module MQ-2
#define BUZZER_PIN      26
#define LED_R_PIN       27
#define LED_G_PIN       14
#define LED_B_PIN       19     // ĐÃ ĐỔI: GPIO12 là chân strapping, tránh dùng để lái LED
#define RELAY1_PIN      25
#define RELAY2_PIN      33
#define RESET_BUTTON_PIN 15    // Nút nhấn: 1 chân xuống GND, dùng INPUT_PULLUP

// Đa số module relay 5V thông dụng là active-LOW (tín hiệu LOW = đóng tiếp
// điểm). Nếu relay của bạn là active-HIGH, đổi thành false.
#define RELAY_ACTIVE_LOW true
#if RELAY_ACTIVE_LOW
  #define RELAY_ON  LOW
  #define RELAY_OFF HIGH
#else
  #define RELAY_ON  HIGH
  #define RELAY_OFF LOW
#endif

// ====== LEDC (PWM) cho LED RGB + Buzzer thụ động ======
#define LEDC_RES_BITS   8
#define LEDC_FREQ_RGB   5000
#define CH_LED_R        0
#define CH_LED_G        1
#define CH_LED_B        2
#define CH_BUZZER       3
#define BUZZER_TONE_HZ  2500

// ====== OLED ======
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

DHT dht(DHTPIN, DHTTYPE);

// ====== ĐỊA CHỈ SERVER (ghép từ secrets.h) ======
const String SERVER_URL = String("http://") + SERVER_HOST + ":" + String(SERVER_PORT) + "/predict";
const String RESET_URL  = String("http://") + SERVER_HOST + ":" + String(SERVER_PORT) + "/reset/" + DEVICE_ID;

// ====== THỜI GIAN GỬI DỮ LIỆU ======
unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL = 1000; // 1 mẫu / giây, khớp SAMPLE_PERIOD_S trong features.py

// ====== WARM-UP MQ-2 (chỉ ổn định dây đốt, KHÔNG thay cho burn-in 24-48h) ======
unsigned long bootTime = 0;
const unsigned long MQ2_WARMUP_MS = 30000;

// ====== WIFI: kết nối không chặn luồng, có timeout + backoff ======
bool wifiConnecting = false;
unsigned long wifiAttemptStart = 0;
unsigned long wifiNextAttemptAt = 0;
unsigned long wifiBackoffMs = 1000;
const unsigned long WIFI_ATTEMPT_TIMEOUT_MS = 10000;
const unsigned long WIFI_BACKOFF_MAX_MS = 30000;

// ====== GIỮ GIÁ TRỊ CẢM BIẾN HỢP LỆ GẦN NHẤT (thay vì bỏ qua khi lỗi) ======
float lastValidTemp = 25.0;
float lastValidHum  = 50.0;
uint32_t dhtErrorCount = 0;

// ====== LỊCH SỬ NHIỆT ĐỘ TẠI CHỖ (cho luật fail-safe khi mất server) ======
#define TEMP_HIST_LEN 6
float tempHist[TEMP_HIST_LEN];
unsigned long tempHistTime[TEMP_HIST_LEN];
uint8_t tempHistCount = 0, tempHistHead = 0;

// ====== LUẬT NGƯỠNG CỨNG TẠI CHỖ (PHẢI khớp GAS_HARD_LIMIT/TEMP_HARD_LIMIT/
// TEMP_RISE_HARD_LIMIT trong inference_server.py, để hành vi lúc mất mạng
// giống hệt lúc còn mạng) ======
const int   GAS_HARD_LIMIT       = 2500;
const float TEMP_HARD_LIMIT      = 60.0;
const float TEMP_RISE_HARD_LIMIT = 5.0; // độ C tăng trong ~5 giây

// ====== CHỐT BÁO ĐỘNG (latch) - chỉ gỡ khi bấm nút reset ======
bool alarmLatched = false;
String lastAlarmReason = "-";

// ====== KẾT QUẢ TỪ SERVER ======
// Khai báo NGAY TẠI ĐÂY (trước mọi hàm dùng nó) vì Arduino IDE tự sinh
// prototype cho các hàm ở đầu file: nếu struct này nằm sau hàm sử dụng nó,
// bản build sẽ báo lỗi "unknown type name 'ServerResult'".
struct ServerResult {
  bool httpOk = false;      // request có tới nơi và server trả 200 không
  bool alarm = false;
  float p_fire = -1.0;      // -1 nghĩa là không có thông tin
  bool warmingUp = false;
  String reason = "-";
  int httpCode = -1;
};

// ====== NHẤP NHÁY LED KHÔNG CHẶN LUỒNG ======
bool blinkOn = false;
unsigned long lastBlinkToggle = 0;
const unsigned long BLINK_INTERVAL_MS = 400;

// ====================================================================


// ====================================================================
// ADC: LẤY TRUNG VỊ ĐỂ GIẢM NHIỄU
// ====================================================================

int readAnalogMedian(int pin, uint8_t samples = 9) {
  int vals[9];
  for (uint8_t i = 0; i < samples; i++) {
    vals[i] = analogRead(pin);
    delayMicroseconds(200);
  }
  // insertion sort - đủ nhanh với mảng nhỏ (9 phần tử)
  for (uint8_t i = 1; i < samples; i++) {
    int key = vals[i];
    int j = i;
    while (j > 0 && vals[j - 1] > key) {
      vals[j] = vals[j - 1];
      j--;
    }
    vals[j] = key;
  }
  return vals[samples / 2];
}

void setup() {
  Serial.begin(115200);

  pinMode(IR_PIN, INPUT);
  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);
  pinMode(RELAY1_PIN, OUTPUT);
  pinMode(RELAY2_PIN, OUTPUT);
  digitalWrite(RELAY1_PIN, RELAY_OFF);
  digitalWrite(RELAY2_PIN, RELAY_OFF);

  // LEDC cho LED RGB + buzzer (thay analogWrite - không có trên core 2.x)
  ledcAttach(LED_R_PIN, LEDC_FREQ_RGB, LEDC_RES_BITS);
  ledcAttach(LED_G_PIN, LEDC_FREQ_RGB, LEDC_RES_BITS);
  ledcAttach(LED_B_PIN, LEDC_FREQ_RGB, LEDC_RES_BITS);
  ledcAttach(BUZZER_PIN, BUZZER_TONE_HZ, LEDC_RES_BITS);
  buzzerOff();
  setRGB(0, 0, 0);

  dht.begin();

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("Khong tim thay OLED!");
  }
  showMessage("Dang khoi dong...");

  bootTime = millis();
  startWifiAttempt();
}

void loop() {
  serviceWiFi();
  serviceResetButton();

  // Trong lúc MQ-2 làm nóng: vẫn hiển thị đếm ngược, KHÔNG gửi dữ liệu
  // (dữ liệu lúc này không đáng tin, gửi lên chỉ làm bẩn tập huấn luyện).
  if (millis() - bootTime < MQ2_WARMUP_MS) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.println("MQ-2 dang lam nong...");
    display.print((MQ2_WARMUP_MS - (millis() - bootTime)) / 1000);
    display.println(" giay con lai");
    display.display();
    serviceAlarmOutputs(); // vẫn cho phép chốt báo động cũ tiếp tục hoạt động
    return;
  }

  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();
    runSensorCycle();
  }

  serviceAlarmOutputs(); // nhấp nháy LED/còi không chặn luồng, chạy mỗi vòng lặp
  yield(); // nhường CPU cho task WiFi/hệ thống, tránh watchdog reset
}

// ====================================================================
// ĐỌC CẢM BIẾN + FAIL-SAFE + GỬI SERVER
// ====================================================================

void runSensorCycle() {
  // ---- 1. Đọc cảm biến (ADC lấy trung vị để giảm nhiễu) ----
  bool sensorOk = true;

  float rawTemp = dht.readTemperature();
  float rawHum  = dht.readHumidity();
  if (isnan(rawTemp) || isnan(rawHum)) {
    dhtErrorCount++;
    sensorOk = false;
    Serial.printf("Loi doc DHT11 (lan %lu), dung gia tri cu.\n", (unsigned long)dhtErrorCount);
  } else {
    lastValidTemp = rawTemp;
    lastValidHum  = rawHum;
  }
  float temp = lastValidTemp;
  float hum  = lastValidHum;

  int light = readAnalogMedian(LDR_PIN);
  int gas   = readAnalogMedian(MQ2_PIN);
  int ir    = digitalRead(IR_PIN);

  unsigned long now = millis();
  pushTempHistory(temp, now);

  // ---- 2. Luật ngưỡng cứng TẠI CHỖ (chạy độc lập, luôn luôn) ----
  float tempRise5s = getTempRise5s(temp, now);
  bool localHardAlarm = (gas >= GAS_HARD_LIMIT) ||
                         (temp >= TEMP_HARD_LIMIT) ||
                         (tempRise5s >= TEMP_RISE_HARD_LIMIT);

  // ---- 3. Gửi lên server (có timeout) & nhận quyết định ----
  ServerResult res = sendToServer(temp, hum, light, ir, gas, sensorOk);

  // ---- 4. Gộp quyết định: cứng tại chỗ HOẶC server báo cháy ----
  bool finalAlarm = localHardAlarm || (res.httpOk && res.alarm);
  String reason;
  if (localHardAlarm) reason = res.httpOk ? "rule (co server)" : "rule (mat server)";
  else if (res.httpOk && res.alarm) reason = "model";
  else reason = "safe";

  if (finalAlarm) {
    alarmLatched = true;   // CHỐT lại - chỉ nút reset mới gỡ được
    lastAlarmReason = reason;
  }

  updateDisplay(temp, hum, light, gas, res, localHardAlarm);
}

// ====================================================================
// GIAO TIẾP SERVER
// ====================================================================

ServerResult sendToServer(float temp, float hum, int light, int ir, int gas, bool sensorOk) {
  ServerResult result;

  if (WiFi.status() != WL_CONNECTED) {
    result.httpCode = -1;
    return result; // không mất công mở HTTPClient khi chắc chắn không có mạng
  }

  HTTPClient http;
  http.setConnectTimeout(2000); // tránh loop bị treo nhiều giây khi server chết
  http.setTimeout(3000);
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  JsonDocument doc; // ArduinoJson v7: tự quản lý kích thước, không cần template<N>
  doc["temp"]      = temp;
  doc["hum"]       = hum;
  doc["light"]     = light;
  doc["ir"]        = ir;
  doc["gas"]       = gas;
  doc["device_id"] = DEVICE_ID;
  doc["sensor_ok"] = sensorOk ? 1 : 0;

  String requestBody;
  serializeJson(doc, requestBody);

  int httpCode = http.POST(requestBody);
  result.httpCode = httpCode;

  if (httpCode == 200) {
    String response = http.getString();
    JsonDocument resDoc;
    DeserializationError err = deserializeJson(resDoc, response);
    if (!err) {
      result.httpOk = true;
      // Ưu tiên "alarm" (schema mới); nếu không có, suy ra từ "prediction"
      // để vẫn tương thích ngược với server cũ.
      if (resDoc["alarm"].is<bool>()) {
        result.alarm = resDoc["alarm"].as<bool>();
      } else if (resDoc["prediction"].is<int>()) {
        result.alarm = (resDoc["prediction"].as<int>() == 1);
      }
      if (resDoc["p_fire"].is<float>()) result.p_fire = resDoc["p_fire"].as<float>();
      result.warmingUp = resDoc["warming_up"] | false;
      result.reason = resDoc["reason"] | "server";
    } else {
      Serial.println("Loi parse JSON tu server: " + String(err.c_str()));
    }
    Serial.println("Server response: " + response);
  } else if (httpCode == 400) {
    // Dữ liệu bị server từ chối (validate_reading nem ValueError) - KHÔNG
    // phải sự cố mạng, mà là dữ liệu cảm biến có vấn đề. In ra để debug.
    String response = http.getString();
    Serial.println("Server tu choi du lieu (400): " + response);
  } else {
    Serial.println("Loi HTTP: " + String(httpCode));
  }

  http.end();
  return result;
}

// ====================================================================
// WIFI - KHÔNG CHẶN LUỒNG, CÓ TIMEOUT + BACKOFF
// ====================================================================

void startWifiAttempt() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  wifiConnecting = true;
  wifiAttemptStart = millis();
  Serial.println("Dang thu ket noi WiFi...");
}

void serviceWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    if (wifiConnecting) {
      wifiConnecting = false;
      wifiBackoffMs = 1000; // reset backoff khi đã nối thành công
      Serial.println("Da ket noi WiFi! IP: " + WiFi.localIP().toString());
    }
    return;
  }

  if (wifiConnecting) {
    // Đang thử mà quá lâu không được -> bỏ cuộc lần này, chờ backoff rồi thử lại
    if (millis() - wifiAttemptStart > WIFI_ATTEMPT_TIMEOUT_MS) {
      wifiConnecting = false;
      wifiNextAttemptAt = millis() + wifiBackoffMs;
      Serial.println("WiFi timeout, thu lai sau " + String(wifiBackoffMs / 1000) + "s");
      wifiBackoffMs = min(wifiBackoffMs * 2, WIFI_BACKOFF_MAX_MS);
    }
  } else if (millis() >= wifiNextAttemptAt) {
    startWifiAttempt();
  }
}

// ====================================================================
// NÚT RESET VẬT LÝ (gỡ chốt báo động)
// ====================================================================

void serviceResetButton() {
  static unsigned long lastPress = 0;
  if (digitalRead(RESET_BUTTON_PIN) == LOW && millis() - lastPress > 300) { // chống dội phím
    lastPress = millis();
    if (alarmLatched) {
      alarmLatched = false;
      lastAlarmReason = "-";
      Serial.println(">>> Da RESET bao dong bang nut nhan <<<");
      if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.setConnectTimeout(1500);
        http.setTimeout(2000);
        http.begin(RESET_URL);
        http.GET(); // đồng bộ gỡ chốt bên server (không cần đọc response)
        http.end();
      }
    }
  }
}

// ====================================================================
// FAIL-SAFE TẠI CHỖ: lịch sử nhiệt độ ~5 giây gần nhất
// ====================================================================

void pushTempHistory(float t, unsigned long now) {
  tempHist[tempHistHead] = t;
  tempHistTime[tempHistHead] = now;
  tempHistHead = (tempHistHead + 1) % TEMP_HIST_LEN;
  if (tempHistCount < TEMP_HIST_LEN) tempHistCount++;
}

float getTempRise5s(float currentTemp, unsigned long now) {
  float oldestInWindow = currentTemp;
  unsigned long bestAge = 0;
  for (uint8_t i = 0; i < tempHistCount; i++) {
    unsigned long age = now - tempHistTime[i];
    if (age <= 6000 && age > bestAge) { // xấp xỉ cửa sổ ~5-6 giây
      bestAge = age;
      oldestInWindow = tempHist[i];
    }
  }
  return currentTemp - oldestInWindow;
}



// ====================================================================
// HIỂN THỊ + ĐẦU RA (LED/BUZZER/RELAY) - KHÔNG DÙNG delay()
// ====================================================================

void updateDisplay(float temp, float hum, int light, int gas,
                    const ServerResult& res, bool localHardAlarm) {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.print("Nhiet do: "); display.print(temp, 1); display.println(" C");
  display.print("Do am: ");    display.print(hum, 0);  display.println(" %");
  display.print("Anh sang: "); display.println(light);
  display.print("Khi gas: ");  display.println(gas);

  if (!res.httpOk) {
    display.println("Server: MAT KET NOI");
  } else if (res.warmingUp) {
    display.println("Server: dang khoi dong");
  } else if (res.p_fire >= 0) {
    display.print("P(chay)="); display.println(res.p_fire, 2);
  }
  display.println("--------------------");

  if (alarmLatched) {
    display.setTextSize(2);
    display.println("CANH BAO!");
    display.println("CHAY/KHOI");
    display.setTextSize(1);
    display.print("Ly do: "); display.println(lastAlarmReason);
    display.println("(Bam nut de reset)");
  } else {
    display.setTextSize(2);
    display.println("An toan");
  }

  display.display();
}

void serviceAlarmOutputs() {
  if (alarmLatched) {
    digitalWrite(RELAY1_PIN, RELAY_ON);
    digitalWrite(RELAY2_PIN, RELAY_ON);
    buzzerOn();

    // Nhấp nháy đỏ không chặn luồng (thay cho delay(150) ở bản cũ)
    if (millis() - lastBlinkToggle >= BLINK_INTERVAL_MS) {
      lastBlinkToggle = millis();
      blinkOn = !blinkOn;
      setRGB(blinkOn ? 255 : 0, 0, 0);
    }
  } else {
    digitalWrite(RELAY1_PIN, RELAY_OFF);
    digitalWrite(RELAY2_PIN, RELAY_OFF);
    buzzerOff();
    setRGB(0, 255, 0); // Xanh lá = an toàn
  }
}

void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  ledcWrite(LED_R_PIN, r);
  ledcWrite(LED_G_PIN, g);
  ledcWrite(LED_B_PIN, b);
}

void buzzerOn()  { ledcWriteTone(BUZZER_PIN, BUZZER_TONE_HZ); }
void buzzerOff() { ledcWriteTone(BUZZER_PIN, 0); }

void showMessage(const char* msg) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(msg);
  display.display();
}
