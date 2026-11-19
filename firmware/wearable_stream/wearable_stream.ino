#include <Arduino.h>
#include <algorithm>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <NetworkClientSecure.h>
#include <ArduinoJson.h>
#include <sys/time.h>
#include "esp_sntp.h"
#include "esp_timer.h"
#include "esp_system.h"

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <TinyGPSPlus.h>

// The dashboard creates this ignored file while flashing. A build without it
// contains no Wi-Fi or relay secret and deliberately stays offline.
#if __has_include("generated_config.h")
#include "generated_config.h"
#else
inline constexpr int PLAYER_ID = 0;
inline constexpr char MATCH_ID[] = "live";
inline constexpr char WIFI_SSID[] = "";
inline constexpr char WIFI_PASS[] = "";
inline constexpr char RELAY_ENDPOINT[] = "https://cpg44.nivaspms.com/api/v1/sensors/ingest";
inline constexpr char RELAY_TOKEN[] = "";
inline constexpr char RELAY_CA_PEM[] = "";
#endif

#define I2C_SDA 4
#define I2C_SCL 5
#define GPS_RX 16
#define GPS_TX 17
#define GPS_BAUD 9600

// AD8232 single-lead ECG and battery divider: ADC1 pins only (ADC2 is not
// usable while Wi-Fi runs). GPIO 35-37 are taken by the N16R8 octal PSRAM and
// 0, 3, 45, 46 are strapping pins, so none of those are used.
#define ECG_OUT 2          // AD8232 OUTPUT  -> ADC1_CH1
#define ECG_LO_PLUS 11     // AD8232 LO+
#define ECG_LO_MINUS 12    // AD8232 LO-
#define BATTERY_ADC 1      // VBAT -> 100k -> GPIO1 -> 100k -> GND, 100 nF to GND
constexpr bool ECG_ENABLED = true;
constexpr uint32_t ECG_HZ = 250;
constexpr size_t ECG_BLOCK = 50;   // 0.2 s per packet

constexpr uint32_t I2C_HZ = 100000;       // stable on the tested board
constexpr uint32_t IMU_PERIOD_US = 20000; // 50 Hz

constexpr size_t RELAY_QUEUE_CAPACITY = 256;
// Throughput, not latency, limits delivery: one HTTPS POST through Cloudflare
// takes 0.5-1 s. The earlier prototype sent at most 32 samples per POST and
// lost half of its data. The vest now produces about 61 events per second
// (IMU 50 Hz, ECG 5 blocks of 50 samples, GNSS 5 Hz, status 0.5 Hz). The relay
// accepts 100 per POST; 96 per POST keeps up even at 1.2 s per round trip.
constexpr size_t RELAY_BATCH_CAPACITY = 96;
constexpr uint32_t RELAY_BATCH_WINDOW_MS = 25;
constexpr uint64_t RELAY_MAX_SAMPLE_AGE_US = 2000000ULL;

enum class SampleType : uint8_t { Imu, Gps, Status, Ecg };

struct SampleRecord {
  SampleType type;
  uint64_t sequence;
  uint64_t deviceUS;
  float accel[3];
  float gyro[3];
  float temperatureC;
  bool gpsRx;
  bool gpsFix;
  double latitude;
  double longitude;
  float speedMps;
  float courseDeg;
  float altitudeM;
  float hdop;
  uint16_t satellites;
  uint32_t gpsChars;
  int16_t rssiDbm;
  uint32_t freeHeap;
  bool gpsReceiving;
  float mag[3];
  bool magValid;
  uint16_t ecg[ECG_BLOCK];
  uint8_t ecgCount;
  bool leadOff;
  float batteryV;
  float batteryPct;
  float gpsRateHz;
  float ecgRateHz;
};

Adafruit_MPU6050 mpu;
TinyGPSPlus gps;
HardwareSerial gpsSerial(1);

bool mpuOK = false;
bool baroFound = false;     // BMP180 on the GY-87: detected for the build check, not streamed
enum class MagChip : uint8_t { None, HMC5883L, QMC5883L };
MagChip magChip = MagChip::None;
volatile uint32_t ecgBlocks = 0;
uint32_t ecgBlocksAtStatus = 0;
uint64_t ecgSeq = 0;
uint32_t gpsFixCount = 0;
uint32_t gpsFixCountAtStatus = 0;
uint32_t bootId = 0;
uint64_t imuSeq = 0;
uint64_t gpsSeq = 0;
uint64_t statusSeq = 0;
uint64_t lastIMUScheduleUS = 0;
uint32_t lastGPSPacketMS = 0;
uint32_t lastGPSByteMS = 0;
uint32_t lastStatusMS = 0;
uint32_t lastClockAnchorMS = 0;

QueueHandle_t relayQueue = nullptr;
SampleRecord relayBatch[RELAY_BATCH_CAPACITY];
size_t relayBatchCount = 0;
portMUX_TYPE stateMux = portMUX_INITIALIZER_UNLOCKED;
int64_t epochOffsetUS = 0;
bool clockReady = false;
uint32_t droppedSamples = 0;

static inline uint64_t deviceTimeUS() {
  return static_cast<uint64_t>(esp_timer_get_time());
}

bool probeI2C(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

// Some sensor libraries call Wire.begin() without pins; rebinding after each
// setup keeps the bus on the tested ESP32-S3 pins 4/5.
void bindI2C() {
  Wire.begin(I2C_SDA, I2C_SCL, I2C_HZ);
}

// Resting LiPo voltage to charge (single cell, light load). Coarse but honest:
// under the vest's ~0.2 A load the reading sits about 0.05 V low.
float lipoPercent(float v) {
  static const float V[] = {3.30f, 3.50f, 3.61f, 3.69f, 3.73f, 3.77f, 3.80f, 3.84f, 3.87f, 3.93f, 4.00f, 4.08f, 4.20f};
  static const float P[] = {0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100};
  if (v <= V[0]) return 0;
  for (int i = 1; i < 13; ++i)
    if (v < V[i]) return P[i - 1] + (P[i] - P[i - 1]) * (v - V[i - 1]) / (V[i] - V[i - 1]);
  return 100;
}

// Cell voltage through the 100k/100k divider (half the cell voltage at the
// pin, inside the ADC's range at 11 dB). Median of 15 readings corrected by the
// chip's eFuse ADC calibration, then the LiPo curve above.
void readBattery(SampleRecord &sample) {
  sample.batteryV = NAN;
  sample.batteryPct = NAN;
  uint32_t mv[15];
  for (int i = 0; i < 15; ++i) mv[i] = analogReadMilliVolts(BATTERY_ADC);
  std::sort(mv, mv + 15);
  const float v = 2.0f * mv[7] / 1000.0f;
  if (v > 2.8f && v < 4.4f) {
    sample.batteryV = v;
    sample.batteryPct = lipoPercent(v);
  }
}

// ---- u-blox NEO-6M: 5 Hz fixes, only the sentences we parse ----
// 1 Hz GNSS aliases 1-2 s sprints. The NEO-6M supports 5 Hz; with only RMC
// (Doppler speed and course) and GGA (fix, satellites, HDOP) enabled, 5 Hz
// needs about 7.3 kbit/s, which fits the module's default 9600 baud, so no
// baud switch is needed. The settings live in RAM and are re-sent every boot.
void sendUBX(uint8_t cls, uint8_t id, const uint8_t *payload, uint16_t len) {
  uint8_t a = 0, b = 0;
  auto ck = [&](uint8_t v) { a += v; b += a; };
  const uint8_t head[] = {cls, id, static_cast<uint8_t>(len & 0xFF), static_cast<uint8_t>(len >> 8)};
  gpsSerial.write(0xB5);
  gpsSerial.write(0x62);
  for (uint8_t v : head) { gpsSerial.write(v); ck(v); }
  for (uint16_t i = 0; i < len; ++i) { gpsSerial.write(payload[i]); ck(payload[i]); }
  gpsSerial.write(a);
  gpsSerial.write(b);
  gpsSerial.flush();
  delay(30);
}

void configureGPS() {
  const uint8_t off[][3] = {{0xF0, 0x01, 0}, {0xF0, 0x02, 0}, {0xF0, 0x03, 0}, {0xF0, 0x05, 0}};  // GLL GSA GSV VTG
  for (const auto &m : off) sendUBX(0x06, 0x01, m, 3);
  const uint8_t rate[] = {200, 0, 1, 0, 1, 0};  // CFG-RATE: 200 ms, 1 cycle, GPS time
  sendUBX(0x06, 0x08, rate, sizeof(rate));
}

void enqueueSample(const SampleRecord &sample) {
  if (relayQueue != nullptr && xQueueSend(relayQueue, &sample, 0) == pdTRUE) return;
  portENTER_CRITICAL(&stateMux);
  ++droppedSamples;
  portEXIT_CRITICAL(&stateMux);
}

uint64_t sourceTimestampNS(uint64_t deviceUS) {
  portENTER_CRITICAL(&stateMux);
  const int64_t offset = epochOffsetUS;
  portEXIT_CRITICAL(&stateMux);
  const int64_t totalUS = offset + static_cast<int64_t>(deviceUS);
  if (totalUS < 1577836800000000LL) {
    return 0;
  }
  return static_cast<uint64_t>(totalUS) * 1000ULL;
}

bool syncClockFromHttpDate() {
  NetworkClientSecure tls;
  tls.setCACert(RELAY_CA_PEM);
  tls.setHandshakeTimeout(10);
  HTTPClient http;
  http.setConnectTimeout(6000);
  http.setTimeout(8000);
  if (!http.begin(tls, "https://cpg44.nivaspms.com/health")) {
    return false;
  }
  const char *headerKeys[] = {"Date"};
  http.collectHeaders(headerKeys, 1);
  const uint32_t tStartMS = millis();
  const int code = http.GET();
  const uint32_t rttMS = millis() - tStartMS;
  if (code > 0 && http.hasHeader("Date")) {
    String dateStr = http.header("Date");
    struct tm tmTime{};
    if (strptime(dateStr.c_str(), "%a, %d %b %Y %H:%M:%S GMT", &tmTime) != nullptr) {
      time_t t = mktime(&tmTime);
      if (t >= 1577836800) {
        // Compensate for half-RTT network flight time
        const uint32_t usec = (rttMS / 2) * 1000;
        timeval tv = { .tv_sec = t, .tv_usec = static_cast<suseconds_t>(usec) };
        settimeofday(&tv, nullptr);
        const int64_t unixUS = static_cast<int64_t>(tv.tv_sec) * 1000000LL + usec;
        const int64_t nextOffset = unixUS - static_cast<int64_t>(deviceTimeUS());
        portENTER_CRITICAL(&stateMux);
        epochOffsetUS = nextOffset;
        clockReady = true;
        portEXIT_CRITICAL(&stateMux);
        Serial.printf("Clock synchronized via HTTP Date: %ld\n", static_cast<long>(t));
        http.end();
        tls.stop();
        return true;
      }
    }
  }
  http.end();
  tls.stop();
  return false;
}

bool refreshClockAnchor() {
  timeval tv{};
  gettimeofday(&tv, nullptr);
  if (tv.tv_sec >= 1577836800) {
    const int64_t unixUS = static_cast<int64_t>(tv.tv_sec) * 1000000LL + tv.tv_usec;
    const int64_t nextOffset = unixUS - static_cast<int64_t>(deviceTimeUS());
    portENTER_CRITICAL(&stateMux);
    epochOffsetUS = nextOffset;
    clockReady = true;
    portEXIT_CRITICAL(&stateMux);
    return true;
  }
  return syncClockFromHttpDate();
}

bool isClockReady() {
  portENTER_CRITICAL(&stateMux);
  const bool ready = clockReady;
  portEXIT_CRITICAL(&stateMux);
  return ready;
}

const char *sampleTypeName(SampleType type) {
  switch (type) {
    case SampleType::Imu: return "imu";
    case SampleType::Gps: return "gps";
    case SampleType::Status: return "status";
    case SampleType::Ecg: return "ecg";
  }
  return "unknown";
}

void addCommonFields(JsonObject event, const SampleRecord &sample) {
  const char *kind = sampleTypeName(sample.type);
  char eventId[160];
  snprintf(
    eventId, sizeof(eventId), "%s:%d:%08lx:%s:%llu",
    MATCH_ID, PLAYER_ID, static_cast<unsigned long>(bootId), kind,
    static_cast<unsigned long long>(sample.sequence)
  );
  event["event_id"] = eventId;
  event["match_id"] = MATCH_ID;
  event["player_id"] = PLAYER_ID;
  event["source"] = "wearable";
  event["source_seq"] = sample.sequence;
  event["source_timestamp_ns"] = sourceTimestampNS(sample.deviceUS);
  event["device_boot_id"] = bootId;
  event["sample_type"] = kind;

  JsonObject clock = event["clock"].to<JsonObject>();
  clock["valid"] = true;
  clock["method"] = "sntp_esp_timer_anchor";

  JsonObject tags = event["tags"].to<JsonObject>();
  tags["session_id"] = MATCH_ID;
  char deviceId[48];
  snprintf(deviceId, sizeof(deviceId), "wearable-%d", PLAYER_ID);
  tags["device_id"] = deviceId;
}

void addPayload(JsonObject event, const SampleRecord &sample) {
  JsonObject payload = event["payload"].to<JsonObject>();
  payload["device_us"] = sample.deviceUS;

  if (sample.type == SampleType::Imu) {
    JsonArray accel = payload["a"].to<JsonArray>();
    JsonArray gyro = payload["g"].to<JsonArray>();
    for (int i = 0; i < 3; ++i) {
      accel.add(sample.accel[i]);
      gyro.add(sample.gyro[i]);
    }
    payload["temp_c"] = sample.temperatureC;
    if (sample.magValid) {
      JsonArray m = payload["m"].to<JsonArray>();
      for (int i = 0; i < 3; ++i) m.add(sample.mag[i]);
    }
    return;
  }

  if (sample.type == SampleType::Ecg) {
    payload["fs"] = ECG_HZ;
    payload["lead_off"] = sample.leadOff;
    JsonArray v = payload["v"].to<JsonArray>();
    for (uint8_t i = 0; i < sample.ecgCount; ++i) v.add(sample.ecg[i]);
    return;
  }

  if (sample.type == SampleType::Gps) {
    payload["rx"] = sample.gpsRx;
    payload["fix"] = sample.gpsFix;
    payload["sat"] = sample.satellites;
    payload["chars"] = sample.gpsChars;
    if (sample.gpsFix) {
      payload["lat"] = sample.latitude;
      payload["lon"] = sample.longitude;
      if (isfinite(sample.speedMps)) payload["speed_mps"] = sample.speedMps;
      if (isfinite(sample.courseDeg)) payload["course_deg"] = sample.courseDeg;
      if (isfinite(sample.altitudeM)) payload["alt_m"] = sample.altitudeM;
      if (isfinite(sample.hdop)) payload["hdop"] = sample.hdop;
    }
    return;
  }

  payload["rssi_dbm"] = sample.rssiDbm;
  payload["heap"] = sample.freeHeap;
  payload["mpu6050"] = mpuOK;
  payload["gps_rx"] = sample.gpsReceiving;
  payload["imu_rate_hz"] = 1000000 / IMU_PERIOD_US;
  payload["imu_accel_range_g"] = 16;
  payload["imu_gyro_range_dps"] = 2000;
  payload["imu_dlpf_hz"] = 21;
  payload["gps_rate_hz"] = sample.gpsRateHz;
  payload["ecg"] = ECG_ENABLED;
  payload["ecg_rate_hz"] = sample.ecgRateHz;
  payload["magnetometer"] = magChip != MagChip::None;
  if (isfinite(sample.batteryV)) payload["battery_v"] = sample.batteryV;
  if (isfinite(sample.batteryPct)) payload["battery_pct"] = sample.batteryPct;
  payload["barometer"] = baroFound;
  payload["clock_anchor_age_ms"] = isClockReady() ? millis() - lastClockAnchorMS : 0;
  portENTER_CRITICAL(&stateMux);
  payload["dropped_samples"] = droppedSamples;
  portEXIT_CRITICAL(&stateMux);
  payload["queued_samples"] = uxQueueMessagesWaiting(relayQueue) + relayBatchCount;
}

static NetworkClientSecure tlsClient;
static HTTPClient httpClient;
static bool tlsInitialized = false;

bool postRelayBatch() {
  if (!relayBatchCount || !isClockReady() || WiFi.status() != WL_CONNECTED) return false;

  JsonDocument document;
  JsonArray events = document.to<JsonArray>();
  for (size_t i = 0; i < relayBatchCount; ++i) {
    JsonObject event = events.add<JsonObject>();
    addCommonFields(event, relayBatch[i]);
    addPayload(event, relayBatch[i]);
  }

  String body;
  body.reserve(measureJson(document) + 1);
  serializeJson(document, body);

  if (!tlsInitialized) {
    tlsClient.setCACert(RELAY_CA_PEM);
    tlsClient.setHandshakeTimeout(10);
    httpClient.setConnectTimeout(5000);
    httpClient.setTimeout(5000);
    httpClient.setReuse(true);
    tlsInitialized = true;
  }

  if (!httpClient.connected()) {
    if (!httpClient.begin(tlsClient, RELAY_ENDPOINT)) {
      Serial.println("Relay HTTPS setup failed");
      return false;
    }
  }

  httpClient.addHeader("Content-Type", "application/json");
  httpClient.addHeader("Accept", "application/json");
  httpClient.addHeader("X-Auth", RELAY_TOKEN);
  httpClient.addHeader("Connection", "keep-alive");
  const int code = httpClient.POST(body);
  String responseDetail = httpClient.getString();

  if (code >= 200 && code < 300) {
    Serial.printf("Relay accepted %u samples; local queue %u\n",
                  static_cast<unsigned>(relayBatchCount),
                  static_cast<unsigned>(uxQueueMessagesWaiting(relayQueue)));
    relayBatchCount = 0;
    return true;
  }
  Serial.printf("Relay POST failed: HTTP %d; response: %s; queued samples: %u\n",
                code, responseDetail.c_str(), static_cast<unsigned>(relayBatchCount));

  if (code >= 400 && code < 500 && code != 429) {
    portENTER_CRITICAL(&stateMux);
    droppedSamples += relayBatchCount;
    portEXIT_CRITICAL(&stateMux);
    relayBatchCount = 0;
  }
  return false;
}

void relayTask(void *) {
  uint32_t retryMS = 500;

  for (;;) {
    if (WiFi.status() != WL_CONNECTED) {
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    if (!isClockReady() || millis() - lastClockAnchorMS >= 60000) {
      if (refreshClockAnchor()) {
        lastClockAnchorMS = millis();
      } else {
        vTaskDelay(pdMS_TO_TICKS(200));
        continue;
      }
    }

    // Compact only expired samples. An ordinary HTTP round trip can queue more
    // than 16 observations; dropping those leaves holes in the ECG trace.
    size_t retained = 0;
    const uint64_t nowUS = deviceTimeUS();
    for (size_t i = 0; i < relayBatchCount; ++i) {
      if (nowUS - relayBatch[i].deviceUS <= RELAY_MAX_SAMPLE_AGE_US) {
        relayBatch[retained++] = relayBatch[i];
      } else {
        portENTER_CRITICAL(&stateMux);
        ++droppedSamples;
        portEXIT_CRITICAL(&stateMux);
      }
    }
    relayBatchCount = retained;

    const uint32_t batchStartedMS = millis();
    while (relayBatchCount < RELAY_BATCH_CAPACITY) {
      const uint32_t elapsed = millis() - batchStartedMS;
      if (elapsed >= RELAY_BATCH_WINDOW_MS) break;
      // Drain an existing backlog immediately. Coalesce only an almost-empty
      // queue; batching must not add another full window after every POST.
      const TickType_t wait = relayBatchCount >= 4 ? 0 : pdMS_TO_TICKS(RELAY_BATCH_WINDOW_MS - elapsed);
      if (xQueueReceive(relayQueue, &relayBatch[relayBatchCount], wait) != pdTRUE) break;
      if (deviceTimeUS() - relayBatch[relayBatchCount].deviceUS <= RELAY_MAX_SAMPLE_AGE_US) {
        ++relayBatchCount;
      } else {
        portENTER_CRITICAL(&stateMux);
        ++droppedSamples;
        portEXIT_CRITICAL(&stateMux);
      }
    }
    if (!relayBatchCount) continue;

    if (postRelayBatch()) {
      retryMS = 500;
    } else {
      vTaskDelay(pdMS_TO_TICKS(retryMS));
      retryMS = min<uint32_t>(retryMS * 2, 15000);
    }
  }
}

// ---- GY-87 magnetometer ----
// The HMC5883L sits on the MPU6050's auxiliary bus; bypass mode connects it to
// the main I2C bus. Many GY-87 boards carry the QMC5883L clone at 0x0D instead.
bool writeReg8(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readBytes(uint8_t addr, uint8_t reg, uint8_t *out, uint8_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, n) != n) return false;
  for (uint8_t i = 0; i < n; ++i) out[i] = Wire.read();
  return true;
}

void setupMagnetometer() {
  writeReg8(0x68, 0x6A, 0x00);   // USER_CTRL: I2C master off
  writeReg8(0x68, 0x37, 0x02);   // INT_PIN_CFG: I2C bypass on
  uint8_t id[3] = {0};
  if (readBytes(0x1E, 0x0A, id, 3) && id[0] == 'H' && id[1] == '4' && id[2] == '3') {
    writeReg8(0x1E, 0x00, 0x18);   // 1 sample average, 75 Hz
    writeReg8(0x1E, 0x01, 0x20);   // +-1.3 G, 1090 LSB/G
    writeReg8(0x1E, 0x02, 0x00);   // continuous
    magChip = MagChip::HMC5883L;
  } else if (probeI2C(0x0D)) {
    writeReg8(0x0D, 0x0B, 0x01);   // set/reset period
    writeReg8(0x0D, 0x09, 0x1D);   // continuous, 200 Hz, +-8 G, OSR 512
    magChip = MagChip::QMC5883L;
  }
}

bool readMagnetometer(float out[3]) {
  uint8_t b[6];
  if (magChip == MagChip::HMC5883L && readBytes(0x1E, 0x03, b, 6)) {
    // register order is X, Z, Y, big endian; 1090 LSB per gauss, 100 uT per gauss
    const int16_t x = (b[0] << 8) | b[1], z = (b[2] << 8) | b[3], y = (b[4] << 8) | b[5];
    out[0] = x / 10.90f; out[1] = y / 10.90f; out[2] = z / 10.90f;
    return x != -4096 && y != -4096 && z != -4096;     // -4096 marks an overflow
  }
  if (magChip == MagChip::QMC5883L && readBytes(0x0D, 0x00, b, 6)) {
    const int16_t x = (b[1] << 8) | b[0], y = (b[3] << 8) | b[2], z = (b[5] << 8) | b[4];
    out[0] = x / 30.0f; out[1] = y / 30.0f; out[2] = z / 30.0f;   // 3000 LSB per gauss at +-8 G
    return true;
  }
  return false;
}

// ---- AD8232 ECG: its own task so Wi-Fi and I2C never delay a sample ----
void ecgTask(void *) {
  SampleRecord block{};
  TickType_t wake = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(1000 / ECG_HZ);
  for (;;) {
    vTaskDelayUntil(&wake, period);
    const uint64_t t = deviceTimeUS();
    if (block.ecgCount == 0) {
      block = SampleRecord{};
      block.type = SampleType::Ecg;
      block.deviceUS = t;
    }
    block.ecg[block.ecgCount++] = static_cast<uint16_t>(analogRead(ECG_OUT));
    block.leadOff |= digitalRead(ECG_LO_PLUS) == HIGH || digitalRead(ECG_LO_MINUS) == HIGH;
    if (block.ecgCount == ECG_BLOCK) {
      block.sequence = ecgSeq++;
      enqueueSample(block);
      ecgBlocks = ecgBlocks + 1;
      block.ecgCount = 0;
    }
  }
}

void serviceIMU() {
  if (!mpuOK) return;
  const uint64_t nowUS = deviceTimeUS();
  if (nowUS - lastIMUScheduleUS < IMU_PERIOD_US) return;

  lastIMUScheduleUS += IMU_PERIOD_US;
  if (nowUS - lastIMUScheduleUS > 5ULL * IMU_PERIOD_US) lastIMUScheduleUS = nowUS;

  sensors_event_t a, g, temp;
  const uint64_t t0 = deviceTimeUS();
  const bool ok = mpu.getEvent(&a, &g, &temp);
  const uint64_t t1 = deviceTimeUS();
  if (!ok) return;

  SampleRecord sample{};
  sample.type = SampleType::Imu;
  sample.sequence = imuSeq++;
  sample.deviceUS = t0 + (t1 - t0) / 2ULL;
  sample.accel[0] = a.acceleration.x;
  sample.accel[1] = a.acceleration.y;
  sample.accel[2] = a.acceleration.z;
  sample.gyro[0] = g.gyro.x;
  sample.gyro[1] = g.gyro.y;
  sample.gyro[2] = g.gyro.z;
  sample.temperatureC = temp.temperature;
  sample.magValid = magChip != MagChip::None && readMagnetometer(sample.mag);
  enqueueSample(sample);
}

bool gpsReceivingNow() {
  return lastGPSByteMS != 0 && millis() - lastGPSByteMS < 3000;
}

void serviceGPS() {
  while (gpsSerial.available()) {
    gps.encode(gpsSerial.read());
    lastGPSByteMS = millis();
  }

  // Send each new fix as it arrives (5 Hz); without a fix, one heartbeat a second.
  const uint32_t nowMS = millis();
  const bool fresh = gps.location.isUpdated();
  if (!fresh && nowMS - lastGPSPacketMS < 1000) return;
  lastGPSPacketMS = nowMS;
  if (fresh) ++gpsFixCount;

  SampleRecord sample{};
  sample.type = SampleType::Gps;
  sample.sequence = gpsSeq++;
  sample.deviceUS = deviceTimeUS();
  sample.gpsRx = gpsReceivingNow();
  sample.gpsFix = gps.location.isValid() && gps.location.age() < 3000;
  sample.satellites = gps.satellites.isValid() ? gps.satellites.value() : 0;
  sample.gpsChars = gps.charsProcessed();
  sample.speedMps = NAN;
  sample.courseDeg = NAN;
  sample.altitudeM = NAN;
  sample.hdop = NAN;

  if (sample.gpsFix) {
    const uint64_t ageUS = static_cast<uint64_t>(gps.location.age()) * 1000ULL;
    if (sample.deviceUS > ageUS) sample.deviceUS -= ageUS;
    sample.latitude = gps.location.lat();   // isUpdated() is cleared by this read
    sample.longitude = gps.location.lng();
    if (gps.speed.isValid() && gps.speed.age() < 3000) sample.speedMps = gps.speed.mps();
    if (gps.course.isValid() && gps.course.age() < 3000) sample.courseDeg = gps.course.deg();
    if (gps.altitude.isValid() && gps.altitude.age() < 5000) sample.altitudeM = gps.altitude.meters();
    if (gps.hdop.isValid()) sample.hdop = gps.hdop.hdop();
  }
  enqueueSample(sample);
}

void serviceStatus() {
  const uint32_t nowMS = millis();
  if (nowMS - lastStatusMS < 2000) return;
  lastStatusMS = nowMS;

  SampleRecord sample{};
  sample.type = SampleType::Status;
  sample.sequence = statusSeq++;
  sample.deviceUS = deviceTimeUS();
  sample.rssiDbm = WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : -127;
  sample.freeHeap = ESP.getFreeHeap();
  sample.gpsReceiving = gpsReceivingNow();
  sample.gpsRateHz = (gpsFixCount - gpsFixCountAtStatus) / 2.0f;
  gpsFixCountAtStatus = gpsFixCount;
  const uint32_t blocks = ecgBlocks;
  sample.ecgRateHz = (blocks - ecgBlocksAtStatus) * ECG_BLOCK / 2.0f;
  ecgBlocksAtStatus = blocks;
  readBattery(sample);
  enqueueSample(sample);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  bootId = esp_random();

  Serial.println();
  Serial.println("=== CPG44 ESP32-S3 VEST: ECG, IMU, GNSS ===");
  if (PLAYER_ID <= 0 || WIFI_SSID[0] == '\0' || strlen(RELAY_TOKEN) < 32 ||
      strlen(RELAY_CA_PEM) < 100 || strcmp(RELAY_ENDPOINT, "https://cpg44.nivaspms.com/api/v1/sensors/ingest") != 0) {
    Serial.println("Provisioning required. Flash player, Wi-Fi, relay token and relay CA from the dashboard.");
    while (true) delay(1000);
  }

  relayQueue = xQueueCreate(RELAY_QUEUE_CAPACITY, sizeof(SampleRecord));
  if (relayQueue == nullptr) {
    Serial.println("Could not allocate the bounded relay queue.");
    while (true) delay(1000);
  }

  bindI2C();
  Serial.print("GY-87 MPU6050 0x68: ");
  Serial.println(probeI2C(0x68) ? "FOUND" : "MISSING");
  baroFound = probeI2C(0x77);
  Serial.print("GY-87 BMP180  0x77: ");
  Serial.println(baroFound ? "FOUND" : "MISSING");

  mpuOK = mpu.begin(0x68, &Wire);
  bindI2C();
  if (mpuOK) {
    // Football kicks, tackles and landings exceed 8 g at the upper back, so use
    // the full +/-16 g and +/-2000 deg/s ranges to avoid clipped impacts.
    mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
    mpu.setGyroRange(MPU6050_RANGE_2000_DEG);
    // The sketch reads at 50 Hz (Nyquist 25 Hz). The on-chip DLPF must sit
    // below Nyquist or impact and vibration energy aliases into the stride
    // band; 21 Hz is the closest setting.
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    // Internal rate 1 kHz / (1 + 9) = 100 Hz with DLPF on; polled at 50 Hz.
    mpu.setSampleRateDivisor(9);
    Serial.println("MPU6050 ready");
    bindI2C();
    setupMagnetometer();
    Serial.println(magChip == MagChip::HMC5883L ? "HMC5883L magnetometer ready"
                   : magChip == MagChip::QMC5883L ? "QMC5883L magnetometer ready" : "No magnetometer found");
  } else {
    Serial.println("MPU6050 FAILED");
  }

  bindI2C();
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX, GPS_TX);
  delay(100);
  configureGPS();
  Serial.println("NEO-6M UART started, 5 Hz RMC+GGA requested");

  WiFi.mode(WIFI_STA);
  WiFi.persistent(false);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  sntp_set_sync_mode(SNTP_SYNC_MODE_IMMED);
  configTime(0, 0, "time.cloudflare.com", "pool.ntp.org", "time.google.com");

  analogReadResolution(12);
  analogSetPinAttenuation(BATTERY_ADC, ADC_11db);
  analogSetPinAttenuation(ECG_OUT, ADC_11db);
  if (ECG_ENABLED) {
    pinMode(ECG_LO_PLUS, INPUT_PULLDOWN);
    pinMode(ECG_LO_MINUS, INPUT_PULLDOWN);
    xTaskCreatePinnedToCore(ecgTask, "cpg44-ecg", 4096, nullptr, 3, nullptr, 1);
    Serial.println("AD8232 ECG sampling at 250 Hz");
  }

  lastIMUScheduleUS = deviceTimeUS();
  xTaskCreatePinnedToCore(relayTask, "cpg44-relay", 16384, nullptr, 1, nullptr, 0);

  Serial.print("Relay: ");
  Serial.println(RELAY_ENDPOINT);
  Serial.print("Match/player: ");
  Serial.print(MATCH_ID);
  Serial.print('/');
  Serial.println(PLAYER_ID);
}

void loop() {
  serviceIMU();
  serviceGPS();
  serviceStatus();
  delay(1);
}
