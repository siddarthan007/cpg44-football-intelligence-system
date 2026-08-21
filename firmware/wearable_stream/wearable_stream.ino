#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <NetworkClientSecure.h>
#include <ArduinoJson.h>
#include <sys/time.h>
#include "esp_timer.h"
#include "esp_system.h"

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "MAX30105.h"
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

constexpr uint32_t I2C_HZ = 100000;       // stable on the tested board
constexpr uint32_t IMU_PERIOD_US = 10000; // 100 Hz
constexpr uint32_t PPG_PERIOD_US = 40000; // 25 Hz after FIFO averaging
constexpr float PPG_HZ = 25.0f;

// MAX30102 settings used by the tested raw stream. The host still calculates
// HR and SpO2, where motion and optical quality can be checked together.
constexpr byte PPG_LED_AMPLITUDE = 60;
constexpr byte PPG_SAMPLE_AVERAGE = 4;
constexpr byte PPG_LED_MODE = 2;
constexpr int PPG_ADC_SAMPLE_RATE = 100;
constexpr int PPG_PULSE_WIDTH_US = 411;
constexpr int PPG_ADC_RANGE_NA = 8192;

constexpr size_t RELAY_QUEUE_CAPACITY = 512;
constexpr size_t RELAY_BATCH_CAPACITY = 96;
constexpr uint32_t RELAY_BATCH_WINDOW_MS = 500;

enum class SampleType : uint8_t { Imu, Ppg, Gps, Status };

struct SampleRecord {
  SampleType type;
  uint64_t sequence;
  uint64_t deviceUS;
  float accel[3];
  float gyro[3];
  float temperatureC;
  uint32_t red;
  uint32_t ir;
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
};

Adafruit_MPU6050 mpu;
MAX30105 max30102;
TinyGPSPlus gps;
HardwareSerial gpsSerial(1);

bool mpuOK = false;
bool maxOK = false;
uint32_t bootId = 0;
uint64_t imuSeq = 0;
uint64_t ppgSeq = 0;
uint64_t gpsSeq = 0;
uint64_t statusSeq = 0;
uint64_t lastIMUScheduleUS = 0;
uint32_t lastGPSPacketMS = 0;
uint32_t lastGPSByteMS = 0;
uint32_t lastStatusMS = 0;

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

// SparkFun MAX30105::begin() calls Wire.begin() without the board pins. Rebind
// after library setup so the sensors remain on the tested ESP32-S3 pins 4/5.
void bindI2C() {
  Wire.begin(I2C_SDA, I2C_SCL, I2C_HZ);
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
  return static_cast<uint64_t>(offset + static_cast<int64_t>(deviceUS)) * 1000ULL;
}

bool refreshClockAnchor() {
  timeval tv{};
  gettimeofday(&tv, nullptr);
  if (tv.tv_sec < 1577836800) return false;
  const int64_t unixUS = static_cast<int64_t>(tv.tv_sec) * 1000000LL + tv.tv_usec;
  const int64_t nextOffset = unixUS - static_cast<int64_t>(deviceTimeUS());
  portENTER_CRITICAL(&stateMux);
  epochOffsetUS = nextOffset;
  clockReady = true;
  portEXIT_CRITICAL(&stateMux);
  return true;
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
    case SampleType::Ppg: return "ppg";
    case SampleType::Gps: return "gps";
    case SampleType::Status: return "status";
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
    return;
  }

  if (sample.type == SampleType::Ppg) {
    payload["red"] = sample.red;
    payload["ir"] = sample.ir;
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
  payload["max30102"] = maxOK;
  payload["gps_rx"] = sample.gpsReceiving;
  portENTER_CRITICAL(&stateMux);
  payload["dropped_samples"] = droppedSamples;
  portEXIT_CRITICAL(&stateMux);
  payload["queued_samples"] = uxQueueMessagesWaiting(relayQueue) + relayBatchCount;
}

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

  NetworkClientSecure tls;
  tls.setCACert(RELAY_CA_PEM);
  tls.setHandshakeTimeout(10);
  HTTPClient http;
  http.setConnectTimeout(6000);
  http.setTimeout(8000);
  if (!http.begin(tls, RELAY_ENDPOINT)) {
    Serial.println("Relay HTTPS setup failed");
    return false;
  }
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Accept", "application/json");
  http.addHeader("X-Auth", RELAY_TOKEN);
  const int code = http.POST(body);
  http.end();
  tls.stop();

  if (code >= 200 && code < 300) {
    Serial.printf("Relay accepted %u samples; local queue %u\n",
                  static_cast<unsigned>(relayBatchCount),
                  static_cast<unsigned>(uxQueueMessagesWaiting(relayQueue)));
    relayBatchCount = 0;
    return true;
  }
  Serial.printf("Relay POST failed: HTTP %d; retaining %u samples\n",
                code, static_cast<unsigned>(relayBatchCount));
  return false;
}

void relayTask(void *) {
  uint32_t retryMS = 500;
  uint32_t lastClockRefreshMS = 0;

  for (;;) {
    if (WiFi.status() != WL_CONNECTED) {
      WiFi.reconnect();
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    if (!isClockReady() || millis() - lastClockRefreshMS >= 60000) {
      if (refreshClockAnchor()) {
        lastClockRefreshMS = millis();
      } else {
        vTaskDelay(pdMS_TO_TICKS(200));
        continue;
      }
    }

    if (!relayBatchCount) {
      if (xQueueReceive(relayQueue, &relayBatch[0], pdMS_TO_TICKS(500)) != pdTRUE) continue;
      relayBatchCount = 1;
    }

    const uint32_t batchStartedMS = millis();
    while (relayBatchCount < RELAY_BATCH_CAPACITY) {
      const uint32_t elapsed = millis() - batchStartedMS;
      if (elapsed >= RELAY_BATCH_WINDOW_MS) break;
      const TickType_t wait = pdMS_TO_TICKS(RELAY_BATCH_WINDOW_MS - elapsed);
      if (xQueueReceive(relayQueue, &relayBatch[relayBatchCount], wait) == pdTRUE) {
        ++relayBatchCount;
      }
    }

    if (postRelayBatch()) {
      retryMS = 500;
    } else {
      vTaskDelay(pdMS_TO_TICKS(retryMS));
      retryMS = min<uint32_t>(retryMS * 2, 15000);
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
  enqueueSample(sample);
}

void servicePPG() {
  if (!maxOK) return;
  max30102.check();

  constexpr int MAX_FIFO_BATCH = 32;
  uint32_t redBatch[MAX_FIFO_BATCH];
  uint32_t irBatch[MAX_FIFO_BATCH];
  int count = 0;
  while (max30102.available() && count < MAX_FIFO_BATCH) {
    redBatch[count] = max30102.getFIFORed();
    irBatch[count] = max30102.getFIFOIR();
    max30102.nextSample();
    ++count;
  }
  if (!count) return;

  const uint64_t newestUS = deviceTimeUS();
  for (int i = 0; i < count; ++i) {
    const uint64_t ageUS = static_cast<uint64_t>(count - 1 - i) * PPG_PERIOD_US;
    SampleRecord sample{};
    sample.type = SampleType::Ppg;
    sample.sequence = ppgSeq++;
    sample.deviceUS = newestUS > ageUS ? newestUS - ageUS : newestUS;
    sample.red = redBatch[i];
    sample.ir = irBatch[i];
    enqueueSample(sample);
  }
}

bool gpsReceivingNow() {
  return lastGPSByteMS != 0 && millis() - lastGPSByteMS < 3000;
}

void serviceGPS() {
  while (gpsSerial.available()) {
    gps.encode(gpsSerial.read());
    lastGPSByteMS = millis();
  }

  const uint32_t nowMS = millis();
  if (nowMS - lastGPSPacketMS < 1000) return;
  lastGPSPacketMS = nowMS;

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
    sample.latitude = gps.location.lat();
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
  enqueueSample(sample);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  bootId = esp_random();

  Serial.println();
  Serial.println("=== CPG44 ESP32-S3 RELAY WEARABLE V3 ===");
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
  Serial.print("MAX30102 0x57: ");
  Serial.println(probeI2C(0x57) ? "FOUND" : "MISSING");
  Serial.print("MPU6050  0x68: ");
  Serial.println(probeI2C(0x68) ? "FOUND" : "MISSING");

  maxOK = max30102.begin(Wire, I2C_SPEED_STANDARD);
  bindI2C();
  if (maxOK) {
    Serial.print("MAX30102 Part ID: 0x");
    Serial.println(max30102.readPartID(), HEX);
    max30102.setup(PPG_LED_AMPLITUDE, PPG_SAMPLE_AVERAGE, PPG_LED_MODE,
                   PPG_ADC_SAMPLE_RATE, PPG_PULSE_WIDTH_US, PPG_ADC_RANGE_NA);
    bindI2C();
    max30102.setPulseAmplitudeRed(PPG_LED_AMPLITUDE);
    max30102.setPulseAmplitudeIR(PPG_LED_AMPLITUDE);
    max30102.clearFIFO();
    Serial.println("MAX30102 ready");
  } else {
    Serial.println("MAX30102 FAILED");
  }

  bindI2C();
  mpuOK = mpu.begin(0x68, &Wire);
  bindI2C();
  if (mpuOK) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_1000_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
    mpu.setSampleRateDivisor(9);
    Serial.println("MPU6050 ready");
  } else {
    Serial.println("MPU6050 FAILED");
  }

  bindI2C();
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX, GPS_TX);
  Serial.println("NEO-6M UART started");

  WiFi.mode(WIFI_STA);
  WiFi.persistent(false);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  configTime(0, 0, "time.cloudflare.com", "pool.ntp.org", "time.google.com");

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
  servicePPG();
  serviceIMU();
  serviceGPS();
  serviceStatus();
  delay(1);
}
