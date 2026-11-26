#include <Arduino.h>
#include <algorithm>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <NetworkClientSecure.h>
#include <ArduinoJson.h>
#include <sys/time.h>
#include <WiFiUdp.h>
#include "miniz.h"              // deflate in the ESP32-S3 ROM (tdefl_*)
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "driver/gpio.h"

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
enum GpsLine : uint8_t { LINE_UNKNOWN, LINE_SENDING, LINE_POWERED_SILENT, LINE_HELD_LOW, LINE_OPEN };

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

// Delivery is limited by round trips, not bytes: on a busy campus network one
// HTTPS POST through Cloudflare took 1 to 2 s. The vest makes about 56 events
// a second (motion 50 Hz, ECG 5 blocks of 50 samples, GNSS, status), so 96
// per POST topped out near 48 a second and data was lost. Batches of up to 288
// (about 10 KB once deflated) keep up even at 2 s per round trip; a relay that
// only takes 100 per POST is detected and sent 96 at a time. Samples wait up to
// 15 s and the queue (in PSRAM) holds about 18 s, so a slow patch delays data
// instead of losing it; every sample carries its own timestamp.
constexpr size_t RELAY_QUEUE_CAPACITY = 1024;
constexpr size_t RELAY_BATCH_CAPACITY = 288;
constexpr size_t RELAY_LEGACY_LIMIT = 96;
constexpr uint32_t RELAY_BATCH_WINDOW_MS = 25;
constexpr uint64_t RELAY_MAX_SAMPLE_AGE_US = 15000000ULL;

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
int8_t txPowerNow = 78;            // quarter-dBm units, as the Wi-Fi driver uses
uint8_t txStrongCount = 0;

bool mpuOK = false;
bool baroFound = false;     // BMP180 on the GY-87: detected for the build check, not streamed
enum class MagChip : uint8_t { None, HMC5883L, QMC5883L, QMC5883P };
MagChip magChip = MagChip::None;
float qmc5883pLsbPerUT = 10.0f;     // set from the range the chip reports
volatile uint32_t ecgBlocks = 0;
// Latest values for the serial self-test line, so the vest can be checked on
// the bench over USB without Wi-Fi or the app.
volatile uint16_t ecgBlockMin = 0;
volatile uint16_t ecgBlockMax = 0;
volatile bool ecgLeadOffNow = true;
float lastAccelMps2 = NAN;
bool lastMagValid = false;
float lastMagUT = NAN;
bool batterySensePresent = false;
uint32_t statusCount = 0;
char i2cFound[48] = "";
uint64_t imuSeqAtCheck = 0;
uint32_t loopMaxUS = 0;
uint64_t loopLastUS = 0;
volatile uint32_t relayAcceptedTotal = 0;
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
SampleRecord *relayBatch = nullptr;          // RELAY_BATCH_CAPACITY records, in PSRAM
size_t relayBatchCount = 0;
size_t relaySendLimit = RELAY_BATCH_CAPACITY;
uint32_t limitSinceMS = 0;
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
// Is the 100k/100k divider wired to GPIO1? Wired, the pin sits at half the
// cell voltage and the pad's ~45 kOhm pull-down only drags it to about half of
// that again; unwired, the pin collapses to almost 0 V. Without this test a
// floating pin could be reported as a made-up battery level. The pull is set on
// the pad directly so the ADC channel stays configured.
bool probeBatteryDivider() {
  const uint32_t open = analogReadMilliVolts(BATTERY_ADC);
  gpio_set_pull_mode(static_cast<gpio_num_t>(BATTERY_ADC), GPIO_PULLDOWN_ONLY);
  delay(15);                                           // 100 nF settles in ~3 ms
  const uint32_t pulled = analogReadMilliVolts(BATTERY_ADC);
  gpio_set_pull_mode(static_cast<gpio_num_t>(BATTERY_ADC), GPIO_FLOATING);
  delay(15);
  const float cell = 2.0f * open / 1000.0f;
  return cell > 2.8f && cell < 4.4f && pulled * 4 >= open;
}

void readBattery(SampleRecord &sample) {
  sample.batteryV = NAN;
  sample.batteryPct = NAN;
  if (!batterySensePresent) return;
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
  // CFG-NAV5: pedestrian platform model (3). It filters for slow, low-acceleration
  // movement like a player's and still allows 30 m/s; only the model is changed
  // (mask bit 0), every other navigation setting keeps the module default.
  uint8_t nav5[36] = {0};
  nav5[0] = 0x01;
  nav5[2] = 3;
  sendUBX(0x06, 0x24, nav5, sizeof(nav5));
}

void enqueueSample(const SampleRecord &sample) {
  // Until the clock is set a sample cannot be timestamped, so it is skipped
  // rather than queued and later counted as dropped; "dropped" then means data
  // that was lost, not the first seconds before Wi-Fi.
  portENTER_CRITICAL(&stateMux);
  const bool timed = clockReady;
  portEXIT_CRITICAL(&stateMux);
  if (!timed && sample.type != SampleType::Status) return;
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

// One NTP query over an ordinary UDP socket, from the relay task once Wi-Fi is
// up. The core's built-in SNTP client is not used: when its first DNS lookup
// failed before Wi-Fi connected, its retry ran DNS without the network-stack
// lock and the board rebooted (assert in udp_new_ip_type). The server's transmit
// time plus half the round trip gives the time at reception, to a few ms.
uint32_t lastNtpMS = 0;

bool syncClockNTP(const char *host) {
  IPAddress ip;
  if (!WiFi.hostByName(host, ip)) return false;
  WiFiUDP udp;
  if (!udp.begin(0)) return false;
  uint8_t pkt[48] = {0};
  pkt[0] = 0x23;                                       // leap 0, version 4, client
  const uint64_t t1 = deviceTimeUS();
  udp.beginPacket(ip, 123);
  udp.write(pkt, sizeof(pkt));
  udp.endPacket();
  while (deviceTimeUS() - t1 < 1500000ULL) {
    if (udp.parsePacket() >= 48) {
      const uint64_t t4 = deviceTimeUS();
      udp.read(pkt, sizeof(pkt));
      udp.stop();
      const uint32_t secs = (uint32_t)pkt[40] << 24 | (uint32_t)pkt[41] << 16 | (uint32_t)pkt[42] << 8 | pkt[43];
      const uint32_t frac = (uint32_t)pkt[44] << 24 | (uint32_t)pkt[45] << 16 | (uint32_t)pkt[46] << 8 | pkt[47];
      if (secs < 2208988800UL + 1577836800UL) return false;      // not a plausible time
      const int64_t unixUS = static_cast<int64_t>(secs - 2208988800UL) * 1000000LL
                             + static_cast<int64_t>((static_cast<uint64_t>(frac) * 1000000ULL) >> 32)
                             + static_cast<int64_t>(t4 - t1) / 2;
      timeval tv = {.tv_sec = static_cast<time_t>(unixUS / 1000000LL), .tv_usec = static_cast<suseconds_t>(unixUS % 1000000LL)};
      settimeofday(&tv, nullptr);
      portENTER_CRITICAL(&stateMux);
      epochOffsetUS = unixUS - static_cast<int64_t>(t4);
      clockReady = true;
      portEXIT_CRITICAL(&stateMux);
      Serial.printf("Clock set by NTP from %s, round trip %lu ms\n", host, static_cast<unsigned long>((t4 - t1) / 1000));
      return true;
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  udp.stop();
  return false;
}

bool refreshClockAnchor() {
  // NTP every 10 minutes keeps crystal drift (~20 ppm, 72 ms an hour) small;
  // between queries the system clock carries the anchor. If the network blocks
  // NTP, the relay's HTTP Date header is the fallback (1 s resolution).
  if (lastNtpMS == 0 || millis() - lastNtpMS >= 600000) {
    if (syncClockNTP("time.cloudflare.com") || syncClockNTP("pool.ntp.org")) {
      lastNtpMS = millis();
      return true;
    }
  }
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

// Only what the relay needs per sample. The relay derives its duplicate key
// from match, player, boot, type, timestamp and sequence, and adds the tags
// itself, so the event id, tags and clock method string are not sent: about a
// third of every motion sample's bytes.
void addCommonFields(JsonObject event, const SampleRecord &sample) {
  event["match_id"] = MATCH_ID;
  event["player_id"] = PLAYER_ID;
  event["source"] = "wearable";
  event["source_seq"] = sample.sequence;
  event["source_timestamp_ns"] = sourceTimestampNS(sample.deviceUS);
  event["device_boot_id"] = bootId;
  event["sample_type"] = sampleTypeName(sample.type);
  event["clock"]["valid"] = true;
}

// Values rounded to the sensors' own resolution, so JSON carries no noise digits.
static inline float roundTo(float x, float step) { return roundf(x / step) * step; }

void addPayload(JsonObject event, const SampleRecord &sample) {
  JsonObject payload = event["payload"].to<JsonObject>();
  payload["device_us"] = sample.deviceUS;

  if (sample.type == SampleType::Imu) {
    JsonArray accel = payload["a"].to<JsonArray>();
    JsonArray gyro = payload["g"].to<JsonArray>();
    for (int i = 0; i < 3; ++i) {
      accel.add(roundTo(sample.accel[i], 0.001f));     // MPU6050 at +-16 g: 0.0048 m/s2 per count
      gyro.add(roundTo(sample.gyro[i], 0.0001f));      // at +-2000 deg/s: 0.0011 rad/s per count
    }
    payload["temp_c"] = roundTo(sample.temperatureC, 0.1f);
    if (sample.magValid) {
      JsonArray m = payload["m"].to<JsonArray>();
      for (int i = 0; i < 3; ++i) m.add(roundTo(sample.mag[i], 0.1f));
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
  payload["chip_c"] = roundTo(temperatureRead(), 0.1f);
  payload["tx_dbm"] = txPowerNow / 4.0f;
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

// Batches are deflated (zlib) with the compressor in the chip's ROM: the JSON
// is repetitive and shrinks several times, which matters on a busy campus
// uplink where a plain 34 KB batch took 2 s. A relay that does not take
// compressed bodies answers 4xx; the vest then sends plain JSON from then on
// and resends the same batch, so nothing is lost while the relay is updated.
bool relayDeflate = true;
uint32_t plainSinceMS = 0;          // when compression was last refused; retried after 10 min
uint32_t lastAcceptedMS = 0;        // last batch the relay took, for the delivery watchdog
bool connectionReset = false;
bool deflateReported = false;
tdefl_compressor *deflater = nullptr;
uint8_t *deflateOut = nullptr;
constexpr size_t DEFLATE_OUT_CAP = 64 * 1024;

size_t deflateBody(const String &body) {
  if (!deflater) deflater = static_cast<tdefl_compressor *>(heap_caps_malloc(sizeof(tdefl_compressor), MALLOC_CAP_SPIRAM));
  if (!deflateOut) deflateOut = static_cast<uint8_t *>(heap_caps_malloc(DEFLATE_OUT_CAP, MALLOC_CAP_SPIRAM));
  if (!deflater || !deflateOut) return 0;
  size_t inLen = body.length(), outLen = DEFLATE_OUT_CAP;
  if (tdefl_init(deflater, nullptr, nullptr, TDEFL_WRITE_ZLIB_HEADER | TDEFL_GREEDY_PARSING_FLAG | 32) != TDEFL_STATUS_OKAY) return 0;
  if (tdefl_compress(deflater, body.c_str(), &inLen, deflateOut, &outLen, TDEFL_FINISH) != TDEFL_STATUS_DONE) return 0;
  return outLen;
}

// Remove the first n records of the batch (sent, or refused for good).
void dropFront(size_t n) {
  if (n >= relayBatchCount) { relayBatchCount = 0; return; }
  memmove(relayBatch, relayBatch + n, (relayBatchCount - n) * sizeof(SampleRecord));
  relayBatchCount -= n;
}

bool postRelayBatch() {
  if (!relayBatchCount || !isClockReady() || WiFi.status() != WL_CONNECTED) return false;

  const size_t n = min(relayBatchCount, relaySendLimit);
  JsonDocument document;
  JsonArray events = document.to<JsonArray>();
  for (size_t i = 0; i < n; ++i) {
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

  for (int attempt = 0; attempt < 2; ++attempt) {
    if (!httpClient.connected()) {
      if (!httpClient.begin(tlsClient, RELAY_ENDPOINT)) {
        Serial.println("Relay HTTPS setup failed");
        return false;
      }
    }
    const size_t packed = relayDeflate ? deflateBody(body) : 0;
    httpClient.addHeader("Content-Type", "application/json");
    httpClient.addHeader("Accept", "application/json");
    httpClient.addHeader("X-Auth", RELAY_TOKEN);
    httpClient.addHeader("Connection", "keep-alive");
    if (packed) httpClient.addHeader("Content-Encoding", "deflate");
    const int code = packed ? httpClient.POST(deflateOut, packed) : httpClient.POST(body);
    String responseDetail = httpClient.getString();

    if (code >= 200 && code < 300) {
      relayAcceptedTotal = relayAcceptedTotal + n;
      lastAcceptedMS = millis();
      connectionReset = false;
      if (packed && !deflateReported) {
        deflateReported = true;
        Serial.printf("Relay takes compressed batches: %u bytes sent as %u\n",
                      static_cast<unsigned>(body.length()), static_cast<unsigned>(packed));
      }
      Serial.printf("Relay accepted %u samples; local queue %u\n",
                    static_cast<unsigned>(n), static_cast<unsigned>(uxQueueMessagesWaiting(relayQueue)));
      dropFront(n);
      return true;
    }
    if (n > RELAY_LEGACY_LIMIT && (code == 413 || code == 422)) {
      relaySendLimit = RELAY_LEGACY_LIMIT;                // older relay: at most 100 per POST
      limitSinceMS = millis();
      Serial.printf("Relay refused %u samples in one batch; sending %u at a time\n",
                    static_cast<unsigned>(n), static_cast<unsigned>(RELAY_LEGACY_LIMIT));
      return postRelayBatch();                           // rebuilt with the smaller batch
    }
    if (packed && code >= 400 && code < 500 && code != 401 && code != 429) {
      relayDeflate = false;                                // older relay: plain JSON for now
      plainSinceMS = millis();
      Serial.println("Relay does not take compressed batches yet; sending plain JSON");
      continue;
    }
    Serial.printf("Relay POST failed: HTTP %d; response: %s; queued samples: %u\n",
                  code, responseDetail.c_str(), static_cast<unsigned>(relayBatchCount));
    if (code >= 400 && code < 500 && code != 429) {
      portENTER_CRITICAL(&stateMux);
      droppedSamples += n;
      portEXIT_CRITICAL(&stateMux);
      dropFront(n);
    }
    return false;
  }
  return false;
}

// When the saved network cannot be joined, list what the vest can actually
// see. The ESP32-S3 radio is 2.4 GHz only, so a 5 GHz-only network never
// appears here; this answers "is my Wi-Fi reachable?" from the vest itself.
char wifiSeen[160] = "";
uint32_t wifiDownSinceMS = 0;
uint32_t lastWifiScanMS = 0;

void scanWifiIfLost() {
  if (WiFi.status() == WL_CONNECTED) { wifiDownSinceMS = 0; return; }
  const uint32_t now = millis();
  if (wifiDownSinceMS == 0) wifiDownSinceMS = now;
  if (now - wifiDownSinceMS < 15000 || (lastWifiScanMS && now - lastWifiScanMS < 60000)) return;
  lastWifiScanMS = now;
  WiFi.disconnect(false, false);                          // a scan fails while a join is in progress
  vTaskDelay(pdMS_TO_TICKS(200));
  const int n = WiFi.scanNetworks(false, true);
  bool savedSeen = false;
  wifiSeen[0] = 0;
  for (int i = 0; i < n; ++i) {
    if (WiFi.SSID(i) == WIFI_SSID) savedSeen = true;
    char one[48];
    snprintf(one, sizeof(one), "%s%s (%d dBm, ch %d)", i ? ", " : "", WiFi.SSID(i).length() ? WiFi.SSID(i).c_str() : "hidden",
             static_cast<int>(WiFi.RSSI(i)), static_cast<int>(WiFi.channel(i)));
    if (strlen(wifiSeen) + strlen(one) < sizeof(wifiSeen) - 1) strcat(wifiSeen, one);
  }
  WiFi.scanDelete();
  Serial.printf("Wi-Fi: \"%s\" %s. 2.4 GHz networks in range: %s\n", WIFI_SSID,
                savedSeen ? "is in range but would not connect (password?)" : "not in range",
                n > 0 ? wifiSeen : n == 0 ? "none" : "scan failed");
  WiFi.begin(WIFI_SSID, WIFI_PASS);                       // scanning interrupts the join attempt
}

// Delivery watchdog. When the relay restarted for an update, the vest kept its
// Wi-Fi but never got a batch through again until it was power-cycled. Whatever
// the stuck layer is (a dead keep-alive socket, a half-closed TLS session), on
// Wi-Fi with nothing accepted for 90 s the HTTPS client is rebuilt from scratch,
// and after 5 minutes the vest restarts cleanly. A relay that once refused
// compression is offered it again after 10 minutes, so an upgraded relay is
// picked up without a reset.
void deliveryWatchdog() {
  const uint32_t now = millis();
  if (!relayDeflate && plainSinceMS && now - plainSinceMS > 600000) relayDeflate = true;
  if (relaySendLimit < RELAY_BATCH_CAPACITY && now - limitSinceMS > 600000) relaySendLimit = RELAY_BATCH_CAPACITY;
  if (lastAcceptedMS == 0 || !isClockReady()) return;
  const uint32_t silent = now - lastAcceptedMS;
  if (silent > 300000) {
    Serial.println("Relay: nothing accepted for 5 min; restarting");
    delay(100);
    ESP.restart();
  }
  if (silent > 90000 && !connectionReset) {
    connectionReset = true;
    Serial.println("Relay: nothing accepted for 90 s; reconnecting from scratch");
    httpClient.end();
    tlsClient.stop();
    tlsInitialized = false;
  }
}

void relayTask(void *) {
  uint32_t retryMS = 500;

  for (;;) {
    // With a backlog this loop never waits on its queue: it serialises,
    // compresses and posts back to back. One tick per pass lets core 0's idle
    // task run; without it the task watchdog fired after 5 s at 160 MHz.
    vTaskDelay(1);
    if (WiFi.status() != WL_CONNECTED) {
      scanWifiIfLost();
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }
    wifiDownSinceMS = 0;
    deliveryWatchdog();

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
  } else if (readBytes(0x2C, 0x00, id, 1) && id[0] == 0x80) {
    // QMC5883P, on recent GY-87 boards (chip ID 0x80 at 0x2C)
    // Order matters on the boards tested: the range write (0x0B) must come
    // before the mode write (0x0A), or it clears the mode. These parts keep
    // their default +-30 G range anyway, so the scale follows the range the
    // chip reports back.
    writeReg8(0x2C, 0x0B, 0x80);   // soft reset
    delay(10);
    writeReg8(0x2C, 0x29, 0x06);   // axis sign definition from QST's example
    writeReg8(0x2C, 0x0B, 0x08);   // ask for +-8 G, set and reset on
    writeReg8(0x2C, 0x0A, 0x0B);   // continuous, 100 Hz, OSR1 8, OSR2 1
    delay(10);
    uint8_t ctrl2 = 0;
    readBytes(0x2C, 0x0B, &ctrl2, 1);
    static const float LSB_PER_GAUSS[4] = {1000.0f, 2500.0f, 3750.0f, 15000.0f};   // 30, 12, 8, 2 G
    qmc5883pLsbPerUT = LSB_PER_GAUSS[(ctrl2 >> 2) & 0x03] / 100.0f;
    magChip = MagChip::QMC5883P;
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
  if (magChip == MagChip::QMC5883P && readBytes(0x2C, 0x01, b, 6)) {
    const int16_t x = (b[1] << 8) | b[0], y = (b[3] << 8) | b[2], z = (b[5] << 8) | b[4];
    out[0] = x / qmc5883pLsbPerUT; out[1] = y / qmc5883pLsbPerUT; out[2] = z / qmc5883pLsbPerUT;
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
      uint16_t lo = 4095, hi = 0;
      for (uint8_t i = 0; i < ECG_BLOCK; ++i) { lo = min(lo, block.ecg[i]); hi = max(hi, block.ecg[i]); }
      ecgBlockMin = lo;
      ecgBlockMax = hi;
      ecgLeadOffNow = block.leadOff;
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
  lastAccelMps2 = sqrtf(sample.accel[0] * sample.accel[0] + sample.accel[1] * sample.accel[1] + sample.accel[2] * sample.accel[2]);
  lastMagValid = sample.magValid;
  if (sample.magValid) lastMagUT = sqrtf(sample.mag[0] * sample.mag[0] + sample.mag[1] * sample.mag[1] + sample.mag[2] * sample.mag[2]);
  enqueueSample(sample);
}

bool gpsReceivingNow() {
  return lastGPSByteMS != 0 && millis() - lastGPSByteMS < 3000;
}

// The GPS link is found, not assumed. TX and RX are easy to swap on the bench,
// and a module whose EEPROM saved another speed talks at 38400 or 115200 baud.
// Until a sentence with a valid checksum arrives: no bytes at all means try the
// other pin (both are plain 3.3 V lines, so the swap is harmless); bytes that
// never form a sentence mean try the next baud rate.
const uint32_t GPS_BAUDS[] = {GPS_BAUD, 4800, 19200, 38400, 57600, 115200};
const uint8_t GPS_BAUD_COUNT = sizeof(GPS_BAUDS) / sizeof(GPS_BAUDS[0]);
uint8_t gpsBaudIndex = 0;
bool gpsPinsSwapped = false;
int gpsRxPin = GPS_RX, gpsTxPin = GPS_TX;      // where the GPS really is (see findGpsPin)
int gpsFoundElsewhere = -1;
uint32_t gpsReprobeMS = 0, gpsReprobeStartMS = 0;
bool gpsReprobing = false;
uint16_t gpsSeenProbes = 0, gpsProbes = 0;
bool gpsLocked = false;
uint32_t gpsAttemptMS = 0;
uint32_t gpsAttemptBytes = 0;
uint8_t gpsNotes = 0;

// What the GPS lines look like electrically, measured once at boot before the
// UART takes the pins. A UART alone cannot tell "wire not connected" from
// "module without power" from "powered but silent"; the pin's own pull-down
// and pull-up, plus counting edges for 1.1 s, can:
//   edges                 -> the module is sending (a baud or pin question)
//   high with pull-down   -> something holds the line high: module powered, silent
//   low with pull-up      -> something holds it low: module has GND but no power
//   follows the pull      -> nothing connected on this pin
GpsLine gpsLine16 = LINE_UNKNOWN, gpsLine17 = LINE_UNKNOWN;
volatile uint32_t gpsProbeEdges = 0, gpsProbeLastUS = 0, gpsProbeMinUS = UINT32_MAX;
void IRAM_ATTR gpsProbeEdge() {
  const uint32_t t = micros(), d = t - gpsProbeLastUS;
  gpsProbeLastUS = t;
  ++gpsProbeEdges;
  if (d > 4 && d < gpsProbeMinUS) gpsProbeMinUS = d;   // the shortest pulse is one bit
}
bool gpsPinKnown = false;          // the probe saw the module sending on this pin: never swap away
int8_t gpsMeasuredBaud = -1;       // index into GPS_BAUDS from the measured bit time

// The baud rate whose bit time is nearest the shortest pulse seen.
int8_t baudFromBitUS(uint32_t bitUS) {
  if (bitUS == UINT32_MAX || bitUS == 0) return -1;
  int8_t best = -1;
  float bestErr = 1e9f;
  for (uint8_t i = 0; i < GPS_BAUD_COUNT; ++i) {
    const float err = fabsf(logf((1e6f / GPS_BAUDS[i]) / bitUS));
    if (err < bestErr) { bestErr = err; best = i; }
  }
  return bestErr < 0.35f ? best : -1;              // within about 40 % of a standard rate
}

GpsLine probeGpsLine(int pin) {
  pinMode(pin, INPUT_PULLDOWN);
  delay(3);
  const bool highWithPullDown = digitalRead(pin);
  pinMode(pin, INPUT_PULLUP);
  delay(3);
  const bool highWithPullUp = digitalRead(pin);
  pinMode(pin, INPUT_PULLDOWN);
  gpsProbeEdges = 0;
  gpsProbeMinUS = UINT32_MAX;
  gpsProbeLastUS = micros();
  attachInterrupt(digitalPinToInterrupt(pin), gpsProbeEdge, CHANGE);
  delay(1100);                                   // NMEA comes in a burst once a second
  detachInterrupt(digitalPinToInterrupt(pin));
  const uint32_t edges = gpsProbeEdges;
  pinMode(pin, INPUT);
  if (edges > 40) {
    gpsMeasuredBaud = baudFromBitUS(gpsProbeMinUS);
    return LINE_SENDING;
  }
  if (highWithPullDown) return LINE_POWERED_SILENT;
  if (!highWithPullUp) return LINE_HELD_LOW;
  return LINE_OPEN;
}

// Free pins on an ESP32-S3 N16R8 board that a GPS TX wire could have landed
// on: not I2C (4, 5), ECG (2, 11, 12), battery (1), USB (19, 20), flash and
// PSRAM (26 to 37), strapping (0, 3, 45, 46) or the RGB LED (48).
const int GPS_SCAN_PINS[] = {6, 7, 8, 9, 10, 13, 14, 15, 18, 21, 38, 39, 40, 41, 42, 43, 44, 47};
volatile uint32_t gpsScanEdges[sizeof(GPS_SCAN_PINS) / sizeof(int)];
void IRAM_ATTR gpsScanEdge(void *arg) { ++gpsScanEdges[reinterpret_cast<uintptr_t>(arg)]; }

// Listen on every free pin at once for 1.2 s; the GPS sends a burst of text
// each second, so the pin with hundreds of edges is where its TX wire is.
int findGpsPin() {
  const size_t n = sizeof(GPS_SCAN_PINS) / sizeof(int);
  for (size_t i = 0; i < n; ++i) {
    gpsScanEdges[i] = 0;
    pinMode(GPS_SCAN_PINS[i], INPUT_PULLDOWN);
    attachInterruptArg(digitalPinToInterrupt(GPS_SCAN_PINS[i]), gpsScanEdge, reinterpret_cast<void *>(i), CHANGE);
  }
  delay(1200);
  int best = -1;
  uint32_t most = 40;
  for (size_t i = 0; i < n; ++i) {
    detachInterrupt(digitalPinToInterrupt(GPS_SCAN_PINS[i]));
    pinMode(GPS_SCAN_PINS[i], INPUT);
    if (gpsScanEdges[i] > most) { most = gpsScanEdges[i]; best = GPS_SCAN_PINS[i]; }
  }
  return best;
}

const char *gpsLineDiagnosis() {
  static char found[64];
  if (gpsSeenProbes > 0 && gpsSeenProbes < gpsProbes) return "seen at times: loose TX wire or contact";
  if (gpsFoundElsewhere >= 0) {
    snprintf(found, sizeof(found), "found on GPIO%d instead of GPIO16, no valid sentence yet", gpsFoundElsewhere);
    return found;
  }
  if (gpsLine16 == LINE_SENDING || gpsLine17 == LINE_SENDING) return "signal present but no valid sentence: odd baud rate";
  if (gpsLine16 == LINE_HELD_LOW || gpsLine17 == LINE_HELD_LOW) return "TX held low: GPS has GND but no power, check VCC";
  if (gpsLine16 == LINE_POWERED_SILENT || gpsLine17 == LINE_POWERED_SILENT) return "GPS powered but silent: module not sending";
  if (gpsLine16 == LINE_OPEN && gpsLine17 == LINE_OPEN) return "nothing connected on GPIO16 or GPIO17: check TX wire";
  return "nothing on GPIO16 or GPIO17: check VCC, GND, TX";
}

void gpsBegin() {
  gpsSerial.end();
  gpsSerial.begin(GPS_BAUDS[gpsBaudIndex], SERIAL_8N1, gpsPinsSwapped ? gpsTxPin : gpsRxPin, gpsPinsSwapped ? gpsRxPin : gpsTxPin);
  gpsAttemptMS = millis();
  gpsAttemptBytes = 0;
}

// While no sentence has arrived, look at both lines again every 20 s with the
// UART stopped (1.1 s, non-blocking). A loose TX wire or contact then recovers
// on its own, and "seen at times" tells the bench check what is wrong.

bool gpsReprobe() {                                // true while a probe is running
  const uint32_t now = millis();
  if (!gpsReprobing) {
    // every 6 s for the first 10 minutes (bench work, wiggle tests), then 20 s
    if (now - gpsReprobeMS < (now < 600000 ? 6000UL : 20000UL)) return false;
    gpsSerial.end();
    const int pins[2] = {gpsRxPin, gpsTxPin};
    for (uintptr_t i = 0; i < 2; ++i) {
      gpsScanEdges[i] = 0;
      pinMode(pins[i], INPUT_PULLDOWN);
      attachInterruptArg(digitalPinToInterrupt(pins[i]), gpsScanEdge, reinterpret_cast<void *>(i), CHANGE);
    }
    gpsReprobing = true;
    gpsReprobeStartMS = now;
    return true;
  }
  if (now - gpsReprobeStartMS < 1100) return true;
  const int pins[2] = {gpsRxPin, gpsTxPin};
  bool heldHigh[2];
  for (int i = 0; i < 2; ++i) {
    detachInterrupt(digitalPinToInterrupt(pins[i]));
    heldHigh[i] = digitalRead(pins[i]);           // still on the pull-down: high only if something drives it
    pinMode(pins[i], INPUT);
  }
  gpsReprobing = false;
  gpsReprobeMS = now;
  ++gpsProbes;
  const uint32_t e0 = gpsScanEdges[0], e1 = gpsScanEdges[1];
  // live line state for the self-test, so moving a wire shows up within seconds
  gpsLine16 = e0 > 40 ? LINE_SENDING : heldHigh[0] ? LINE_POWERED_SILENT : LINE_OPEN;
  gpsLine17 = e1 > 40 ? LINE_SENDING : heldHigh[1] ? LINE_POWERED_SILENT : LINE_OPEN;
  if (e0 > 40 || e1 > 40) {
    ++gpsSeenProbes;
    gpsPinsSwapped = e1 > e0;
    gpsPinKnown = true;
    gpsLine16 = gpsPinsSwapped ? LINE_OPEN : LINE_SENDING;
    gpsLine17 = gpsPinsSwapped ? LINE_SENDING : LINE_OPEN;
    Serial.printf("GPS: signal back on GPIO%d\n", gpsPinsSwapped ? gpsTxPin : gpsRxPin);
  }
  gpsBegin();
  return false;
}

void checkGPSWiring() {
  if (gpsLocked) return;
  if (gpsReprobe()) return;
  if (gps.passedChecksum() > 0) {
    gpsLocked = true;
    Serial.printf("GPS: sentences on GPIO%d at %lu baud%s\n", gpsPinsSwapped ? gpsTxPin : gpsRxPin,
                  static_cast<unsigned long>(GPS_BAUDS[gpsBaudIndex]),
                  gpsPinsSwapped ? " (TX and RX wires are swapped; working anyway)" : "");
    configureGPS();                                      // 5 Hz, now on the pin the module listens to
    return;
  }
  if (millis() - gpsAttemptMS < 3000) return;
  if (gpsAttemptBytes > 20 || gpsPinKnown) gpsBaudIndex = (gpsBaudIndex + 1) % GPS_BAUD_COUNT;
  else gpsPinsSwapped = !gpsPinsSwapped;
  gpsBegin();
  if (gpsNotes < 4) {
    ++gpsNotes;
    Serial.printf("GPS: nothing valid yet, trying GPIO%d at %lu baud\n", gpsPinsSwapped ? gpsTxPin : gpsRxPin,
                  static_cast<unsigned long>(GPS_BAUDS[gpsBaudIndex]));
  }
}

void serviceGPS() {
  checkGPSWiring();
  while (!gpsReprobing && gpsSerial.available()) {
    gps.encode(gpsSerial.read());
    ++gpsAttemptBytes;
    if (gpsLocked) lastGPSByteMS = millis();
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

// One line every 2 s on USB: every sensor's state in plain numbers.
void printSelfTest(const SampleRecord &status) {
  char gpsText[80];
  if (gpsReceivingNow()) {
    snprintf(gpsText, sizeof(gpsText), "%lu chars, %u sats, %s", static_cast<unsigned long>(gps.charsProcessed()),
             gps.satellites.isValid() ? static_cast<unsigned>(gps.satellites.value()) : 0U,
             gps.location.isValid() && gps.location.age() < 3000 ? "fix" : "no fix");
  } else {
    snprintf(gpsText, sizeof(gpsText), "%s", gpsAttemptBytes > 20 ? "bytes but no valid sentence yet" : gpsLineDiagnosis());
  }
  char battText[32];
  if (isfinite(status.batteryV)) snprintf(battText, sizeof(battText), "%.2f V %.0f %%", status.batteryV, status.batteryPct);
  else snprintf(battText, sizeof(battText), "0 V on GPIO1");
  const uint16_t lo = ecgBlockMin, hi = ecgBlockMax;
  const float imuHz = (imuSeq - imuSeqAtCheck) / 2.0f;
  imuSeqAtCheck = imuSeq;
  const unsigned long pauseMS = loopMaxUS / 1000;
  loopMaxUS = 0;
  Serial.printf("[check] i2c%s | imu %s | mag %s | gps %s | ecg %u-%u, electrodes %s | battery %s | wifi %s | relay sent %lu | chip %.0f C, tx %.1f dBm, %lu MHz\n",
                i2cFound[0] ? i2cFound : " none",
                mpuOK && isfinite(lastAccelMps2) ? (String(lastAccelMps2, 1) + " m/s2 at " + String(imuHz, 0) + " Hz"
                                                    + (pauseMS > 15 ? ", loop paused " + String(pauseMS) + " ms" : "")).c_str() : "missing",
                magChip == MagChip::None ? "none" : (lastMagValid ? (String(magChip == MagChip::HMC5883L ? "HMC5883L " : magChip == MagChip::QMC5883L ? "QMC5883L " : "QMC5883P ")
                                 + String(lastMagUT, 0) + " uT").c_str() : "no data"),
                gpsText, lo, hi, ecgLeadOffNow ? "off" : "on", battText,
                WiFi.status() == WL_CONNECTED ? (String(WIFI_SSID) + " " + String(WiFi.RSSI()) + " dBm").c_str() : "not connected",
                static_cast<unsigned long>(relayAcceptedTotal), temperatureRead(), txPowerNow / 4.0f,
                static_cast<unsigned long>(getCpuFrequencyMhz()));
}

void serviceStatus() {
  const uint32_t nowMS = millis();
  if (nowMS - lastStatusMS < 2000) return;
  lastStatusMS = nowMS;
  tuneTxPower(lastAcceptedMS != 0 && nowMS - lastAcceptedMS > 10000);

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
  if (++statusCount % 30 == 0) batterySensePresent = probeBatteryDivider();   // re-check once a minute
  readBattery(sample);
  enqueueSample(sample);
  printSelfTest(sample);
}

void setup() {
  setCpuFrequencyMhz(160);        // 50 Hz IMU, 250 Hz ECG and compression need far less than 240 MHz
  Serial.begin(115200);
  // USB serial must never block: with the cable plugged in but no program
  // reading the port, each print waited for a reader and the sensor loop
  // stalled (motion fell from 50 to about 20 samples a second).
  Serial.setTxTimeoutMs(0);
  delay(1000);
  bootId = esp_random();

  Serial.println();
  Serial.println("=== CPG44 ESP32-S3 VEST: ECG, IMU, GNSS ===");
  if (PLAYER_ID <= 0 || WIFI_SSID[0] == '\0' || strlen(RELAY_TOKEN) < 32 ||
      strlen(RELAY_CA_PEM) < 100 || strcmp(RELAY_ENDPOINT, "https://cpg44.nivaspms.com/api/v1/sensors/ingest") != 0) {
    Serial.println("Provisioning required. Flash player, Wi-Fi, relay token and relay CA from the dashboard.");
    while (true) delay(1000);
  }

  relayBatch = static_cast<SampleRecord *>(heap_caps_malloc(sizeof(SampleRecord) * RELAY_BATCH_CAPACITY, MALLOC_CAP_SPIRAM));
  relayQueue = xQueueCreateWithCaps(RELAY_QUEUE_CAPACITY, sizeof(SampleRecord), MALLOC_CAP_SPIRAM);
  if (relayQueue == nullptr || relayBatch == nullptr) {
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
    for (uint8_t a = 1; a < 127; ++a) {
      if (probeI2C(a) && strlen(i2cFound) < sizeof(i2cFound) - 6) {
        char one[6];
        snprintf(one, sizeof(one), " %02X", a);
        strcat(i2cFound, one);
      }
    }
    Serial.printf("I2C devices:%s\n", i2cFound);
    Serial.println(magChip == MagChip::HMC5883L ? "HMC5883L magnetometer ready"
                   : magChip == MagChip::QMC5883L ? "QMC5883L magnetometer ready"
                   : magChip == MagChip::QMC5883P ? "QMC5883P magnetometer ready" : "No magnetometer found");
  } else {
    Serial.println("MPU6050 FAILED");
  }

  bindI2C();
  gpsLine16 = probeGpsLine(GPS_RX);
  gpsLine17 = probeGpsLine(GPS_TX);
  Serial.printf("GPS lines: GPIO16 %s, GPIO17 %s\n",
                gpsLine16 == LINE_SENDING ? "sending" : gpsLine16 == LINE_POWERED_SILENT ? "held high, silent"
                : gpsLine16 == LINE_HELD_LOW ? "held low" : "open",
                gpsLine17 == LINE_SENDING ? "sending" : gpsLine17 == LINE_POWERED_SILENT ? "held high, silent"
                : gpsLine17 == LINE_HELD_LOW ? "held low" : "open");
  if (gpsLine17 == LINE_SENDING && gpsLine16 != LINE_SENDING) gpsPinsSwapped = true;   // start on the pin that talks
  gpsPinKnown = gpsLine16 == LINE_SENDING || gpsLine17 == LINE_SENDING;
  if (gpsLine16 != LINE_SENDING && gpsLine17 != LINE_SENDING) {
    const int pin = findGpsPin();
    if (pin >= 0) {
      gpsFoundElsewhere = gpsRxPin = pin;
      gpsPinKnown = true;
      Serial.printf("GPS: its TX wire is on GPIO%d, not GPIO16; reading it there (move the wire when convenient)\n", pin);
    } else {
      Serial.println("GPS: no signal on any free pin: the module has no power, or its TX is not connected");
    }
  }
  if (gpsMeasuredBaud >= 0) {
    gpsBaudIndex = gpsMeasuredBaud;
    Serial.printf("GPS: measured %lu baud from its shortest pulse\n", static_cast<unsigned long>(GPS_BAUDS[gpsBaudIndex]));
  }
  gpsBegin();
  delay(100);
  configureGPS();
  Serial.println("NEO-6M UART started, 5 Hz RMC+GGA requested");

  WiFi.mode(WIFI_STA);
  WiFi.persistent(false);
  // The radio stays awake (modem sleep would delay every relay reply by up to a
  // beacon interval); heat is cut instead by a lower CPU clock and by sending
  // no louder than the link needs (tuneTxPower).
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  analogReadResolution(12);
  analogSetPinAttenuation(BATTERY_ADC, ADC_11db);
  analogSetPinAttenuation(ECG_OUT, ADC_11db);
  batterySensePresent = probeBatteryDivider();
  Serial.println(batterySensePresent ? "Battery divider on GPIO1: found" : "Battery divider on GPIO1: not wired, battery not reported");
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

// Transmit power follows the access point's signal, with hysteresis: a strong
// link (-55 dBm or better) needs 11 dBm, not the default 19.5 dBm, and the
// power amplifier is the hottest part of the module while sending. Power
// goes back up at once when the signal weakens or deliveries start failing.
void tuneTxPower(bool deliveriesFailing) {
  if (WiFi.status() != WL_CONNECTED) return;
  const int rssi = WiFi.RSSI();
  wifi_power_t want = deliveriesFailing || rssi <= -75 ? WIFI_POWER_19_5dBm
                    : rssi <= -67 ? WIFI_POWER_17dBm
                    : rssi <= -55 ? WIFI_POWER_15dBm : WIFI_POWER_11dBm;
  if (static_cast<int8_t>(want) < txPowerNow) {      // lower only after 3 steady readings
    if (++txStrongCount < 3) return;
  }
  txStrongCount = 0;
  if (static_cast<int8_t>(want) != txPowerNow) {
    WiFi.setTxPower(want);
    txPowerNow = static_cast<int8_t>(want);
  }
}

void loop() {
  const uint64_t nowUS = deviceTimeUS();
  if (loopLastUS) loopMaxUS = max<uint32_t>(loopMaxUS, static_cast<uint32_t>(nowUS - loopLastUS));
  loopLastUS = nowUS;
  serviceIMU();
  serviceGPS();
  serviceStatus();
  delay(1);
}
