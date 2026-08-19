#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <Network.h>
#include "esp_timer.h"
#include "esp_system.h"

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "MAX30105.h"
#include <TinyGPSPlus.h>

// ============================================================================
// USER CONFIG  — working wearable_stream board (I2C 4/5, GPS 16/17)
// ============================================================================
const char *WIFI_SSID = "EACCESS-M1";
const char *WIFI_PASS = "hostelnet";

#define I2C_SDA 4
#define I2C_SCL 5
#define GPS_RX 16                     // ESP32 RX <- NEO-6M TX
#define GPS_TX 17                     // optional
#define GPS_BAUD 9600

constexpr uint16_t STREAM_PORT = 9000;
constexpr uint32_t I2C_HZ = 100000;   // proven stable on this device

// MPU6050: DLPF enabled + divisor 9 => ~100 Hz register update.
constexpr uint32_t IMU_PERIOD_US = 10000;

// MAX30102 configuration.
// 100 ADC samples/s with FIFO averaging=4 => 25 FIFO samples/s.
constexpr uint32_t PPG_PERIOD_US = 40000;
constexpr float PPG_HZ = 25.0f;

// IR mean around 245k sat near the 18-bit ceiling (262143) at 4096 nA.
// 8192 nA preserves waveform headroom for SpO2.
constexpr byte PPG_LED_AMPLITUDE = 60;
constexpr byte PPG_SAMPLE_AVERAGE = 4;
constexpr byte PPG_LED_MODE = 2;      // Red + IR
constexpr int PPG_ADC_SAMPLE_RATE = 100;
constexpr int PPG_PULSE_WIDTH_US = 411;
constexpr int PPG_ADC_RANGE_NA = 8192;

Adafruit_MPU6050 mpu;
MAX30105 max30102;
TinyGPSPlus gps;
HardwareSerial gpsSerial(1);

NetworkServer streamServer(STREAM_PORT);
NetworkClient streamClient;

bool mpuOK = false;
bool maxOK = false;

uint32_t bootId = 0;
uint64_t imuSeq = 0;
uint64_t ppgSeq = 0;
uint64_t gpsSeq = 0;
uint64_t lastIMUScheduleUS = 0;
uint32_t lastGPSPacketMS = 0;
uint32_t lastGPSByteMS = 0;
uint32_t lastStatusMS = 0;

static inline uint64_t deviceTimeUS() {
  return (uint64_t)esp_timer_get_time();
}

bool clientConnected() {
  return streamClient && streamClient.connected();
}

void sendLine(const char *line) {
  if (!clientConnected()) return;
  streamClient.print(line);
  streamClient.print('\n');
}

bool probeI2C(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

void jsonDouble(char *out, size_t n, double value, int digits) {
  if (!isfinite(value)) snprintf(out, n, "null");
  else snprintf(out, n, "%.*f", digits, value);
}

// SparkFun MAX30105::begin() calls Wire.begin() with no pins. On ESP32-S3 that
// often switches the bus to the default SDA/SCL (8/9) and the MAX30102 on 4/5
// goes silent. Always re-bind to the board pins after library init.
void bindI2C() {
  Wire.begin(I2C_SDA, I2C_SCL, I2C_HZ);
}

char commandBuffer[128];
size_t commandLength = 0;

void handleCommand(const char *line, uint64_t t2US) {
  unsigned long syncId = 0;
  unsigned long long t1NS = 0;

  if (sscanf(line, "SYNC,%lu,%llu", &syncId, &t1NS) == 2) {
    const uint64_t t3US = deviceTimeUS();
    char reply[160];
    snprintf(
      reply, sizeof(reply),
      "SYNC_REPLY,%lu,%llu,%llu,%llu",
      syncId, t1NS,
      (unsigned long long)t2US,
      (unsigned long long)t3US
    );
    sendLine(reply);
  }
}

void serviceClientInput() {
  if (!clientConnected()) return;

  while (streamClient.available()) {
    const char c = (char)streamClient.read();

    if (c == '\n') {
      const uint64_t t2US = deviceTimeUS();
      commandBuffer[commandLength] = '\0';
      if (commandLength) handleCommand(commandBuffer, t2US);
      commandLength = 0;
      continue;
    }

    if (c == '\r') continue;

    if (commandLength < sizeof(commandBuffer) - 1) {
      commandBuffer[commandLength++] = c;
    } else {
      commandLength = 0;
    }
  }
}

void sendHello() {
  IPAddress ip = WiFi.localIP();
  char packet[440];

  snprintf(
    packet, sizeof(packet),
    "{\"t\":\"hello\",\"boot_id\":%lu,\"device\":\"esp32-s3-wearable\"," \
    "\"ip\":\"%u.%u.%u.%u\",\"imu_hz\":100,\"ppg_hz\":%.1f," \
    "\"ppg_timestamp\":\"fifo-reconstructed\",\"ppg_led\":%u," \
    "\"ppg_average\":%u,\"ppg_adc_sample_rate\":%d,\"ppg_adc_range_na\":%d," \
    "\"mpu6050\":%d,\"max30102\":%d,\"gps_baud\":%u}",
    (unsigned long)bootId,
    ip[0], ip[1], ip[2], ip[3],
    PPG_HZ,
    PPG_LED_AMPLITUDE,
    PPG_SAMPLE_AVERAGE,
    PPG_ADC_SAMPLE_RATE,
    PPG_ADC_RANGE_NA,
    mpuOK, maxOK, GPS_BAUD
  );

  sendLine(packet);
}

void serviceClient() {
  if (clientConnected()) return;

  if (streamClient) streamClient.stop();

  NetworkClient candidate = streamServer.accept();
  if (!candidate) return;

  streamClient = candidate;
  streamClient.setNoDelay(true);
  commandLength = 0;

  Serial.print("Subscriber connected: ");
  Serial.println(streamClient.remoteIP());
  sendHello();
}

void serviceIMU() {
  if (!mpuOK) return;

  const uint64_t nowUS = deviceTimeUS();
  if (nowUS - lastIMUScheduleUS < IMU_PERIOD_US) return;

  lastIMUScheduleUS += IMU_PERIOD_US;
  if (nowUS - lastIMUScheduleUS > 5ULL * IMU_PERIOD_US) {
    lastIMUScheduleUS = nowUS;
  }

  sensors_event_t a, g, temp;
  const uint64_t t0 = deviceTimeUS();
  const bool ok = mpu.getEvent(&a, &g, &temp);
  const uint64_t t1 = deviceTimeUS();
  if (!ok) return;

  const uint64_t sampleUS = t0 + (t1 - t0) / 2ULL;

  char packet[380];
  snprintf(
    packet, sizeof(packet),
    "{\"t\":\"imu\",\"boot_id\":%lu,\"seq\":%llu,\"device_us\":%llu," \
    "\"a\":[%.6f,%.6f,%.6f],\"g\":[%.6f,%.6f,%.6f],\"temp_c\":%.3f}",
    (unsigned long)bootId,
    (unsigned long long)imuSeq++,
    (unsigned long long)sampleUS,
    a.acceleration.x, a.acceleration.y, a.acceleration.z,
    g.gyro.x, g.gyro.y, g.gyro.z,
    temp.temperature
  );

  sendLine(packet);
}

void servicePPG() {
  if (!maxOK) return;

  max30102.check();

  constexpr int MAX_BATCH = 32;
  uint32_t redBatch[MAX_BATCH];
  uint32_t irBatch[MAX_BATCH];
  int n = 0;

  while (max30102.available() && n < MAX_BATCH) {
    redBatch[n] = max30102.getFIFORed();
    irBatch[n] = max30102.getFIFOIR();
    max30102.nextSample();
    ++n;
  }

  if (!n) return;

  const uint64_t newestUS = deviceTimeUS();

  for (int i = 0; i < n; ++i) {
    const uint64_t ageUS = (uint64_t)(n - 1 - i) * PPG_PERIOD_US;
    const uint64_t sampleUS = newestUS > ageUS ? newestUS - ageUS : newestUS;

    char packet[240];
    snprintf(
      packet, sizeof(packet),
      "{\"t\":\"ppg\",\"boot_id\":%lu,\"seq\":%llu,\"device_us\":%llu," \
      "\"red\":%lu,\"ir\":%lu}",
      (unsigned long)bootId,
      (unsigned long long)ppgSeq++,
      (unsigned long long)sampleUS,
      (unsigned long)redBatch[i],
      (unsigned long)irBatch[i]
    );
    sendLine(packet);
  }
}

bool gpsReceiving() {
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

  const uint64_t nowUS = deviceTimeUS();
  const bool fix = gps.location.isValid() && gps.location.age() < 3000;

  char packet[680];

  if (!fix) {
    snprintf(
      packet, sizeof(packet),
      "{\"t\":\"gps\",\"boot_id\":%lu,\"seq\":%llu,\"device_us\":%llu," \
      "\"rx\":%d,\"fix\":0,\"chars\":%lu,\"sat\":%lu}",
      (unsigned long)bootId,
      (unsigned long long)gpsSeq++,
      (unsigned long long)nowUS,
      gpsReceiving(),
      (unsigned long)gps.charsProcessed(),
      (unsigned long)(gps.satellites.isValid() ? gps.satellites.value() : 0)
    );
    sendLine(packet);
    return;
  }

  uint64_t fixUS = nowUS;
  const uint32_t locationAgeMS = gps.location.age();
  const uint64_t ageUS = (uint64_t)locationAgeMS * 1000ULL;
  if (fixUS > ageUS) fixUS -= ageUS;

  char speed[24], course[24], altitude[24], hdop[24];
  jsonDouble(speed, sizeof(speed),
             gps.speed.isValid() && gps.speed.age() < 3000 ? gps.speed.mps() : NAN, 4);
  jsonDouble(course, sizeof(course),
             gps.course.isValid() && gps.course.age() < 3000 ? gps.course.deg() : NAN, 3);
  jsonDouble(altitude, sizeof(altitude),
             gps.altitude.isValid() && gps.altitude.age() < 5000 ? gps.altitude.meters() : NAN, 3);
  jsonDouble(hdop, sizeof(hdop), gps.hdop.isValid() ? gps.hdop.hdop() : NAN, 3);

  char utc[48] = "null";
  if (gps.date.isValid() && gps.time.isValid()) {
    snprintf(
      utc, sizeof(utc),
      "\"%04d-%02d-%02dT%02d:%02d:%02d.%02dZ\"",
      gps.date.year(), gps.date.month(), gps.date.day(),
      gps.time.hour(), gps.time.minute(), gps.time.second(), gps.time.centisecond()
    );
  }

  snprintf(
    packet, sizeof(packet),
    "{\"t\":\"gps\",\"boot_id\":%lu,\"seq\":%llu,\"device_us\":%llu," \
    "\"rx\":1,\"fix\":1,\"lat\":%.7f,\"lon\":%.7f,\"speed_mps\":%s," \
    "\"course_deg\":%s,\"alt_m\":%s,\"sat\":%lu,\"hdop\":%s," \
    "\"location_age_ms\":%lu,\"utc\":%s,\"chars\":%lu}",
    (unsigned long)bootId,
    (unsigned long long)gpsSeq++,
    (unsigned long long)fixUS,
    gps.location.lat(), gps.location.lng(),
    speed, course, altitude,
    (unsigned long)(gps.satellites.isValid() ? gps.satellites.value() : 0),
    hdop,
    (unsigned long)locationAgeMS,
    utc,
    (unsigned long)gps.charsProcessed()
  );

  sendLine(packet);
}

void serviceStatus() {
  const uint32_t nowMS = millis();
  if (nowMS - lastStatusMS < 2000) return;
  lastStatusMS = nowMS;

  if (!clientConnected() || WiFi.status() != WL_CONNECTED) return;

  IPAddress ip = WiFi.localIP();
  char packet[360];
  snprintf(
    packet, sizeof(packet),
    "{\"t\":\"status\",\"boot_id\":%lu,\"device_us\":%llu," \
    "\"ip\":\"%u.%u.%u.%u\",\"rssi_dbm\":%d,\"heap\":%u," \
    "\"mpu6050\":%d,\"max30102\":%d,\"gps_rx\":%d}",
    (unsigned long)bootId,
    (unsigned long long)deviceTimeUS(),
    ip[0], ip[1], ip[2], ip[3],
    WiFi.RSSI(), ESP.getFreeHeap(),
    mpuOK, maxOK, gpsReceiving()
  );
  sendLine(packet);
}

void serviceWiFi() {
  static uint32_t lastRetryMS = 0;
  if (WiFi.status() == WL_CONNECTED) return;

  const uint32_t nowMS = millis();
  if (nowMS - lastRetryMS < 5000) return;
  lastRetryMS = nowMS;
  WiFi.reconnect();
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  bootId = esp_random();

  Serial.println();
  Serial.println("=== ESP32-S3 RAW SENSOR STREAM V2 ===");

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

    max30102.setup(
      PPG_LED_AMPLITUDE,
      PPG_SAMPLE_AVERAGE,
      PPG_LED_MODE,
      PPG_ADC_SAMPLE_RATE,
      PPG_PULSE_WIDTH_US,
      PPG_ADC_RANGE_NA
    );
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

  Serial.print("Connecting Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    servicePPG();
    serviceGPS();
    Serial.print('.');
    delay(50);
  }
  Serial.println();

  IPAddress ip = WiFi.localIP();
  Serial.print("ESP32 IP: ");
  Serial.println(ip);

  streamServer.setNoDelay(true);
  streamServer.begin();

  Serial.print("RAW STREAM: tcp://");
  Serial.print(ip);
  Serial.print(':');
  Serial.println(STREAM_PORT);

  lastIMUScheduleUS = deviceTimeUS();
}

void loop() {
  serviceClient();
  serviceClientInput();
  servicePPG();
  serviceIMU();
  serviceGPS();
  serviceStatus();
  serviceWiFi();
  delay(1);
}
