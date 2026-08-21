/*
 * soccer_wearable.ino  —  CPG44 player wearable (ESP32-S3 WROOM N16R8)
 * ---------------------------------------------------------------------
 * Streams IMU + HR/SpO2 + GPS as JSON to the analytics endpoint over Wi-Fi.
 *
 * Sensors (all 3.3 V):
 *   GY-87 10-DoF  : MPU6050 (accel+gyro @0x68), HMC5883L (mag, behind MPU aux
 *                   bus @0x1E), BMP180 (baro @0x77) — shared I2C bus.
 *   MAX30102      : HR + SpO2 (@0x57) — shared I2C bus.
 *   NEO-6M GPS    : UART @9600.
 *
 * Endpoint (see soccer_analytics.sensors.server): HTTP POST JSON to /ingest.
 * Runs at ~10 Hz. HR/SpO2 update slower (a few seconds) and are motion-sensitive
 * — treat them as resting/recovery indicators, not per-second match HR.
 *
 * Libraries (Arduino Library Manager):
 *   - "MPU6050" by Electronic Cats  (or Adafruit MPU6050)
 *   - "Adafruit BMP085 Library"     (BMP180-compatible)
 *   - "SparkFun MAX3010x Pulse and Proximity Sensor Library"
 *   - "TinyGPSPlus" by Mikal Hart
 *   - "ArduinoJson" by Benoit Blanchon
 *   Board: "ESP32S3 Dev Module" (esp32 core >= 3.x).
 *
 * NOTE: HMC5883L on the GY-87 sits behind the MPU6050 — enable I2C bypass
 * (INT_PIN_CFG bit) to reach it. Many "GY-87" boards actually carry a QMC5883L;
 * if magnetometer init fails, the pipeline does not require mag, so it is
 * optional and skipped automatically.
 */

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <MPU6050.h>
#include <Adafruit_BMP085.h>
#include <TinyGPSPlus.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"

// Provisioning values are generated locally and ignored by git. This sketch is
// inert when built without a generated header; credentials never live in source.
#if __has_include("generated_config.h")
#include "generated_config.h"
#else
inline constexpr int PLAYER_ID = 0;
inline constexpr char WIFI_SSID[] = "";
inline constexpr char WIFI_PASS[] = "";
inline constexpr char ENDPOINT[] = "";
inline constexpr char AUTH_TOKEN[] = "";
inline constexpr uint32_t SEND_MS = 100;
#endif

// I2C + UART pins (adjust to your board's silkscreen)
const int SDA_PIN = 8, SCL_PIN = 9;
const int GPS_RX  = 18, GPS_TX = 17;                // ESP RX<-GPS TX, ESP TX->GPS RX
// -----------------------------------------------------------

MPU6050          mpu;
Adafruit_BMP085  bmp;
MAX30105         max3102;
TinyGPSPlus      gps;
HardwareSerial   GPSserial(1);

bool hasBmp = false, hasMax = false;
uint32_t lastSend = 0;

// MAX30102 SpO2 working buffers
uint32_t irBuf[100], redBuf[100];
int32_t  spo2Val; int8_t spo2Valid; int32_t hrVal; int8_t hrValid;
int      sampleCount = 0;
float    lastHR = -1, lastSpO2 = -1;

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) { delay(300); Serial.print('.'); }
  Serial.println(WiFi.status() == WL_CONNECTED ? " ok " + WiFi.localIP().toString() : " FAILED");
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  mpu.initialize();
  mpu.setI2CBypassEnabled(true);                    // expose HMC5883L on main bus
  Serial.println(mpu.testConnection() ? "MPU6050 ok" : "MPU6050 FAIL");

  hasBmp = bmp.begin();
  Serial.println(hasBmp ? "BMP180 ok" : "BMP180 skip");

  hasMax = max3102.begin(Wire, I2C_SPEED_FAST);
  if (hasMax) {
    // 50 Hz, 411us pulse, moderate LED current — SpO2-friendly config
    max3102.setup(60, 4, 2, 50, 411, 4096);
    Serial.println("MAX30102 ok");
  } else Serial.println("MAX30102 skip");

  GPSserial.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

  connectWifi();
  configTime(0, 0, "pool.ntp.org");                 // epoch time for sample sync
}

// pull HR/SpO2 in the background; returns quickly
void serviceMax() {
  if (!hasMax) return;
  while (max3102.available()) {
    redBuf[sampleCount] = max3102.getRed();
    irBuf[sampleCount]  = max3102.getIR();
    max3102.nextSample();
    if (++sampleCount >= 100) {
      maxim_heart_rate_and_oxygen_saturation(irBuf, 100, redBuf,
        &spo2Val, &spo2Valid, &hrVal, &hrValid);
      if (hrValid && hrVal > 30 && hrVal < 230) lastHR = hrVal;
      if (spo2Valid && spo2Val > 70) lastSpO2 = spo2Val;
      sampleCount = 0;
    }
  }
  max3102.check();
}

void serviceGps() {
  while (GPSserial.available()) gps.encode(GPSserial.read());
}

bool timeSynced() {
  time_t now; time(&now);
  return now > 100000;                 // NTP has set a real epoch
}

double epochNow() {
  time_t now; time(&now);
  return (double)now + (millis() % 1000) / 1000.0;
}

// ---- sample batching (efficiency + loss recovery) -------------------------
// Samples accumulate into a bounded JSON array and are POSTed together every
// FLUSH_MS. One HTTP request per ~5 samples cuts radio on-time ~5× (battery),
// and a failed POST simply keeps the batch for the next flush — no data lost on
// brief Wi-Fi dropouts (bounded: oldest dropped beyond MAX_BATCH).
const uint32_t FLUSH_MS  = 500;
const size_t   MAX_BATCH = 25;
DynamicJsonDocument batch(8192);
JsonArray batchArr = batch.to<JsonArray>();
uint32_t lastFlush = 0;

void buildSample(JsonObject j) {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  j["player_id"] = PLAYER_ID;
  // Only send `t` when NTP is synced — an unsynced clock (millis-based) would
  // NEVER match the server's wall-clock window and the data would be silently
  // ignored. Without `t`, the server stamps arrival time (good to ~100 ms).
  if (timeSynced()) j["t"] = epochNow();
  // MPU6050 default ±2 g full-scale → /16384 = g
  JsonArray a = j.createNestedArray("accel");
  a.add(ax / 16384.0); a.add(ay / 16384.0); a.add(az / 16384.0);
  j["accel_unit"] = "g";
  JsonArray gyr = j.createNestedArray("gyro");
  gyr.add(gx / 131.0); gyr.add(gy / 131.0); gyr.add(gz / 131.0);   // deg/s (±250)
  if (hasBmp) j["altitude"] = bmp.readAltitude();
  if (lastHR   > 0) j["hr"]   = lastHR;
  if (lastSpO2 > 0) j["spo2"] = lastSpO2;
  if (gps.location.isValid()) {
    JsonArray g = j.createNestedArray("gps");
    g.add(gps.location.lat()); g.add(gps.location.lng());
  }
  j["source"] = "esp32";
}

void flushBatch() {
  if (batchArr.size() == 0) return;
  if (WiFi.status() != WL_CONNECTED) { connectWifi(); return; }  // keep batch
  String body; serializeJson(batchArr, body);
  HTTPClient http;
  http.begin(ENDPOINT);
  http.addHeader("Content-Type", "application/json");
  if (strlen(AUTH_TOKEN)) http.addHeader("X-Auth", AUTH_TOKEN);
  http.setTimeout(400);
  int code = http.POST(body);
  http.end();
  if (code >= 200 && code < 300) {
    batch.clear(); batchArr = batch.to<JsonArray>();   // sent → reset
  } else {
    Serial.printf("POST err %d (batch kept, %u samples)\n", code, batchArr.size());
  }
}

void loop() {
  serviceMax();
  serviceGps();
  if (millis() - lastSend >= SEND_MS) {
    lastSend = millis();
    if (batchArr.size() >= MAX_BATCH) batchArr.remove(0);   // bounded: drop oldest
    buildSample(batchArr.createNestedObject());
  }
  if (millis() - lastFlush >= FLUSH_MS) {
    lastFlush = millis();
    flushBatch();
  }
}
