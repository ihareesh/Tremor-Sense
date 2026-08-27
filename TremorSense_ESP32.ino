#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "driver/i2s.h"
#include "voice_data.h"

// ================= WIFI =================
const char* AP_SSID = "ESP32_VOWEL";
const char* AP_PASS = "12345678";

IPAddress broadcastIP(192, 168, 4, 255);

// ================= UDP PORTS =================
const int AUDIO_PORT = 5005;
const int IMU_SEND_PORT = 1234;
const int PWM_CMD_PORT = 1235;

WiFiUDP udpAudio;
WiFiUDP udpIMU;
WiFiUDP udpCMD;

// ================= MIC =================
#define MIC_PIN 4
#define AUDIO_PACKET_SIZE 512

int16_t audio_buf[AUDIO_PACKET_SIZE];
int audio_index = 0;

// ================= MPU6050 =================
#define MPU_ADDR 0x68
#define SDA_PIN 21
#define SCL_PIN 15

float ax, ay, az, gx, gy, gz, temp;

// ================= MOTOR =================
#define MOTOR_PIN 5
const int pwmFreq = 200;
const int pwmResolution = 8;
int currentPWM = 0;

// ================= SPEAKER =================
#define I2S_BCLK    16
#define I2S_LRC     17
#define I2S_DOUT    18
#define SAMPLE_RATE 16000
#define I2S_PORT    I2S_NUM_0
#define VOLUME_GAIN 6.5f

// ================= RGB LED COMMON ANODE =================
#define LED_R 13
#define LED_G 12
#define LED_B 11

// ================= LED FUNCTIONS =================
void setColor(bool r, bool g, bool b) {
  digitalWrite(LED_R, r ? LOW : HIGH);
  digitalWrite(LED_G, g ? LOW : HIGH);
  digitalWrite(LED_B, b ? LOW : HIGH);
}

void allOff()   { setColor(false, false, false); }
void ledRed()   { setColor(true, false, false); }
void ledGreen() { setColor(false, true, false); }
void ledBlue()  { setColor(false, false, true); }
void ledWhite() { setColor(true, true, true); }

void ledBlink(int times, int on_ms, int off_ms) {
  for (int i = 0; i < times; i++) {
    ledRed();
    delay(on_ms);
    allOff();
    delay(off_ms);
  }
}

// ================= MOTOR PWM =================
void applyPWM(int pwmValue) {
  pwmValue = constrain(pwmValue, 0, 255);
  currentPWM = pwmValue;
  ledcWrite(MOTOR_PIN, currentPWM);

  Serial.print("Applied PWM: ");
  Serial.println(currentPWM);
}

// ================= SPEAKER SETUP =================
void setupSpeaker() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK,
    .ws_io_num = I2S_LRC,
    .data_out_num = I2S_DOUT,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_zero_dma_buffer(I2S_PORT);
}

// ================= VOICE PLAYBACK =================
void playVoice(const int16_t* data, int length) {
  int16_t stereo[2];
  size_t bytes_written;

  for (int i = 0; i < length; i++) {
    int32_t boosted = (int32_t)(data[i] * VOLUME_GAIN);

    if (boosted > 32767) boosted = 32767;
    if (boosted < -32768) boosted = -32768;

    stereo[0] = (int16_t)boosted;
    stereo[1] = (int16_t)boosted;

    i2s_write(I2S_PORT, stereo, sizeof(stereo), &bytes_written, portMAX_DELAY);
  }
}

// ================= MODE FUNCTIONS =================
void modeWake() {
  Serial.println(">> O — RED — System Awake");
  ledRed();
  playVoice(voice_wake, voice_wake_len);
}

void modeDrink() {
  Serial.println(">> A — BLUE — Drink Mode");
  ledBlue();
  playVoice(voice_drink, voice_drink_len);
}

void modeEat() {
  Serial.println(">> E — GREEN — Eat Mode");
  ledGreen();
  playVoice(voice_eat, voice_eat_len);
}

void modePill() {
  Serial.println(">> I — WHITE — Pill Mode");
  ledWhite();
  playVoice(voice_pill, voice_pill_len);
}

void modeStop() {
  Serial.println(">> U — BLINK RED — System Stopped");
  ledBlink(3, 200, 200);
  allOff();
  applyPWM(0);
  playVoice(voice_stop, voice_stop_len);
}

void handleCommand(char command) {
  command = toupper(command);

  switch (command) {
    case 'O': modeWake();  break;
    case 'A': modeDrink(); break;
    case 'E': modeEat();   break;
    case 'I': modePill();  break;
    case 'U': modeStop();  break;
    default: break;
  }
}

// ================= SETUP MPU =================
void setupMPU() {
  Wire.begin(SDA_PIN, SCL_PIN);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission();

  Serial.println("MPU6050 initialized");
}

// ================= READ + SEND IMU =================
void sendIMUData() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  if (Wire.available() < 14) return;

  int16_t AcX = Wire.read() << 8 | Wire.read();
  int16_t AcY = Wire.read() << 8 | Wire.read();
  int16_t AcZ = Wire.read() << 8 | Wire.read();
  int16_t Tmp = Wire.read() << 8 | Wire.read();
  int16_t GyX = Wire.read() << 8 | Wire.read();
  int16_t GyY = Wire.read() << 8 | Wire.read();
  int16_t GyZ = Wire.read() << 8 | Wire.read();

  ax = AcX / 16384.0;
  ay = AcY / 16384.0;
  az = AcZ / 16384.0;

  gx = GyX / 131.0;
  gy = GyY / 131.0;
  gz = GyZ / 131.0;

  temp = Tmp / 340.0 + 36.53;

  float t = millis() / 1000.0;

  String data =
    String(t, 3) + "," +
    String(ax, 3) + "," +
    String(ay, 3) + "," +
    String(az, 3) + "," +
    String(gx, 3) + "," +
    String(gy, 3) + "," +
    String(gz, 3) + "," +
    String(temp, 2);

  udpIMU.beginPacket(broadcastIP, IMU_SEND_PORT);
  udpIMU.print(data);
  udpIMU.endPacket();
}

// ================= RECEIVE UDP COMMAND =================
void receiveUDPCommand() {
  int packetSize = udpCMD.parsePacket();

  if (packetSize > 0) {
    char incomingPacket[50];
    int len = udpCMD.read(incomingPacket, sizeof(incomingPacket) - 1);

    if (len > 0) {
      incomingPacket[len] = '\0';

      String cmd = String(incomingPacket);
      cmd.trim();

      Serial.print("UDP CMD Received: ");
      Serial.println(cmd);

      if (cmd == "STOP") {
        applyPWM(0);
      }
      else if (cmd.startsWith("PWM:")) {
        int pwmValue = cmd.substring(4).toInt();
        applyPWM(pwmValue);
      }
      else if (cmd.length() == 1) {
        char command = cmd.charAt(0);
        handleCommand(command);
      }
    }
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_R, OUTPUT);
  pinMode(LED_G, OUTPUT);
  pinMode(LED_B, OUTPUT);
  allOff();

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  setupSpeaker();
  setupMPU();

  ledcAttach(MOTOR_PIN, pwmFreq, pwmResolution);
  applyPWM(0);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);

  udpAudio.begin(AUDIO_PORT);
  udpIMU.begin(IMU_SEND_PORT);
  udpCMD.begin(PWM_CMD_PORT);

  Serial.println("-----------------------------------");
  Serial.println("INTEGRATED GLOVE SYSTEM READY");
  Serial.print("WiFi Name: ");
  Serial.println(AP_SSID);
  Serial.print("Password: ");
  Serial.println(AP_PASS);
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.println("Audio UDP Port: 5005");
  Serial.println("IMU UDP Port: 1234");
  Serial.println("PWM CMD Port: 1235");
  Serial.println("Waiting for Python UDP commands...");
  Serial.println("-----------------------------------");
}

// ================= LOOP =================
void loop() {
  // 1. MIC AUDIO STREAM
  int raw = analogRead(MIC_PIN);
  audio_buf[audio_index++] = (int16_t)((raw - 2048) * 16);

  if (audio_index >= AUDIO_PACKET_SIZE) {
    udpAudio.beginPacket(broadcastIP, AUDIO_PORT);
    udpAudio.write((uint8_t*)audio_buf, AUDIO_PACKET_SIZE * 2);
    udpAudio.endPacket();
    audio_index = 0;
  }

  // 2. IMU STREAM
  static unsigned long lastIMU = 0;
  if (millis() - lastIMU >= 10) {
    sendIMUData();
    lastIMU = millis();
  }

  // 3. RECEIVE UDP COMMANDS
  receiveUDPCommand();

  // 4. OPTIONAL USB SERIAL TEST
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command != '\n' && command != '\r') {
      handleCommand(command);
    }
  }

  delayMicroseconds(62);
}
