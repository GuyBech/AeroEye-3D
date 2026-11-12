/*
 * AeroEye-3D — non-blocking UART servo control for tracking.
 * PC streams "PAN,TILT,LASER\n" targets; the board acknowledges IMMEDIATELY
 * and, in its main loop, continuously chases the latest target in fine 1-deg
 * steps (smooth, no visible jumps).
 *
 * Portable: builds for BOTH boards (PlatformIO envs: -e uno / -e esp32).
 *   Arduino UNO R3 : Pan=D9  | Tilt=D10 | Laser=D13 (onboard LED mirrors it)
 *   ESP32 DevKit   : Pan=13  | Tilt=14  | Laser=27     | LED=2
 * Laser drives a 2N2222A transistor base through 1k. Servo power comes from
 * the external 5V supply; ALL grounds must be common.
 */
#include <Arduino.h>
#ifdef ARDUINO_ARCH_ESP32
  #include <ESP32Servo.h>
  constexpr uint8_t PAN_PIN = 13, TILT_PIN = 14, LASER_PIN = 27, LED_PIN = 2;
#else
  #include <Servo.h>
  constexpr uint8_t PAN_PIN = 9, TILT_PIN = 10, LASER_PIN = 13, LED_PIN = 13;
#endif

constexpr uint32_t BAUD = 115200;
constexpr int PAN_MIN  = 0,   PAN_MAX  = 180;
constexpr int TILT_MIN = 3,   TILT_MAX = 150;
constexpr int STEP_DEG = 1;     // fine steps — smooth motion
constexpr uint32_t STEP_MS = 2; // -> ~500 deg/s max slew

Servo panServo, tiltServo;
int panAngle = 90, tiltAngle = 20;
int panTarget = 90, tiltTarget = 20;
uint32_t lastStep = 0;

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

int stepToward(int cur, int target) {
  if (cur < target) return min(cur + STEP_DEG, target);
  if (cur > target) return max(cur - STEP_DEG, target);
  return cur;
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;
  int c1 = line.indexOf(','), c2 = line.indexOf(',', c1 + 1);
  if (c1 < 0 || c2 < 0) { Serial.println(F("ERR")); return; }
  panTarget  = clampi(line.substring(0, c1).toInt(),      PAN_MIN,  PAN_MAX);
  tiltTarget = clampi(line.substring(c1 + 1, c2).toInt(), TILT_MIN, TILT_MAX);
  int laser  = line.substring(c2 + 1).toInt();
  digitalWrite(LASER_PIN, laser ? HIGH : LOW);
  digitalWrite(LED_PIN,   laser ? HIGH : LOW);
  // immediate ack (portable — AVR has no Serial.printf)
  Serial.print(F("OK "));  Serial.print(panTarget);
  Serial.print(' ');       Serial.print(tiltTarget);
  Serial.print(' ');       Serial.println(laser ? 1 : 0);
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);
  Serial.begin(BAUD);
  Serial.setTimeout(10);   // never block long on a torn line
  delay(300);
#ifdef ARDUINO_ARCH_ESP32
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN, 400, 2500);
#else
  panServo.attach(PAN_PIN);              // UNO: standard Servo.h (Timer1)
  tiltServo.attach(TILT_PIN, 400, 2500); // widened pulse range for tilt
#endif
  panServo.write(panAngle);
  tiltServo.write(tiltAngle);
  Serial.println(F("\nAeroEye-3D :: non-blocking fast tracking"));
}

void loop() {
  if (Serial.available()) {
    handleLine(Serial.readStringUntil('\n'));
  }
  if (millis() - lastStep >= STEP_MS) {
    lastStep = millis();
    int np = stepToward(panAngle, panTarget);
    int nt = stepToward(tiltAngle, tiltTarget);
    if (np != panAngle)  { panAngle = np;  panServo.write(panAngle); }
    if (nt != tiltAngle) { tiltAngle = nt; tiltServo.write(tiltAngle); }
  }
}
