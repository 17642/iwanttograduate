#include <Adafruit_NeoPixel.h>
#include <Arduino.h>
#include <hp_BH1750.h>  //inlude the library
#include "Seeed_Arduino_mmWave.h"
#include <PubSubClient.h>
#include <WiFi.h>

#ifdef ESP32
#  include <HardwareSerial.h>
HardwareSerial mmwaveSerial(0);
#else
#  define mmwaveSerial Serial1
#endif

#define LIGHT_GPIO D0

/****** WI-FI Setting ******/

const char* WIFI_NAME = "RPI_HOTSPOT_319";
const char* WIFI_PWD = "RPIHOTSPOT319";

/****** mqtt Setting ******/

const char* MQTT_BROKER = "192.168.1.10";
const uint16_t MQTT_PORT = 1883;
const char* MQTT_TOPIC = "home/sensors/fall";

/****** instance ******/

hp_BH1750 BH1750;  // create the sensor object

SEEED_MR60FDA2 mmWave;

WiFiClient espClient;
PubSubClient mqtt(espClient);

Adafruit_NeoPixel pixels =
    Adafruit_NeoPixel(1, /* pixelPin */ D1, NEO_GRB + NEO_KHZ800);

/****** funtions ******/

void relay_init();
void relay_on();
void relay_off();

void reconnect_mqtt();

/****** variables ******/
uint32_t sensitivity = 15;
float height = 2.8, threshold = 1.0;
float rect_XL, rect_XR, rect_ZF, rect_ZB;

const uint8_t dark_lux = 10;

void setup() {
  bool result;
  int failCount = 0;
  int maxFailCount = 100;
  Serial.begin(115200);
  mmWave.begin(&mmwaveSerial);
  /* init relay device*/
  relay_init();

  /* init RGB LED */
  pixels.begin();
  pixels.clear();
  pixels.setBrightness(8);
  pixels.show();
  pixels.setPixelColor(0, pixels.Color(125, 125, 125));
  /* init built-in light ambient light sensor */
  BH1750.begin(BH1750_TO_GROUND);  // will be false no sensor found
                                   // | already connected to I2C
  BH1750.calibrateTiming();
  BH1750.start(BH1750_QUALITY_HIGH2,
               254);  // start the first measurement in setup
  /* set mmwave-fall parameters */
  mmWave.setUserLog(0);

  /** set the height of the installation **/
  if (mmWave.setInstallationHeight(height)) {
    Serial.printf("setInstallationHeight success: %.2f\n", height);
  } else {
    Serial.println("setInstallationHeight failed");
  }

  /** Set threshold **/
  if (mmWave.setThreshold(threshold)) {
    Serial.printf("setThreshold success: %.2f\n", threshold);
  } else {
    Serial.println("setThreshold failed");
  }

  /** Set sensitivity **/
  if (mmWave.setSensitivity(sensitivity)) {
    Serial.printf("setSensitivity success %d\n", sensitivity);
  } else {
    Serial.println("setSensitivity failed");
  }

  /** get new parameters of mmwave **/
  if (mmWave.getRadarParameters(height, threshold, sensitivity, rect_XL,
                                rect_XR, rect_ZF, rect_ZB)) {
    Serial.printf("height: %.2f\tthreshold: %.2f\tsensitivity: %d\n", height,
                  threshold, sensitivity);
    Serial.printf(
        "rect_XL: %.2f\trect_XR: %.2f\trect_ZF: %.2f\trect_ZB: %.2f\n", rect_XL,
        rect_XR, rect_ZF, rect_ZB);
  } else {
    Serial.println("getRadarParameters failed");
  }

  WiFi.begin(WIFI_NAME, WIFI_PWD);
  Serial.print("WiFi Connecting");
  while(WiFi.status()!=WL_CONNECTED){
    failCount++;
    Serial.print(".");
    delay(300);

    if(failCount==maxFailCount){
      Serial.println(" Failed");
      Serial.println("WIFI is not Connected. test mode initialized");
      break;
    }
  }
  Serial.println(" Connected!");

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
}

typedef enum {
  EXIST_PEOPLE,
  NO_PEOPLE,
  PEOPLE_FALL,
} MMWAVE_STATUS;

MMWAVE_STATUS status = NO_PEOPLE, last_status = NO_PEOPLE;
float lux = 100;

void loop(){
  if (WiFi.status() == WL_CONNECTED && !mqtt.connected()){
    reconnect_mqtt();
  }

  if(mmWave.update(100)){
    bool is_human, is_fall;

    if(mmWave.getHuman(is_human)){
      if(mmWave.getFall(is_fall)){
        if(!is_human && !is_fall){
          status = NO_PEOPLE;
        }
        else if(is_fall){
          status = PEOPLE_FALL;
        }else{
          status = EXIST_PEOPLE;
        }
      }
    }

    if (!mmWave.getHuman(is_human) && !mmWave.getFall(is_fall)) {
      status = NO_PEOPLE;  // No human and no fall detected
    } else if (is_fall) {
      status = PEOPLE_FALL;  // Fall detected
    } else {
      status = EXIST_PEOPLE;  // Human detected without fall
    }

  }

  switch (status) {
    case NO_PEOPLE:
      Serial.printf("Waiting for people");
      break;
    case EXIST_PEOPLE:
      Serial.printf("PEOPLE !!!");
      break;
    case PEOPLE_FALL:
      Serial.printf("FALL !!!");
      break;
    default:
      break;
  }
  Serial.print("\n");

  
  char buf[128];
  bool pub_msg = false;

  char nst[10];

/* change interactive Light*/
  if (status != last_status) {  // switching LED
    pub_msg = true;
    switch (status) {
      case NO_PEOPLE:
        pixels.setPixelColor(0, pixels.Color(0, 0, 255));  // BLUE
        strcpy(nst, "\"NONE\"");
        break;
      case EXIST_PEOPLE:
        pixels.setPixelColor(0, pixels.Color(0, 255, 0));  // GREEN
        strcpy(nst, "\"EXIST\"");
        break;
      case PEOPLE_FALL:
        pixels.setPixelColor(0, pixels.Color(255, 0, 0));  // RED
        strcpy(nst, "\"FALL\"");
        break;
      default:
        break;
    }
    pixels.show();
    last_status = status;
  }

  /* update lux value */
  if (BH1750.hasValue() == true) {
    lux = BH1750.getLux();
    BH1750.start(BH1750_QUALITY_HIGH2, 254);
  }

  Serial.print("LUX: ");
  Serial.print(lux);
  Serial.print("\t");

  int len = snprintf(buf, sizeof(buf),"{\"timestamp\":%lu,\"status\":%s,\"lux\":%.1f}",millis()/1000,nst,lux);

  if(pub_msg){
    mqtt.publish(MQTT_TOPIC,buf,len);
    Serial.println(buf);
  }

  if ((status == EXIST_PEOPLE || status == PEOPLE_FALL) && lux < dark_lux) {
    relay_on();
  } else {
    relay_off();
  }
}

void relay_init() {
  pinMode(LIGHT_GPIO, OUTPUT);
}
void relay_on() {
  digitalWrite(LIGHT_GPIO, HIGH);
}
void relay_off() {
  digitalWrite(LIGHT_GPIO, LOW);
}

void reconnect_mqtt(){
  while(!mqtt.connected()){
    Serial.print("MQTT connecting...");
    if (mqtt.connect("esp32_mr60")){
      Serial.println(" Connected.");
      break;
    }

    Serial.print(" Failed, rc = ");
    Serial.print(mqtt.state());
    Serial.println(" Retrying in 2s");
    delay(2000);
  }
}