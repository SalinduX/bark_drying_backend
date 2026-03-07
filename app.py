# -*- coding: utf-8 -*-
import RPi.GPIO as GPIO
import adafruit_dht
import board
import time
from datetime import datetime
from picamera2 import Picamera2
import cv2
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

# ---- Your Pins ----
HEATER_PIN = 27
FAN_PIN    = 17
DHT_PIN    = board.D4

# ---- Control Settings ----
HEATER_ON_TEMP  = 40    # Heater ON  if temp BELOW this
HEATER_OFF_TEMP = 55    # Heater OFF if temp ABOVE this

# ---- Camera & Model ----
MODEL_PATH = "/home/sali/bark_dry_project/cinnaDry-model.eim"

# ---- Drying Time Settings ----
BASE_TIME = 360         # Base drying time in minutes (6 hours)

# ---- Log File ----
LOG_FILE = "/home/sali/bark_dry_project/bark_dry_log.txt"

# ---- GPIO Setup ----
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(HEATER_PIN, GPIO.OUT)
GPIO.setup(FAN_PIN, GPIO.OUT)

# Heater OFF, Fan always ON at startup
GPIO.output(HEATER_PIN, GPIO.HIGH)  # HIGH = OFF
GPIO.output(FAN_PIN, GPIO.LOW)      # LOW  = ON (Active LOW relay)

dht_sensor   = adafruit_dht.DHT22(DHT_PIN)
heater_state = "OFF"
fan_state    = "ON"
drying_start = datetime.now()

# ---- Helper Functions ----
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(message):
    line = "[" + get_time() + "] " + message
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def heater_on():
    GPIO.output(HEATER_PIN, GPIO.LOW)   # LOW  = ON

def heater_off():
    GPIO.output(HEATER_PIN, GPIO.HIGH)  # HIGH = OFF

def calculate_drying_rate(temp, hum):
    if hum == 0:
        hum = 1
    return (temp / 45) * (50 / hum)

def estimate_remaining_time(temp, hum):
    rate = calculate_drying_rate(temp, hum)
    if rate == 0:
        return BASE_TIME
    total_time = BASE_TIME / rate
    elapsed    = (datetime.now() - drying_start).seconds / 60
    remaining  = total_time - elapsed
    return max(0, remaining)

# ---- Camera Setup ----
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)

# ---- Startup Log ----
log("==============================")
log("  BARK DRYER CONTROLLER START ")
log("  Fan        : ALWAYS ON      ")
log("  Heater ON  : temp < 40C     ")
log("  Heater OFF : temp > 55C     ")
log("  Base drying time : 360 min  ")
log("  Drying started   : " + get_time())
log("==============================")

# ---- Main Loop ----
with ImageImpulseRunner(MODEL_PATH) as runner:
    model_info = runner.init()
    log("Model loaded: " + model_info["project"]["name"])
    log("Labels: " + str(model_info["model_parameters"]["labels"]))

    try:
        while True:
            try:
                time.sleep(2)  # DHT22 needs 2 sec between reads
                temp = dht_sensor.temperature
                hum  = dht_sensor.humidity

                if temp is None or hum is None:
                    log("WARNING: Sensor failed to read - retrying...")
                    continue

                # ---- Log Sensor Reading ----
                log("Temp: " + str(round(temp, 1)) + " C   Humidity: " + str(round(hum, 1)) + " %")

                # ---- HEATER LOGIC ----
                if temp < HEATER_ON_TEMP and heater_state == "OFF":
                    heater_on()
                    heater_state = "ON"
                    log("HEATER ON  -> temp " + str(round(temp, 1)) + "C is below 40C")

                elif temp > HEATER_OFF_TEMP and heater_state == "ON":
                    heater_off()
                    heater_state = "OFF"
                    log("HEATER OFF -> temp " + str(round(temp, 1)) + "C is above 55C")

                # ---- FAN always ON ----
                fan_state = "ON"

                # ---- DRYING TIME ESTIMATION ----
                remaining   = estimate_remaining_time(temp, hum)
                hours       = int(remaining // 60)
                minutes     = int(remaining % 60)
                elapsed_min = round((datetime.now() - drying_start).seconds / 60, 1)

                if remaining == 0:
                    log("DRYING STATUS : COMPLETE!")
                else:
                    log("DRYING STATUS : In progress")

                log("Elapsed time  : " + str(elapsed_min) + " minutes")
                log("Remaining time: " + str(hours) + " hr " + str(minutes) + " min")

                # ---- Log Device States ----
                log("Heater: " + heater_state + "   Fan: " + fan_state)
                log("------------------------------")

                # ---- Wait 15 min then capture photo ----
                log("Waiting 15 minutes before next camera check...")
                time.sleep(900)

                # ---- CAMERA CAPTURE ----
                frame = picam2.capture_array()
                log("Photo captured!")

                # ---- ML INFERENCE ----
                try:
                    features, cropped = runner.get_features_from_image(frame)
                    result = runner.classify(features)

                    if "classification" in result["result"]:
                        scores    = result["result"]["classification"]
                        top_label = max(scores, key=scores.get)
                        conf      = scores[top_label]
                        log("ML decision: " + top_label + " (" + str(round(conf * 100, 1)) + "%)")

                        if top_label.lower() == "dry" and conf > 0.7:
                            log("RESULT: YES - bark is dry!")
                        else:
                            log("RESULT: NO  - bark not dry yet")

                    elif "bounding_boxes" in result["result"]:
                        for bb in result["result"]["bounding_boxes"]:
                            log("Detected: " + bb["label"] + " at x=" + str(bb["x"]) + " y=" + str(bb["y"]))

                except Exception as e:
                    log("ML ERROR: " + str(e))

            except Exception as e:
                log("ERROR: " + str(e))
                time.sleep(2)

    except KeyboardInterrupt:
        log("System stopped by user.")

    finally:
        picam2.stop()
        heater_off()
        GPIO.output(FAN_PIN, GPIO.HIGH)  # Fan OFF on exit
        GPIO.cleanup()
        log("Everything OFF. Safe to exit.")
        log("==============================")
