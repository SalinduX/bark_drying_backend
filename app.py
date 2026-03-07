# -*- coding: utf-8 -*-
import time
import board
import adafruit_dht
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import cv2
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

DHT_PIN = 4
HEATER_PIN = 27
FAN_PIN = 17
MODEL_PATH = "/home/sali/bark_dry_project/cinnaDry-model.eim"
TEMP_TARGET = 35

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(HEATER_PIN, GPIO.OUT)
GPIO.setup(FAN_PIN, GPIO.OUT)

GPIO.output(HEATER_PIN, GPIO.HIGH)
GPIO.output(FAN_PIN, GPIO.LOW)

dht_sensor = adafruit_dht.DHT22(board.D4)

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

time.sleep(2)

print("Bark drying system STARTED! Press Ctrl+C to stop.")

with ImageImpulseRunner(MODEL_PATH) as runner:
    model_info = runner.init()
    print("Model loaded: " + model_info["project"]["name"])

    temp = None

    try:
        while True:

            try:
                temp = dht_sensor.temperature
                hum = dht_sensor.humidity

                if temp is not None and hum is not None:
                    print("Temp: " + str(round(temp, 1)) + " C   Humidity: " + str(round(hum, 1)) + " %")
                else:
                    print("Sensor failed to read")

            except Exception as e:
                print("Sensor error: " + str(e))

            if temp is not None:

                if temp < TEMP_TARGET:
                    GPIO.output(HEATER_PIN, GPIO.LOW)
                    print("Heater ON")
                else:
                    GPIO.output(HEATER_PIN, GPIO.HIGH)
                    print("Heater OFF")

                GPIO.output(FAN_PIN, GPIO.HIGH)
                print("Fan ON")

            print("Waiting before next capture...")
            time.sleep(10)

            frame = picam2.capture_array()
            print("Photo captured")

            # Show captured image in a window
            cv2.imshow("Bark Drying Camera", frame)
            cv2.waitKey(1)

            try:
                features, cropped = runner.get_features_from_image(frame)
                result = runner.classify(features)

                if "classification" in result["result"]:
                    scores = result["result"]["classification"]

                    top_label = max(scores, key=scores.get)
                    conf = scores[top_label]

                    decision_text = top_label + " (" + str(round(conf * 100, 1)) + "%)"
                    print("ML decision: " + decision_text)

                    if top_label.lower() == "dry" and conf > 0.7:
                        print("Backend decision: YES - bark is dry")
                        label_text = "DRY"
                    else:
                        print("Backend decision: NO - bark not dry yet")
                        label_text = "NOT DRY"

                    # Draw ML result on image
                    cv2.putText(
                        frame,
                        label_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    cv2.imshow("Bark Drying Camera", frame)
                    cv2.waitKey(1)

            except Exception as e:
                print("ML error: " + str(e))

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Clean exit")
