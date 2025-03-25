import RPi.GPIO as GPIO

class gpioController: # 모터 구동 예정
    def __init__(self, button_pin=17, speaker_pin=18):
        self.button_pin = button_pin
        self.speaker_pin = speaker_pin

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(speaker_pin, GPIO.OUT)

        self.buzzer = GPIO.PWM(speaker_pin, 1000)

    def get_button(self):
        return GPIO.input(self.button_pin) == GPIO.HIGH
    
    def make_buzz(self, t=50):
        self.buzzer.start(t)

    def stop_buzz(self):
        self.buzzer.stop()

    def end_proc(self):
        GPIO.cleanup()
