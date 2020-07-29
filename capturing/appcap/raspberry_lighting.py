from appcap.argument_handler import ArgumentHandler
import RPi.GPIO as GPIO


class RaspberryLighting:

    def __init__(self):
        
        args = ArgumentHandler.get_args()
        
        self.lamp_names = args.lamp_names
        self.pins = args.lamp_pins
        self.led_cnt = len(self.pins)
        GPIO.setmode(GPIO.BCM)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)

    def get_led_cnt(self):
        return self.led_cnt

    def turn_off_all(self):
        for pin in self.pins:
            GPIO.output(pin, GPIO.HIGH)
	print("Turned off all LEDs.")

    def turn_on_all(self):
        for pin in self.pins:
            GPIO.output(pin, GPIO.LOW)
	print("Turned on all LEDs.")

    def turn_on_idx(self, led_idx):
	if led_idx==-1:
		self.turn_on_all()
	else:
	        self.turn_off_all()
        	GPIO.output(self.pins[led_idx], GPIO.LOW)
	        print("Turned on led " + str(led_idx) + ".")

    def get_angle_for_idx(self, led_idx):
	if led_idx==-1:
		return 360
        return self.lamp_names[led_idx]
