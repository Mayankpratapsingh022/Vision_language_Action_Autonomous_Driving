LEFT_TURN_INTENT_ID = 0
LEFT_TURN_INTENT = "turn_left_intersection"
LEFT_TURN_INSTRUCTION = (
    "Proceed through the city and make the protected left turn at the main intersection."
)

CAMERA_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

STATE_NAMES = ("speed_mps", "steering", "previous_throttle", "previous_brake")
ACTION_NAMES = ("throttle", "brake", "steering")

MAX_SPEED_MPS = 24.0
CAPTURE_FPS = 10
IMAGE_SIZE = 128

