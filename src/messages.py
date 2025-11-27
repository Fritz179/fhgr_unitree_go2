from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, IMUState_, BmsState_, SportModeState_, UwbState_
from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelSubscriber

from typing import TYPE_CHECKING


disable_robot = None

def register_messages(disable_robot_callback):
    global disable_robot
    disable_robot = disable_robot_callback

    # State subscriber
    low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_sub.Init(low_state_handler, 10)

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(handle_sport_state, 10)

    sub = ChannelSubscriber("rt/uwbstate", UwbState_)
    sub.Init(handle_uwb_state, 10)

def mode_to_str(mode: int) -> str:
    # Mapping based on Go1/Go2 HighCmd docs (not official names, just handy labels)
    return {
        0: "idle",
        1: "stand (force control)",
        2: "walk (velocity)",
        3: "walk (position/path)",
        4: "walk (given path)",
        5: "stand down",
        6: "stand up",
        7: "damping",
        8: "recovery",
        9: "backflip",
        10: "jumpYaw",
        11: "straightHand",
        12: "dance1",
        13: "dance2",
    }.get(mode, f"unknown({mode})")


def gait_to_str(gait: int) -> str:
    return {
        0: "idle",
        1: "trot walk",
        2: "trot run",
        3: "stairs",
        4: "trot obstacle",
    }.get(gait, f"unknown({gait})")


sport_state = None
def handle_sport_state(msg: SportModeState_):
    global sport_state
    sport_state = msg

def print_sport_state():
    msg: SportModeState_ = sport_state
    if msg is None:
        return
    
    # Basic sanity / “ready” heuristic
    ready_for_motion = (msg.error_code == 0) and (msg.mode in (1, 2, 3, 4, 6))
    in_damping = (msg.mode == 7)

    # Short line summary so you can tail it
    print(
        "State:\n"
        f"  error={msg.error_code} "
        f"  mode={mode_to_str(msg.mode)}, {msg.mode} "
        f"  gait={gait_to_str(msg.gait_type)} "
        f"  ready={ready_for_motion} "
        f"  damping={in_damping} "
    )

low_state = None
def low_state_handler(msg: LowState_):
    global low_state
    low_state = msg

def print_low_state():
    msg: LowState_ = low_state
    if msg is None:
        return

    imu: IMUState_ = msg.imu_state
    bms: BmsState_ = msg.bms_state
    # mode: SportModeState_ = msg.sport_mode_state

    gyro = imu.gyroscope
    acc = imu.accelerometer
    rpy = imu.rpy
    temp = imu.temperature

    battery_percent = bms.soc
    battery_current = bms.current / 1000.0  # convert to A   

    print(
        "IMU:\n"
        f"  gyroscope    = [{gyro[0]: .4f}, {gyro[1]: .4f}, {gyro[2]: .4f}]  # rad/s\n"
        f"  accelerometer= [{acc[0]: .4f}, {acc[1]: .4f}, {acc[2]: .4f}]  # m/s^2\n"
        f"  rpy          = [{rpy[0]: .4f}, {rpy[1]: .4f}, {rpy[2]: .4f}]  # rad\n"
        f"  temperature  = {int(temp)} °C\n"
        ""
        f"Battery:       = {battery_percent:.1f}% / {battery_current:.1f}A"
    )
    # if you also want battery etc, you can look at msg.power_v / msg.power_a here

uwb_state: UwbState_ = None
uwb_was_active = 0

def handle_uwb_state(msg: UwbState_):
    global uwb_state, uwb_was_active
    uwb_state = msg

    if msg.buttons != 0 or msg.joystick[0] != 0 or msg.joystick[1] != 0:
        uwb_was_active = 20  # keep active for 20 frames

    if msg.buttons == 4: # M button
        print("M button pressed")
        disable_robot()

from dataclasses import dataclass
from typing import Sequence

@dataclass
class State:
    gyroscope: Sequence[float]
    accelerometer: Sequence[float]
    rpy: Sequence[float]
    temperature: float
    battery_percent: float
    battery_current: float
    uwb_active: bool

def get_state() -> State:
    global low_state, uwb_state, uwb_was_active

    uwb_active = uwb_was_active > 0
    if uwb_active:
        uwb_was_active -= 1
    
    return State(
        gyroscope = [0, 0, 0] if low_state is None else low_state.imu_state.gyroscope,
        accelerometer = [0, 0, 0] if low_state is None else low_state.imu_state.accelerometer,
        rpy = [0, 0, 0] if low_state is None else low_state.imu_state.rpy,
        temperature = 0 if low_state is None else low_state.imu_state.temperature,

        battery_percent = 0.0 if low_state is None else low_state.bms_state.soc,
        battery_current = 0.0 if low_state is None else low_state.bms_state.current / 1000.0,

        uwb_active = uwb_active,
    )