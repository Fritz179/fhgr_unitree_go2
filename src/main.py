import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1] 
sys.path.insert(0, str(ROOT))

from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

from unitree_sdk2_python.unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2_python.unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2_python.unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, IMUState_, BmsState_, SportModeState_

import cv2
import numpy as np
import sys
import time

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
        f"  battery      = {battery_percent:.1f}% / {battery_current:.1f}A"
    )
    # if you also want battery etc, you can look at msg.power_v / msg.power_a here


from ai.ai import AIClient

if __name__ == "__main__":
    ChannelFactoryInitialize(0, "wlo1")
    # ChannelFactoryInitialize(0)

    # https://support.unitree.com/home/en/developer/sports_services
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    # https://support.unitree.com/home/en/developer/Multimedia_Services
    video_client = VideoClient()  # Create a video client 
    video_client.SetTimeout(3.0)
    video_client.Init()

    # https://support.unitree.com/home/en/developer/Motion%20Switcher%20Service%20Interface
    state_client = MotionSwitcherClient()
    state_client.SetTimeout(5.0)
    state_client.Init()

    # State subscriber
    low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_sub.Init(low_state_handler, 10)

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(handle_sport_state, 10)

    ai_client = AIClient()
    frame = 0

    wait = (20, "reset")

    window_name = "The Dog"

    while True:
        
        # Get Image data from Go2 robot
        code, data = video_client.GetImageSample()

        if code != 0:
            print(f"Failed to get image data: {code}")
            continue

        image = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print("Failed to decode image")
            continue

        if image.size == 0:
            print("Empty image")
            continue

        image = cv2.resize(image, (1280, 720))

        action, image = ai_client.update(image)

        
        code, info = state_client.CheckMode()
        # print(f"Frame: {frame}, Action: {action}, wait: {wait}, Current motion mode: {info}, code: {code}")
        # print_low_state()
        # print_sport_state()
        frame += 1

        # Display image
        cv2.imshow(window_name, image)

        """ Available Poses:
            grabbing, grip, holy, point, call, three3, 
            timeout, xsign, hand_heart, hand_heart2, little_finger, 
            middle_finger, take_picture, dislike, fist, four, 
            like, mute, ok, one, palm, peace, peace_inverted, 
            rock, stop, stop_inverted, three, three2, two_up, 
            two_up_inverted, three_gun, thumb_index, thumb_index2, no_gesture
        """

        """ Available Sport APIs:
            Damp, BalanceStand, StopMove, StandUp, StandDown, RecoveryStand, 
            Euler, Move, Sit, RiseSit, SpeedLevel, Hello, Stretch, Content, 
            Dance1, Dance2, SwitchJoystick, Pose, Scrape, FrontFlip, FrontJump, 
            FrontPounce, Heart, StaticWalk, TrotRun, EconomicGait, LeftFlip, 
            BackFlip, HandStand, FreeWalk, FreeBound, FreeJump, FreeAvoid, 
            ClassicWalk, WalkUpright, CrossStep, AutoRecoverySet, AutoRecoveryGet, 
            SwitchAvoidMode
        """

        """ Interesting Sport APIs:
            Damp, BalanceStand, StopMove, StandUp, StandDown
            Euler, Move, Sit, RiseSit, SpeedLevel, Hello, Stretch, Content, 
            Dance1, Dance2, SwitchJoystick, Pose, Scrape, FrontFlip, FrontJump, 
            FrontPounce, Heart, StaticWalk, TrotRun, EconomicGait, LeftFlip, 
            BackFlip, HandStand, FreeWalk, FreeBound, FreeJump, WalkUpright
        """

        """
            a, hand_heart, hand_heart2 -> Heart [blocking]
            s, holy (hands together) -> Scrape (chinese greet) [blocking]
            d, palm (open fingers) -> Hello (wave hand) [blocking]
            f, stop (palm closed fingers) -> StandDown [toggle with BalanceStand]
            g, fist -> Sit [toggle with RiseSit]
            h, peace (V sign) -> Content [blocking]
            j, like -> WalkUpright [toggle with WalkUpright(False)]
            k, dislike -> HandStand [toggle with HandStand(False)]
            l, three_gun -> LeftFlip [blocking]
            w, e, rock -> dance1, dance2 [blocking]
            r, middle_finger -> pounce [blocking]
            t, peace_inverted -> jump_forward [blocking]
            z, grabbing -> stretch [blocking]
        """

        print(f"Action: {action}, wait: {wait}")

        # Press ESC to stop
        if action == "quit":
            break

        if action == " ":
            sport_client.BalanceStand()
            cv2.waitKey(1)
            sport_client.Euler(0, -0.3, 0)
            cv2.waitKey(1)
            sport_client.BalanceStand()
            cv2.waitKey(1)
            sport_client.Euler(0, -0.3, 0)
            wait = (0, "done")
            continue

        if wait[0] > 0:
            wait = (wait[0]-1, wait[1])
            continue

        if wait[0] == 0 and wait[1] != "done":
            cv2.waitKey(1)

            action = wait[1]
            wait = (0, "done")
            print(f"Finished waiting for action: {action}")
            print(f"Finished waiting for action: {action}")
            
            if action == "rise_sit":
                sport_client.RiseSit()
            elif action == "stand_up":
                sport_client.StandUp()
            elif action == "walk_upright_off":
                sport_client.WalkUpright(False)
                wait = (10, "reset")
            elif action == "hand_stand_off":
                sport_client.HandStand(False)
                wait = (10, "reset")
            elif action == "reset":
                print("Resetting posture")
                sport_client.BalanceStand()
                cv2.waitKey(1)
                sport_client.Euler(0, -0.3, 0)
                cv2.waitKey(1)
                sport_client.BalanceStand()
                cv2.waitKey(1)
                sport_client.Euler(0, -0.3, 0)

            continue

        if action == "hand_heart" or action == "hand_heart2" or action == "a":
            cv2.waitKey(1)
            sport_client.Heart() # Blocking call

        elif action == "holy" or action == "s":
            cv2.waitKey(1)
            sport_client.Scrape() # Blocking call

        elif action == "palm" or action == "d":
            cv2.waitKey(1)
            sport_client.Hello()

        elif action == "stop" or action == "f":
            cv2.waitKey(1)
            sport_client.StandDown()

            wait = (50, "stand_up")  # after 50 frames, stand back up

        elif action == "fist" or action == "g":
            cv2.waitKey(1)
            sport_client.Sit()

            wait = (50, "rise_sit")  # after 50 frames, rise back up

        elif action == "peace" or action == "h":
            cv2.waitKey(1)
            sport_client.Content()  # Blocking call

        elif action == "like" or action == "j":
            cv2.waitKey(1)
            sport_client.WalkUpright(True)
            wait = (50, "walk_upright_off")

        elif action == "dislike" or action == "k":
            cv2.waitKey(1)
            sport_client.HandStand(True)
            wait = (50, "hand_stand_off")

        elif action == "three_gun" or action == "l":
            cv2.waitKey(1)
            sport_client.LeftFlip()  # Blocking call
            wait = (20, "reset")


        elif action == "rock" or action == "w" or action == "e":
            cv2.waitKey(1)

            if action == "w":
                sport_client.Dance1()
            elif action == "e":
                sport_client.Dance2()

            elif np.random.rand() < 0.5:
                sport_client.Dance2()
            else:
                sport_client.Dance1()

        elif action == "middle_finger" or action == "r":
            cv2.waitKey(1)
            sport_client.FrontPounce()

        elif action == "peace_inverted" or action == "t":
            cv2.waitKey(1)
            sport_client.FrontJump()

        elif action == "grabbing" or action == "z":
            cv2.waitKey(1)
            sport_client.Stretch()

        elif action != "no_gesture" and action != None:
            print(f"Unknown action: {action}")



    cv2.destroyWindow(window_name)

"""
TODO:
- same command has to be held for some time to be executed
- hold command that is being exeuted
- show camera while preset moves?
- cleanup code
"""
