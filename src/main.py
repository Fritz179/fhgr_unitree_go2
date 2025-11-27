import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1] 
sys.path.insert(0, str(ROOT))

from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2_python.unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2_python.unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2_python.unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import cv2
import numpy as np
from pynput import keyboard
from ai.ai import AIClient

from messages import get_state, register_messages

if __name__ == "__main__":
    # ChannelFactoryInitialize(0, "wlo1")
    ChannelFactoryInitialize(0)

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

    ai_client = AIClient()

    # App state, wait = (frames to wait, action after wait)
    wait = (20, "reset")

    disabled = False
    safe_mode = True
    following = False
    following_state = None
    frame = 0

    def disable_robot():
        global disabled

        print(f"Please ensure the robot is safe and press Enter to re-enable")
        disabled = True
        sport_client.Damp()

    register_messages(disable_robot)

    def on_press(key):
        global disabled, safe_mode, following, wait

        if key == keyboard.Key.space:
            print(f"Space key pressed, damping the robot")
            disable_robot()

        elif key == keyboard.Key.enter:
            print(f"Enter key pressed, enabling the robot")
            disabled = False
            wait = (1, "reset")

        elif key == keyboard.Key.ctrl:
            safe_mode = not safe_mode
            following = False
            print(f"Ctrl key pressed, toggling safe mode (currently {'ON' if safe_mode else 'OFF'})")

        elif key == keyboard.Key.shift:
            if safe_mode:
                print("Safe mode is ON, cannot enable following mode")
                return
            
            following = not following
            print(f"Shift key pressed, toggling following mode (currently {'ON' if following else 'OFF'})")

    def reset_pose():
        # Sometimes the robot needs a bit of a nudge to get back to normal posture
        sport_client.RecoveryStand()
        sport_client.BalanceStand()
        cv2.waitKey(1)
        sport_client.Euler(0, 0, 0)
        cv2.waitKey(1)
        sport_client.BalanceStand()
        cv2.waitKey(1)
        sport_client.Euler(0, -0.3, 0)

    listener = keyboard.Listener(on_press)
    listener.start()

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

        action, direction, image = ai_client.update(image)
        state = get_state()

        # print(f"Frame: {frame}, Action: {action}, wait: {wait}")

        frame += 1

        if disabled:
            cv2.putText(image, "Robot Disabled - Press Enter to Enable", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        if not safe_mode:
            cv2.putText(image, "Safe Mode OFF", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        if following:
            cv2.putText(image, "Following Mode ON", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        if state.uwb_active:
            cv2.putText(image, "UWB Active", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Display image
        cv2.imshow(window_name, image)

        # Press ESC to stop
        if action == "quit":
            break

        if disabled or state.uwb_active:
            cv2.waitKey(1)
            continue

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

        # print(f"Action: {action}, wait: {wait}")

        if action == ".":
            reset_pose()
            wait = (0, "done")
            continue

        if wait[0] > 0:
            wait = (wait[0]-1, wait[1])
            continue

        if wait[0] == 0 and wait[1] != "done" and wait[1] != "following":
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
                reset_pose()
                sport_client.Euler(0, -0.3, 0)


            continue


        if following:
            if wait[1] != "following":
                sport_client.ClassicWalk(True)
                following_state = {"phase": "move", "count": 0}
                wait = (0, "following")

            if following_state is None:
                following_state = {"phase": "move", "count": 0}

            if not direction:
                print("No direction detected!!!!")
                if following_state["phase"] == "move":
                    sport_client.StopMove()
                    following_state = {"phase": "wait", "count": 0}
                continue

            (x, y, z), (px, py, width_ratio) = direction

            def clamp_speed(v, limit=0.5):
                return max(-limit, min(limit, v))

            y -= 60  # compensate for small downward bias

            YAWGAIN = 2.5
            YAW_DEADBAND = 0.05

            hand_ratio = width_ratio if width_ratio is not None else None

            forward_from_pitch = 0.0
            forward_from_dist = 0.0
            if hand_ratio is not None:
                target_ratio = 0.15
                dist_err = target_ratio - hand_ratio
                forward_from_dist = clamp_speed(dist_err * 2.0)

            vx = clamp_speed(forward_from_pitch + forward_from_dist)
            vy = clamp_speed(-x / 40.0)

            center_err = px - 0.5
            if abs(center_err) < YAW_DEADBAND:
                center_err = 0.0
            vyaw = clamp_speed(-center_err * YAWGAIN)

            ratio_text = f"{hand_ratio:.3f}" if hand_ratio is not None else "none"

            print(
                f"follow: x={x:.2f}, y={y:.2f}, px={px:.2f}, size={ratio_text} -> "
                f"vx={vx:.2f} (pitch {forward_from_pitch:.2f} + dist {forward_from_dist:.2f}), "
                f"vy={vy:.2f}, vyaw={vyaw:.2f}, phase={following_state['phase']}, count={following_state['count']}"
            )

            if following_state["phase"] == "move":
                sport_client.Move(vx, vy, vyaw)
                following_state["count"] += 1
                if following_state["count"] >= 5:
                    sport_client.StopMove()
                    following_state = {"phase": "wait", "count": 0}
            else:  # wait phase
                following_state["count"] += 1
                if following_state["count"] >= 5:
                    following_state = {"phase": "move", "count": 0}

            continue

        elif wait[1] == "following":
            print("Exiting following mode, stopping movement")
            sport_client.StopMove()
            wait = (10, "reset")
            following_state = None
            continue
        else:
            following_state = None

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
            if safe_mode:
                print("Safe mode is ON, skipping WalkUpright command")
            else:
                cv2.waitKey(1)
                sport_client.WalkUpright(True)
                wait = (50, "walk_upright_off")

        elif action == "dislike" or action == "k":
            if safe_mode:
                print("Safe mode is ON, skipping HandStand command")
            else:
                cv2.waitKey(1)
                sport_client.HandStand(True)
                wait = (50, "hand_stand_off")

        elif action == "three_gun" or action == "l":
            if safe_mode:
                print("Safe mode is ON, skipping LeftFlip command")
            else:
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
            if safe_mode:
                print("Safe mode is ON, skipping FrontPounce command")
            else:
                cv2.waitKey(1)
                sport_client.FrontPounce()

        elif action == "peace_inverted" or action == "t":
            if safe_mode:
                print("Safe mode is ON, skipping FrontJump command")
            else:
                cv2.waitKey(1)
                sport_client.FrontJump()

        elif action == "grabbing" or action == "z":
            cv2.waitKey(1)
            sport_client.Stretch()

        elif action != None:
            print(f"Unknown action: {action}")

        if wait[0] == 0 and action == None:
            # Look slightly up when idle
            sport_client.Euler(0, -0.3, 0)

    listener.stop()
    listener.join()

    cv2.destroyWindow(window_name)

    # TODO: Hand following
    # TODO: Better UI
    # TODO: Wifi?
