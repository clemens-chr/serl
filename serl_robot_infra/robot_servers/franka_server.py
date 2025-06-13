"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""
from flask import Flask, request, jsonify, render_template
import numpy as np
import rospy
import time
import subprocess
import os
import signal
import sys
import atexit
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from serl_franka_controllers.msg import ZeroJacobian
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client as ReconfClient

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "172.16.0.2", "IP address of the franka robot's controller box"
)
flags.DEFINE_string(
    "gripper_ip", "192.168.1.114", "IP address of the robotiq gripper if being used"
)
flags.DEFINE_string(
    "gripper_type", "Franka", "Type of gripper to use: Robotiq, Franka, or None"
)
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -1.9, -0, 2, 0],
    "Target joint angles for the robot to reset to",
)


class FrankaServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""

    def __init__(self, robot_ip, gripper_type, ros_pkg_name, reset_joint_target):
        self.robot_ip = robot_ip
        self.ros_pkg_name = ros_pkg_name
        self.reset_joint_target = reset_joint_target
        self.gripper_type = gripper_type

        # Initialize all member variables
        self.pos = np.zeros(7)  # [x, y, z, qx, qy, qz, qw]
        self.vel = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        self.force = np.zeros(3)  # [fx, fy, fz]
        self.torque = np.zeros(3)  # [tx, ty, tz]
        self.q = np.zeros(7)  # joint positions
        self.dq = np.zeros(7)  # joint velocities
        self.jacobian = np.zeros((6, 7))  # jacobian matrix
        self.impedance_running = False
        self.last_update_time = time.time()

        self.eepub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped,
            queue_size=10,
        )
        self.resetpub = rospy.Publisher(
            "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1
        )
        self.jacobian_sub = rospy.Subscriber(
            "/cartesian_impedance_controller/franka_jacobian",
            ZeroJacobian,
            self._set_jacobian,
        )
        self.state_sub = rospy.Subscriber(
            "franka_state_controller/franka_states", FrankaState, self._set_currpos
        )

        self.imp = None
        self.joint_controller = None
        self.cleanup_registered = False

    def get_all_data(self):
        """Returns all robot data as a dictionary"""
        return {
            "pos": self.pos.tolist(),
            "vel": self.vel.tolist(),
            "force": self.force.tolist(),
            "torque": self.torque.tolist(),
            "q": self.q.tolist(),
            "dq": self.dq.tolist(),
            "jacobian": self.jacobian.tolist(),
            "impedance_running": self.impedance_running,
            "last_update_time": self.last_update_time,
            "robot_ip": self.robot_ip,
            "gripper_type": self.gripper_type,
            "reset_joint_target": self.reset_joint_target
        }

    def start_impedance(self):
        """Launches the impedance controller"""
        self.imp = subprocess.Popen(
            [
                "roslaunch",
                self.ros_pkg_name,
                "impedance.launch",
                "robot_ip:=" + self.robot_ip,
                f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
            ],
            stdout=subprocess.PIPE,
        )
        self.impedance_running = True
        time.sleep(5)

    def stop_impedance(self):
        """Stops the impedance controller"""
        self.imp.terminate()
        self.impedance_running = False
        time.sleep(1)

    def clear(self):
        """Clears any errors"""
        msg = ErrorRecoveryActionGoal()
        self.resetpub.publish(msg)

    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        # First Stop impedance
        try:
            self.stop_impedance()
            self.clear()
        except:
            print("impedance Not Running")
        time.sleep(3)
        self.clear()

        # Launch joint controller reset
        # set rosparm with rospkg
        # rosparam set /target_joint_positions '[q1, q2, q3, q4, q5, q6, q7]'
        rospy.set_param("/target_joint_positions", self.reset_joint_target)

        self.joint_controller = subprocess.Popen(
            [
                "roslaunch",
                self.ros_pkg_name,
                "joint.launch",
                "robot_ip:=" + self.robot_ip,
                f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(1)
        print("RUNNING JOINT RESET")
        self.clear()

        # Wait until target joint angles are reached
        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(self.reset_joint_target) - np.array(self.q),
            0,
            atol=1e-2,
            rtol=1e-2,
        ):
            time.sleep(1)
            count += 1
            if count > 30:
                print("joint reset TIMEOUT")
                break

        # Stop joint controller
        print("RESET DONE")
        self.joint_controller.terminate()
        time.sleep(1)
        self.clear()
        print("KILLED JOINT RESET", self.pos)

        # Restart impedece controller
        self.start_impedance()
        print("impedance STARTED")

    def move(self, pose: list):
        """Moves to a pose: [x, y, z, qx, qy, qz, qw]"""
        assert len(pose) == 7
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
        msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
        self.eepub.publish(msg)

    def _set_currpos(self, msg):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        self.pos = pose
        self.dq = np.array(list(msg.dq)).reshape((7,))
        self.q = np.array(list(msg.q)).reshape((7,))
        self.force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self.torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        self.last_update_time = time.time()
        try:
            self.vel = self.jacobian @ self.dq
        except:
            self.vel = np.zeros(6)
            rospy.logwarn(
                "Jacobian not set, end-effector velocity temporarily not available"
            )

    def _set_jacobian(self, msg):
        jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")
        self.jacobian = jacobian

    def cleanup(self):
        """Cleanup function to be called on shutdown"""
        print("\nCleaning up resources...")
        try:
            if self.impedance_running:
                self.stop_impedance()
            if hasattr(self, 'imp') and self.imp:
                self.imp.terminate()
                self.imp.wait(timeout=5)
            if hasattr(self, 'joint_controller') and self.joint_controller:
                self.joint_controller.terminate()
                self.joint_controller.wait(timeout=5)
        except Exception as e:
            print(f"Error during cleanup: {e}")


###############################################################################

def kill_existing_roscore():
    """Kills any existing roscore processes"""
    try:
        # Find roscore processes
        roscore_procs = subprocess.check_output(['pgrep', '-f', 'roscore']).decode().split()
        if roscore_procs:
            print("Found existing roscore processes:", roscore_procs)
            # Kill each roscore process
            for pid in roscore_procs:
                os.kill(int(pid), signal.SIGTERM)
            print("Killed existing roscore processes")
            time.sleep(2)  # Wait for processes to terminate
    except subprocess.CalledProcessError:
        print("No existing roscore processes found")
    except Exception as e:
        print(f"Error while killing roscore: {e}")

def cleanup_resources(robot_server, roscore_process):
    """Cleanup function for all resources"""
    print("\nShutting down gracefully...")
    if robot_server:
        robot_server.cleanup()
    if roscore_process:
        roscore_process.terminate()
        roscore_process.wait(timeout=5)
    kill_existing_roscore()
    print("Cleanup complete. Exiting...")
    sys.exit(0)

def main(_):
    ROS_PKG_NAME = "serl_franka_controllers"
    robot_server = None
    roscore_process = None

    def signal_handler(signum, frame):
        """Handle interrupt signals"""
        print("\nReceived interrupt signal. Starting cleanup...")
        cleanup_resources(robot_server, roscore_process)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    ROBOT_IP = FLAGS.robot_ip
    GRIPPER_IP = FLAGS.gripper_ip
    GRIPPER_TYPE = FLAGS.gripper_type
    RESET_JOINT_TARGET = FLAGS.reset_joint_target

    print('Gripper Type:', GRIPPER_TYPE)
    print('Robot IP:', ROBOT_IP)
    print('Gripper IP:', GRIPPER_IP)
    print('Reset Joint Target:', RESET_JOINT_TARGET)
    print('\n=== Web Interface Access ===')
    print('Local access: http://localhost:5000')
    print('Network access: http://{}:5000'.format(ROBOT_IP))
    print('=======================================\n')

    # Initialize Flask app with proper template and static folders
    webapp = Flask(__name__, 
                   template_folder='templates',
                   static_folder='static')

    # Kill existing roscore and start a new one
    kill_existing_roscore()
    try:
        roscore_process = subprocess.Popen("roscore")
        time.sleep(1)
    except Exception as e:
        raise Exception("Failed to start roscore", e)

    # Start ros node
    rospy.init_node("franka_control_api")

    # Initialize gripper server
    if GRIPPER_TYPE == "Franka":
        from robot_servers.franka_gripper_server import FrankaGripperServer
        gripper_server = FrankaGripperServer()
    elif GRIPPER_TYPE == "None":
        gripper_server = None
    else:
        raise NotImplementedError("Gripper Type Not Implemented")

    # Initialize robot server
    robot_server = FrankaServer(
        robot_ip=ROBOT_IP,
        gripper_type=GRIPPER_TYPE,
        ros_pkg_name=ROS_PKG_NAME,
        reset_joint_target=RESET_JOINT_TARGET,
    )
    robot_server.start_impedance()

    # Register cleanup function
    atexit.register(lambda: cleanup_resources(robot_server, roscore_process))

    reconf_client = ReconfClient(
        "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
    )

    # Route for web interface
    @webapp.route("/", methods=["GET"])
    def web_interface():
        try:
            print(f"Rendering web interface for {GRIPPER_TYPE} gripper")
            return render_template('index.html')
        except Exception as e:
            print(f"Error in web_interface route: {str(e)}")
            return "Error loading web interface", 500

    # Route for getting all data
    @webapp.route("/get_all_data", methods=["GET"])
    def get_all_data():
        try:
            data = robot_server.get_all_data()
            if gripper_server:
                data["gripper_pos"] = gripper_server.gripper_pos
            else:
                data["gripper_pos"] = "N/A"
            return jsonify(data)
        except Exception as e:
            print(f"Error in get_all_data route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Route for Starting impedance
    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        try:
            robot_server.clear()
            robot_server.start_impedance()
            return "Started impedance"
        except Exception as e:
            print(f"Error in start_impedance route: {str(e)}")
            return str(e), 500

    # Route for Stopping impedance
    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        try:
            robot_server.stop_impedance()
            return "Stopped impedance"
        except Exception as e:
            print(f"Error in stop_impedance route: {str(e)}")
            return str(e), 500

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        try:
            return jsonify({"pose": np.array(robot_server.pos).tolist()})
        except Exception as e:
            print(f"Error in get_pos route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pos_euler():
        try:
            r = R.from_quat(robot_server.pos[3:])
            euler = r.as_euler("xyz")
            return jsonify({"pose": np.concatenate([robot_server.pos[:3], euler]).tolist()})
        except Exception as e:
            print(f"Error in get_pos_euler route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        try:
            return jsonify({"vel": np.array(robot_server.vel).tolist()})
        except Exception as e:
            print(f"Error in get_vel route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        try:
            return jsonify({"force": np.array(robot_server.force).tolist()})
        except Exception as e:
            print(f"Error in get_force route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        try:
            return jsonify({"torque": np.array(robot_server.torque).tolist()})
        except Exception as e:
            print(f"Error in get_torque route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        try:
            return jsonify({"q": np.array(robot_server.q).tolist()})
        except Exception as e:
            print(f"Error in get_q route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        try:
            return jsonify({"dq": np.array(robot_server.dq).tolist()})
        except Exception as e:
            print(f"Error in get_dq route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        try:
            return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})
        except Exception as e:
            print(f"Error in get_jacobian route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        try:
            if gripper_server:
                return jsonify({"gripper": gripper_server.gripper_pos})
            else:
                return jsonify({"gripper": "N/A"})
        except Exception as e:
            print(f"Error in get_gripper route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        try:
            robot_server.clear()
            robot_server.reset_joint()
            return "Reset Joint"
        except Exception as e:
            print(f"Error in joint_reset route: {str(e)}")
            return str(e), 500

    # Route for Activating the Gripper
    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        try:
            if gripper_server:
                print("activate gripper")
                gripper_server.activate_gripper()
                return "Activated"
            else:
                return "No gripper available"
        except Exception as e:
            print(f"Error in activate_gripper route: {str(e)}")
            return str(e), 500

    # Route for Resetting the Gripper
    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        try:
            if gripper_server:
                print("reset gripper")
                gripper_server.reset_gripper()
                return "Reset"
            else:
                return "No gripper available"
        except Exception as e:
            print(f"Error in reset_gripper route: {str(e)}")
            return str(e), 500

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        try:
            if gripper_server:
                print("open")
                gripper_server.open()
                return "Opened"
            else:
                return "No gripper available"
        except Exception as e:
            print(f"Error in open_gripper route: {str(e)}")
            return str(e), 500

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        try:
            if gripper_server:
                print("close")
                gripper_server.close()
                return "Closed"
            else:
                return "No gripper available"
        except Exception as e:
            print(f"Error in close_gripper route: {str(e)}")
            return str(e), 500

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        try:
            if gripper_server:
                gripper_pos = request.json
                pos = np.clip(int(gripper_pos["gripper_pos"]), 0, 255)  # 0-255
                print(f"move gripper to {pos}")
                gripper_server.move(pos)
                return "Moved Gripper"
            else:
                return "No gripper available"
        except Exception as e:
            print(f"Error in move_gripper route: {str(e)}")
            return str(e), 500

    # Route for Clearing Errors
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        try:
            robot_server.clear()
            return "Clear"
        except Exception as e:
            print(f"Error in clearerr route: {str(e)}")
            return str(e), 500

    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        try:
            pos = np.array(request.json["arr"])
            print("Moving to", pos)
            robot_server.move(pos)
            return "Moved"
        except Exception as e:
            print(f"Error in pose route: {str(e)}")
            return str(e), 500

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        try:
            return jsonify({
                "pose": np.array(robot_server.pos).tolist(),
                "vel": np.array(robot_server.vel).tolist(),
                "force": np.array(robot_server.force).tolist(),
                "torque": np.array(robot_server.torque).tolist(),
                "q": np.array(robot_server.q).tolist(),
                "dq": np.array(robot_server.dq).tolist(),
                "jacobian": np.array(robot_server.jacobian).tolist(),
                "gripper_pos": gripper_server.gripper_pos if gripper_server else "N/A",
            })
        except Exception as e:
            print(f"Error in getstate route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Route for updating compliance parameters
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        try:
            reconf_client.update_configuration(request.json)
            return "Updated compliance parameters"
        except Exception as e:
            print(f"Error in update_param route: {str(e)}")
            return str(e), 500

    try:
        webapp.run(host="0.0.0.0")
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Starting cleanup...")
        cleanup_resources(robot_server, roscore_process)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        cleanup_resources(robot_server, roscore_process)


if __name__ == "__main__":
    app.run(main)
