# Jiaxi Zhang
# George McAlear


# If you run into an "[NSApplication _setup] unrecognized selector" problem on macOS,
# try uncommenting the following snippet

try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass

from skimage import color
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
from PIL import Image

from markers import detect, annotator

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *

from time import sleep
import time

from cozmo.util import distance_mm, degrees, speed_mmps

from rrt import *

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

# tmp cache
flag_odom_init = False

# map


    
global camera_settings
# pick up location for the robot to drive to, (x, y, theta)
async def run(robot: cozmo.robot.Robot):
   
    global grid, gui, pf
    ###################
   
    # pickup point
    #pickup_node = Node((153, 240))
    # dropoff_node = Node((544, 344))
    # localize the robot
    await look_around_until_converge(robot)

    # intialize an explorer after localized
    #cosimo = CozmoExplorer(robot, x_0=last_pose.position.x, y_0=last_pose.position.y, theta_0=last_pose.rotation.angle_z.radians)
    # move robot to pickup zone once localized
    print("LAST POSE IS:", last_pose)
    #print("COZMO CONVERTED THAT TO A START AT:", cosimo.last_arena_pose)
    directions = goal_pose - last_pose
    current_pose = last_pose
    last_robot_pose = robot.pose
    print("SETTING LAST ROBOT POSE TO: ", last_robot_pose)
    print("SO WE GOING TO FOLLOW THIS TO PICKUP ZONE:", directions)
    
    await execute_directions(robot, directions)
    await robot.turn_in_place(angle=cozmo.util.Angle(degrees=45)).wait_for_completed()
    print("LAST ROBOT POSE IS: ", last_robot_pose)
    print("CURRENT POSE IS:", robot.pose)
    print("WE THINK WE MOVED THIS MUCH TO GO TO PICKUP ZONE: ", convertPoseToInches(robot.pose - last_robot_pose))
    current_pose = current_pose + convertPoseToInches(rotate_point(robot.pose, - last_robot_pose)
    last_robot_pose = robot.pose
    print("COZMO THINKS IT IS AT AFTER DRIVING TO PICKUPZONE: ", current_pose)
    
    # await robot.say_text('Ready for pick up!').wait_for_completed()
    drop_off_directions = [(3, 4.5, 0), (21.75, 4.5, 90), (21.75, 13.75, 90)]
    pick_up_directions = [(21.75, 4.5, 90), (3, 4.5, 0), (4.5, 20)]
    while True:
      
      cube = await robot.world.wait_for_observed_light_cube(timeout=30)
      print("Found cube: %s" % cube)
    
      await robot.pickup_object(cube, num_retries=5).wait_for_completed()
      current_pose = current_pose + convertPoseToInches(robot.pose - last_robot_pose)
      print("WE THINK WE MOVED THIS MUCH TO PICK UP CUBE: ", convertPoseToInches(robot.pose - last_robot_pose))
      last_robot_pose = robot.pose
      #cosimo.update_pose()
        
      print("COZMO THINKS IT IS AT AFTER PICKING UP CUBE: ", current_pose)

      #await look_around_until_converge(robot)

        # intialize an explorer after localized
        #cosimo = CozmoExplorer(robot, x_0=last_pose.position.x, y_0=last_pose.position.y, theta_0=last_pose.rotation.angle_z.radians)
        # move robot to pickup zone once localized
      
        #print("COZMO CONVERTED THAT TO A START AT:", cosimo.last_arena_pose)
      #current_pose = last_pose
      
      # rrt to drop zone and drop off cube
      for destination in drop_off_directions:
        directions = convertInchesToPose(destination) - current_pose
        await execute_directions(robot,directions)
        current_pose = current_pose + convertPoseToInches(robot.pose - last_robot_pose)
        print("WE THINK WE MOVED THIS MUCH TO FOLLOW DIRECTIONS: ", convertPoseToInches(robot.pose - last_robot_pose))
        last_robot_pose = robot.pose
        print("COZMO THINKS IT IS AT AFTER FOLLOWING DIRECTIONS: ", current_pose)
      #await cosimo.go_to_goal(goal_node=dropoff_node)  
      await robot.set_lift_height(0.0).wait_for_completed()


      
      # rrt to just in front of pick up zone
      # await cosimo.go_to_goal(goal_node=pickup_node)  


def CozmoWarehouseWorker:

    def __init__(self, robot, current_arena_pose, ):
        self.current_arena_pose = current_arena_pose
        self.last_robot_pose = robot.pose
        self.robot = robot
        # start streaming
        await robot.set_head_angle(cozmo.util.degrees(3)).wait_for_completed()
        robot.camera.image_stream_enabled = True
        robot.camera.color_image_enabled = False
        robot.camera.enable_auto_exposure()
      
        # Obtain the camera intrinsics matrix
        fx, fy = robot.camera.config.focal_length.x_y
        cx, cy = robot.camera.config.center.x_y
        self.camera_settings = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float)

        self.grid = CozGrid("map_arena.json")
        self.pf = ParticleFilter(self.grid)
        self.gui = GUIWindow(self.grid, show_camera=True)

    async def execute_directions(directions):
        print("Robot is at: ", self.robot.pose)
        await self.robot.turn_in_place(angle=directions.rotation.angle_z).wait_for_completed()
        print("ROBOT is at AFTER TURNING to be parallel to X: ", self.robot.pose)
        await self.robot.drive_straight(distance=distance_mm(directions.position.x * grid.scale), speed=speed_mmps(80)).wait_for_completed()     
        print("ROBOT is at AFTER DRIVING in the X direction: ", self.robot.pose)
        await self.robot.turn_in_place(angle=cozmo.util.Angle(degrees=90)).wait_for_completed()
        print("ROBOT is at AFTER TURNING to be parallel to Y: ", self.robot.pose)
        await self.robot.drive_straight(distance=distance_mm(directions.position.y * grid.scale), speed=speed_mmps(80)).wait_for_completed()
        print("ROBOT is at AFTER DRIVING in the Y direction: ", self.robot.pose)
        
    async def localize(self):   

        # reset our location estimates
        conf = False
        self.current_arena_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
        self.pf = ParticleFilter(grid)

        # reset lift and head
        await self.robot.set_lift_height(0.0).wait_for_completed()
        await self.robot.set_head_angle(cozmo.util.degrees(3)).wait_for_completed()
    
        while not conf:
            # move a little 
            self.last_robot_pose = self.robot.pose
            await self.robot.turn_in_place(angle=cozmo.util.Angle(degrees=20)).wait_for_completed()
            odometry = compute_odometry()
            detected_markers, camera_image = await marker_processing()

            # update, motion, and measurment with the odometry and marker data
            curr_x, curr_y, curr_h, conf = pf.update(odometry, detected_markers)
        
            # update gui
            self.gui.show_particles(pf.particles)
            self.gui.show_mean(curr_x, curr_y, curr_h)
            self.gui.show_camera_image(camera_image) 
            self.gui.updated.set()

        self.current_arena_pose = cozmo.util.Pose(curr_x , curr_y, 0, angle_z=cozmo.util.Angle(degrees=curr_h))

    def compute_odometry(self, cvt_inch=True):
        '''
        Compute the odometry given the current pose of the robot (use robot.pose)

        Input:
            - curr_pose: a cozmo.robot.Pose representing the robot's current location
            - cvt_inch: converts the odometry into grid units
        Returns:
            - 3-tuple (dx, dy, dh) representing the odometry
        '''
        last_x, last_y, last_h = self.last_robot_pose.position.x, self.last_robot_pose.position.y, \
            self.last_robot_pose.rotation.angle_z.degrees
        curr_x, curr_y, curr_h = self.robot.pose.position.x, self.robot.pose.position.y, \
            self.robot.pose.rotation.angle_z.degrees
        
        dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
        if cvt_inch:
            dx, dy = dx / grid.scale, dy / grid.scale

        return (dx, dy, diff_heading_deg(curr_h, last_h))


    async def marker_processing(self, show_diagnostic_image=False):
        '''
        Obtain the visible markers from the current frame from Cozmo's camera. 
        Since this is an async function, it must be called using await, for example:

            markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

        Input:
            - robot: cozmo.robot.Robot object
            - camera_settings: 3x3 matrix representing the camera calibration settings
            - show_diagnostic_image: if True, shows what the marker detector sees after processing
        Returns:
            - a list of detected markers, each being a 3-tuple (rx, ry, rh) 
            (as expected by the particle filter's measurement update)
            - a PIL Image of what Cozmo's camera sees with marker annotations
        '''

        global grid

        # Wait for the latest image from Cozmo
        image_event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

        # Convert the image to grayscale
        image = np.array(image_event.image)
        image = color.rgb2gray(image)
        
        # Detect the markers
        markers, diag = detect.detect_markers(image, self.camera_settings, include_diagnostics=True)

        # Measured marker list for the particle filter, scaled by the grid scale
        marker_list = [marker['xyh'] for marker in markers]
        marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]

        # Annotate the camera image with the markers
        if not show_diagnostic_image:
            annotated_image = image_event.image.resize((image.shape[1] * 2, image.shape[0] * 2))
            annotator.annotate_markers(annotated_image, markers, scale=2)
        else:
            diag_image = color.gray2rgb(diag['filtered_image'])
            diag_image = Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
            annotator.annotate_markers(diag_image, markers, scale=2)
            annotated_image = diag_image

        return marker_list, annotated_image

class CozmoThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)

if __name__ == '__main__':
    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()