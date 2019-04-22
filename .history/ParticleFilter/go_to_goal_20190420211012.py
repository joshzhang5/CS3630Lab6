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
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)

def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a cozmo.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees
    
    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))


async def marker_processing(robot, camera_settings, show_diagnostic_image=False):
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
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)

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
    
global camera_settings

# pick up location for the robot to drive to, (x, y, theta)
goal_pose = cozmo.util.Pose(6, 12, 0, angle_z=cozmo.util.Angle(degrees=0))
current_pose = None
async def run(robot: cozmo.robot.Robot):
    global flag_odom_init, last_pose, goal_pose
    global grid, gui, pf
    global camera_settings
    # start streaming
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(3)).wait_for_completed()

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)

    ###################
   
    # pickup point
    #pickup_node = Node((153, 240))
    # dropoff_node = Node((544, 344))
    # localize the robot
    await look_around_until_converge(robot, 12)

    # intialize an explorer after localized
    #cosimo = CozmoExplorer(robot, x_0=last_pose.position.x, y_0=last_pose.position.y, theta_0=last_pose.rotation.angle_z.radians)
    # move robot to pickup zone once localized
    print("LAST POSE IS:", last_pose)
    #print("COZMO CONVERTED THAT TO A START AT:", cosimo.last_arena_pose)
    directions = goal_pose - last_pose
    current_pose = last_pose
    last_robot_pose = robot.pose
    print("SO WE GOING TO FOLLOW THIS TO PICKUP ZONE:", directions)
    await execute_directions(robot, directions)
    await robot.turn_in_place(angle=cozmo.util.Angle(degrees=45)).wait_for_completed()
    current_pose = current_pose + convertPoseToInches(robot.pose - last_robot_pose)
    print("WE THINK WE MOVED THIS MUCH TO GO TO PICKUP ZONE: ", convertPoseToInches(robot.pose - last_robot_pose))
    last_robot_pose = robot.pose

    # recalculate where cosimo is
    #cosimo.update_pose()
    print("COZMO THINKS IT IS AT AFTER DRIVING TO PICKUPZONE: ", current_pose)
    
    await robot.say_text('Ready for pick up!').wait_for_completed()
    
    while True:
      
      cube = await robot.world.wait_for_observed_light_cube(timeout=30)
      print("Found cube: %s" % cube)
    
      await robot.pickup_object(cube, num_retries=5).wait_for_completed()
      current_pose = current_pose + convertPoseToInches(robot.pose - last_robot_pose)
      print("WE THINK WE MOVED THIS MUCH TO PICK UP CUBE: ", convertPoseToInches(robot.pose - last_robot_pose))
      last_robot_pose = robot.pose
      #cosimo.update_pose()
        
      print("COZMO THINKS IT IS AT AFTER PICKING UP CUBE: ", current_pose)
      
      # rrt to drop zone and drop off cube
      
      #await cosimo.go_to_goal(goal_node=dropoff_node)  
      await robot.set_lift_height(0.0).wait_for_completed()
      
        
      # rrt to just in front of pick up zone
      # await cosimo.go_to_goal(goal_node=pickup_node)  
    
def convertPoseToInches(pose):
    pos = pose.position 
    angle = pose.rotation.angle_z
    return cozmo.util.Pose(x=pos.x/25, y=pos.y/25, z=pos.z/25, angle_z=angle)
    
async def execute_directions(robot, directions):
    await robot.turn_in_place(angle=directions.rotation.angle_z).wait_for_completed()
    await robot.drive_straight(distance=distance_mm(directions.position.x * grid.scale), speed=speed_mmps(80)).wait_for_completed()     
    await robot.turn_in_place(angle=cozmo.util.Angle(degrees=90)).wait_for_completed()
    await robot.drive_straight(distance=distance_mm(directions.position.y * grid.scale), speed=speed_mmps(80)).wait_for_completed()
        
async def look_around_until_converge(robot: cozmo.robot.Robot):   
    
    # globals
    global flag_odom_init, last_pose
    global grid, gui, pf

    # reset variables
    conf = False
    last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
    pf = ParticleFilter(grid)

    # reset lift and head
    await robot.set_lift_height(0.0).wait_for_completed()
    await robot.set_head_angle(cozmo.util.degrees(3)).wait_for_completed()
  

    while not conf:
        if (await is_picked_up(robot)):
            continue
        # move a little 
        last_pose = robot.pose
        await robot.turn_in_place(angle=cozmo.util.Angle(degrees=20)).wait_for_completed()
        curr_pose = robot.pose
        detected_markers, camera_image = await marker_processing(robot, camera_settings)
        
        # update, motion, and measurment with the odometry and marker data
        odometry = compute_odometry(curr_pose)
        curr_x, curr_y, curr_h, conf = pf.update(odometry, detected_markers)
       
        # update gui
        gui.show_particles(pf.particles)
        gui.show_mean(curr_x, curr_y, curr_h)
        gui.show_camera_image(camera_image) 
        gui.updated.set()
    last_pose = cozmo.util.Pose(curr_x , curr_y, 0, angle_z=cozmo.util.Angle(degrees=curr_h))
    return last_pose

async def robot_is_at_goal(robot):
    global goal_pose
    await robot.play_anim(name="anim_poked_giggle").wait_for_completed()
    while not robot.is_picked_up:
       await asyncio.sleep(.1)
    await robot.say_text("Put me down!").wait_for_completed()

async def is_picked_up(robot):
    global pf, last_pose
    if robot.is_picked_up:
        await robot.say_text("Put me down!").wait_for_completed()
        last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
        pf = ParticleFilter(grid)
        while (robot.is_picked_up):
            await asyncio.sleep(0.01)
        return True
    return False

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