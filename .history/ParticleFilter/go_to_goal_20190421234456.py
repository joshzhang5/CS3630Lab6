# Jiaxi Zhang
# George McAlear

try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass

from skimage import color

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
import asyncio

import cozmo
from cozmo.util import distance_mm, degrees, speed_mmps, Pose

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

class CozmoWarehouseWorker:

    def __init__(self, robot: cozmo.robot.Robot):
        self.current_arena_pose = None
        self.current_robot_pose = robot.pose
        self.robot = robot
        # start streaming
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

        self.pick_up_pose = Pose(x=4.5, y=13.75, z=0, angle_z=degrees(90))
        self.drop_off_pose = Pose(x=21.75, y=13.75, z=0, angle_z=degrees(90))
       
        self.drop_off_directions = [Pose(x=3, y=4.5, z=0, angle_z=degrees(0)), Pose(x=21.75, y=4.5, z=0, angle_z=degrees(90)), self.drop_off_pose]
        self.pick_up_directions = [Pose(x=21.75, y=4.5, z=0, angle_z=degrees(90)), Pose(x=3, y=4.5, z=0, angle_z=degrees(0)), self.pick_up_pose]

        self.drive_speed = speed_mmps(50)
        print("Robot initialized!")
        self.grid = CozGrid("map_arena.json")
        self.pf = ParticleFilter(self.grid)
        print("Robot initialized!")
        threading.Thread(target=self.runGUI).start()


    def runGUI(self):
        self.gui = GUIWindow(self.grid, show_camera=True)
        self.gui.show_particles(self.pf.particles)
        self.gui.show_mean(0, 0, 0)
        self.gui.start()

    async def drive_to(self, directions):
        print("-" * 20 + "DRIVING" + "-" * 20)
        if isinstance(directions, (list,)):
            for pose in directions:
                await self.__drive_to_pose(pose)
        else:
            await self.__drive_to_pose(directions)

    async def __drive_to_pose(self, pose):
        print("We are at ", self.current_arena_pose, " and we are driving to ", pose)
        directions = pose - self.current_arena_pose
        print("We will follow these directions: ", directions)
        await self.__execute_directions(directions) 
        print("Directions followed!")       

    def update_current_arena_pose(self):
        print("-" * 20 + "UPDATING POSE" + "-" * 20)
        coordinate_systems_diff = diff_heading_deg(self.current_robot_pose.rotation.degrees, self.current_arena_pose.rotation.degrees)
        arena_initial_pose_mm = rotate_point(self.current_robot_pose.position.x, self.current_robot_pose.position.y, coordinate_systems_diff)
        arena_final_pose_mm = rotate_point(self.robot.pose.position.x, self.robot.pose.position.y, coordinate_systems_diff)
        print("We think we moved ", convertPoseFromMmToInches(arena_final_pose_mm - arena_initial_pose_mm))
        self.current_arena_pose = self.current_arena_pose + convertPoseFromMmToInches(arena_final_pose_mm - arena_initial_pose_mm)
        print("Current pose is now ", self.current_arena_pose)
    
    
    async def pick_up_cube(self, tries=5):
        print("-" * 20 + "GETTING CUBE" + "-" * 20)
        cube = await self.robot.world.wait_for_observed_light_cube(timeout=30)
        print("Found cube: %s" % cube)
        picked_up_cube = await self.robot.pickup_object(cube, num_retries=tries).wait_for_completed().obj
        if (picked_up_cube == None):
            print("Could not get the cube.")
            await self.robot.say_text("Help me!").wait_for_completed()
            asyncio.sleep(5)
        else:
            print("Picked up cube!")
       
    
    async def set_down_cube(self):
        print("-" * 20 + "SETTING DOWN CUBE" + "-" * 20)
        await self.robot.set_lift_height(0.0).wait_for_completed()
        await self.robot.set_head_angle(degrees(3)).wait_for_completed()
   

      
    async def __execute_directions(self, directions):
        print("Current arena pose is:", self.current_arena_pose)
        await self.robot.turn_in_place(angle = -1 * self.current_arena_pose.rotation.angle_z).wait_for_completed()
        print("ROBOT is at AFTER TURNING to be parallel to X: ", self.robot.pose)
        await self.robot.drive_straight(distance=distance_mm(directions.position.x * self.grid.scale), speed=self.drive_speed).wait_for_completed()     
        print("ROBOT is at AFTER DRIVING in the X direction: ", self.robot.pose)
        await self.robot.turn_in_place(angle=degrees(90)).wait_for_completed()
        print("ROBOT is at AFTER TURNING to be parallel to Y: ", self.robot.pose)
        await self.robot.drive_straight(distance=distance_mm(directions.position.y * self.grid.scale), speed=self.drive_speed).wait_for_completed()
        print("ROBOT is at AFTER DRIVING in the Y direction: ", self.robot.pose)
        await self.robot.turn_in_place(angle=directions.rotation.angle_z).wait_for_completed()
        
    async def localize(self, turn_angle=20):   
        print("-" * 20 + "LOCALIZING" + "-" * 20)
        # reset our location estimates
        conf = False
        self.current_arena_pose = Pose(0,0,0,angle_z=degrees(0))
        self.pf = ParticleFilter(self.grid)

        # reset lift and head
        await self.robot.set_lift_height(0.0).wait_for_completed()
        await self.robot.set_head_angle(degrees(3)).wait_for_completed()
    
        while not conf:
            # move a little 
            self.current_robot_pose = self.robot.pose
            await self.robot.turn_in_place(angle=degrees(turn_angle)).wait_for_completed()
            odometry = self.__compute_odometry()
            detected_markers, camera_image = await self.__marker_processing()

            # update, motion, and measurment with the odometry and marker data
            curr_x, curr_y, curr_h, conf = self.pf.update(odometry, detected_markers)
        
            # update gui
            self.gui.show_particles(self.pf.particles)
            self.gui.show_mean(curr_x, curr_y, curr_h)
            self.gui.show_camera_image(camera_image) 
            self.gui.updated.set()

        self.current_arena_pose = Pose(curr_x , curr_y, 0, angle_z=degrees(curr_h))
        print("We localized to arena location ", self.current_arena_pose)
    
    def __compute_odometry(self, cvt_inch=True):
        '''
        Compute the odometry given the current pose of the robot (use robot.pose)

        Input:
            - curr_pose: a cozmo.robot.Pose representing the robot's current location
            - cvt_inch: converts the odometry into grid units
        Returns:
            - 3-tuple (dx, dy, dh) representing the odometry
        '''
        last_x, last_y, last_h = self.current_robot_pose.position.x, self.current_robot_pose.position.y, \
            self.current_robot_pose.rotation.angle_z.degrees
        curr_x, curr_y, curr_h = self.robot.pose.position.x, self.robot.pose.position.y, \
            self.robot.pose.rotation.angle_z.degrees
        
        dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
        if cvt_inch:
            dx, dy = dx / self.grid.scale, dy / self.grid.scale

        return (dx, dy, diff_heading_deg(curr_h, last_h))


    async def __marker_processing(self, show_diagnostic_image=False):
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
        # Wait for the latest image from Cozmo
        image_event = await self.robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

        # Convert the image to grayscale
        image = np.array(image_event.image)
        image = color.rgb2gray(image)
        
        # Detect the markers
        markers, diag = detect.detect_markers(image, self.camera_settings, include_diagnostics=True)

        # Measured marker list for the particle filter, scaled by the grid scale
        marker_list = [marker['xyh'] for marker in markers]
        marker_list = [(x/self.grid.scale, y/self.grid.scale, h) for x,y,h in marker_list]

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

async def run(robot: cozmo.robot.Robot):
    cosimo = CozmoWarehouseWorker(robot)
    await cosimo.localize()
    await cosimo.drive_to(cosimo.pick_up_pose)
    while True:
        await cosimo.pick_up_cube(tries=5)
        await cosimo.drive_to(cosimo.drop_off_directions)
        await cosimo.set_down_cube()
        await cosimo.drive_to(cosimo.pick_up_directions)
    
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
