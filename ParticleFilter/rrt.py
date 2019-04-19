import cozmo
import math
import sys
import time

from cmap import *
from gui import *
from utils import *
import numpy as np
import random
import asyncio
MAX_NODES = 20000

def step_from_to(node0, node1, limit=75):
    ########################################################################
    # TODO: please enter your code below.
    # 1. If distance between two nodes is less than limit, return node1
    # 2. Otherwise, return a node in the direction from node0 to node1 whose
    #    distance to node0 is limit. Recall that each iteration we can move
    #    limit units at most
    # 3. Hint: please consider using np.arctan2 function to get vector angle
    # 4. Note: remember always return a Node object
    dist = get_dist(node0, node1)
    if dist < limit:
        return node1
    else:
        #difference angle
        theta = np.arctan2(node1.y - node0.y, node1.x - node0.x)
        # directed angle
        #theta = np.arctan2(node1.y, node1.x) - np.arctan2(node0.y, node0.x)
        #if theta < 0:
        #    theta = theta + 2 * np.pi
        newY = node0.y + (limit*(np.sin(theta)))
        newX = node0.x + (limit*(np.cos(theta)))
        return Node((newX, newY))
    ############################################################################


def node_generator(cmap):
    rand_node = None
    ############################################################################
    # TODO: please enter your code below.
    # 1. Use CozMap width and height to get a uniformly distributed random node
    # 2. Use CozMap.is_inbound and CozMap.is_inside_obstacles to determine the
    #    legitimacy of the random node.
    # 3. Note: remember always return a Node object
    ############################################################################
    max_width, max_height = cmap.get_size()
    rand_node = Node((random.uniform(0, max_width), random.uniform(0, max_height)))
    while not cmap.is_inbound(rand_node) and not cmap.is_inside_obstacles(rand_node):
        rand_node = Node((random.uniform(0, max_width), random.uniform(0, max_height)))
    return_goal = random.uniform(0, 1) < 0.05
    return random.choice(cmap.get_goals()) if return_goal else rand_node


def RRT(cmap, start):
    cmap.add_node(start)
    map_width, map_height = cmap.get_size()
    while (cmap.get_num_nodes() < MAX_NODES):
        ########################################################################
        # TODO: please enter your code below.
        # 1. Use CozMap.get_random_valid_node() to get a random node. This
        #    function will internally call the node_generator above
        # 2. Get the nearest node to the random node from RRT
        # 3. Limit the distance RRT can move
        # 4. Add one path from nearest node to random node
        #
        rand_node = cmap.get_random_valid_node()
        rand_node = Node(rand_node.coord)
        min_dist = 1e9
        for node in cmap.get_nodes():
            dist = get_dist(rand_node, node)
            if min_dist > dist:
                min_dist = dist
                nearest_node = node
        rand_node = step_from_to(nearest_node, rand_node,75)
        ########################################################################
        time.sleep(0.01)
        cmap.add_path(nearest_node, rand_node)
        if cmap.is_solved():
            break

    path = cmap.get_path()
    smoothed_path = cmap.get_smooth_path()

    if cmap.is_solution_valid():
        print("A valid solution has been found :-) ")
        print("Nodes created: ", cmap.get_num_nodes())
        print("Path length: ", len(path))
        print("Smoothed path length: ", len(smoothed_path))
        return smoothed_path
    else:
        print("Please try again :-(")


async def CozmoPlanning(startNode, goalNode, heading, robot: cozmo.robot.Robot):
    # Allows access to map and stopevent, which can be used to see if the GUI
    # has been closed by checking stopevent.is_set()
    global cmap, stopevent
    
    # reset lift and head
    await robot.set_head_angle(cozmo.util.degrees(3)).wait_for_completed()
    await robot.set_lift_height(0.0).wait_for_completed()
    
    cmap.set_start(startNode)
    # Initialize robot position and start_node
    cosimo = CozmoExplorer(robot, x_0=startNode.x, y_0=startNode.y, theta_0=heading)
    await cosimo.go_to_goal()        

class CozmoExplorer:

    def __init__(self, robot, x_0=0, y_0=0, theta_0=0, map_scale=25):
        self.DRIVE_SPEED = cozmo.util.Speed(speed_mmps=50)
        self.robot = robot
        self.last_pose = robot.pose
        self.last_arena_pose = cozmo.util.Pose(x_0 * map_scale, y_0 * map_scale, 0, angle_z=cozmo.util.Angle(degrees=theta_0))
        self.goal_node = None 
        self.blocks_seen = {}
    
    async def go_to_goal(self, goal_node=None):
        global cmap
        if goal_node == None:
            if (self.goal_node == None):
                raise Exception("No goal implicitly or explicitly specified!")
            goal_node = self.goal_node
       
        cmap.reset()
        cmap.add_goal(goal_node)
        cmap.set_start(pose_to_node(self.last_arena_pose))
        path = RRT(cmap, cmap.get_start())
        print("Goal Node:", goal_node.x, ",", goal_node.y)
        for node in path:
            success = await self.__go_to_node(node)
            if not success:
                print("Didn't reach a target node, fuck")
                return False
        print("Reached goal!")
        return True

    async def __go_to_node(self, target_node):
        # Turn cozmo to point towards the goal node
        drive_distance, turn_angle = self.__calculate_movement_vector(target_node)
        print("Turning ", turn_angle, " radians to face ", target_node)
        await self.robot.turn_in_place(angle=turn_angle).wait_for_completed()
        await self.robot.drive_straight(speed=self.DRIVE_SPEED, distance=drive_distance).wait_for_completed()
        self.update_pose()
        
    async def __detect_obstacles(self):
        # Wait until we start driving
        #asyncio.sleep(0.5)

        #while self.robot.are_wheels_moving:
           # await asyncio.sleep(0.25)
            #print("robot is moving")
        updated, goal_center_node = await detect_cube_and_update_cmap(self.robot, self.blocks_seen, pose_to_node(self.__get_current_arena_pose()))
            # if cmap was updated there was an obstacle added
        if updated:
            print("Obstacle detected!")
            self.robot.stop_all_motors()
        if goal_center_node:
            self.goal_node = goal_center_node
        return updated

    async def look_around_for_goal(self):
        if (self.goal_node != None):
            return True
        # do a full turn looking for goal node
        TURN_ANGLE = cozmo.util.Angle(degrees=45)
        for i in range(18):
            await self.robot.turn_in_place(TURN_ANGLE).wait_for_completed()
            self.update_pose()
            if await self.look_for_goal():
                break
        return not self.goal_node == None

    async def look_for_goal(self):
        if (self.goal_node != None):
            return True
        cmap.clear_goals()
        _, goal_center_node = await detect_cube_and_update_cmap(self.robot, self.blocks_seen, pose_to_node(self.last_arena_pose))
        if goal_center_node:
            self.goal_node = goal_center_node
            print("Goal was spotted at (", self.goal_node.x, "," ,self.goal_node.y, ")!")
        return not self.goal_node == None
                
    def __get_current_arena_pose(self):
        movement_since_last_node = self.robot.pose - self.last_pose
        curr_arena_pose = self.last_arena_pose + movement_since_last_node
        print(curr_arena_pose)
        return curr_arena_pose

    def update_pose(self):
        movement_since_last_node = self.robot.pose - self.last_pose
        curr_arena_pose = self.last_arena_pose + movement_since_last_node
        self.last_pose = self.robot.pose
        self.last_arena_pose = curr_arena_pose
    
    def __calculate_movement_vector(self, target_node):
        current_node = pose_to_node(self.last_arena_pose)
        target_angle = np.arctan2(target_node.y - current_node.y, target_node.x - current_node.x)
        turn_angle = target_angle - self.last_arena_pose.rotation.angle_z.radians
        dist = get_dist(current_node, target_node)
        return cozmo.util.distance_mm(dist), cozmo.util.Angle(radians=turn_angle)

def pose_to_node(pose):
    return Node((pose.position.x, pose.position.y))

def get_global_node(local_angle, local_origin, node):
    """Helper function: Transform the node's position (x,y) from local coordinate frame specified by local_origin and local_angle to global coordinate frame.
                        This function is used in detect_cube_and_update_cmap()
        Arguments:
        local_angle, local_origin -- specify local coordinate frame's origin in global coordinate frame
        local_angle -- a single angle value
        local_origin -- a Node object

        Outputs:
        new_node -- a Node object that decribes the node's position in global coordinate frame
    """
    ########################################################################
    # TODO: please enter your code below.
    rotated_point = rotate_point(node.x, node.y, local_angle)
    global_point = local_origin.x + rotated_point[0], local_origin.y + rotated_point[1]
    new_node = Node(global_point)
    return new_node

async def detect_cube_and_update_cmap(robot, marked, cozmo_pos):
    """Helper function used to detect obstacle cubes and the goal cube.
       1. When a valid goal cube is detected, old goals in cmap will be cleared and a new goal corresponding to the approach position of the cube will be added.
       2. Approach position is used because we don't want the robot to drive to the center position of the goal cube.
       3. The center position of the goal cube will be returned as goal_center.

        Arguments:
        robot -- provides the robot's pose in G_Robot
                 robot.pose is the robot's pose in the global coordinate frame that the robot initialized (G_Robot)
                 also provides light cubes
        cozmo_pose -- provides the robot's pose in G_Arena
                 cozmo_pose is the robot's pose in the global coordinate we created (G_Arena)
        marked -- a dictionary of detected and tracked cubes (goal cube not valid will not be added to this list)

        Outputs:
        update_cmap -- when a new obstacle or a new valid goal is detected, update_cmap will set to True
        goal_center -- when a new valid goal is added, the center of the goal cube will be returned
    """
    global cmap

    # Padding of objects and the robot for C-Space
    cube_padding = 60.
    cozmo_padding = 100.

    # Flags
    update_cmap = False
    goal_center = None

    # Time for the robot to detect visible cubes
    time.sleep(1)

    for obj in robot.world.visible_objects:

        if obj.object_id in marked:
            continue

        # Calculate the object pose in G_Arena
        # obj.pose is the object's pose in G_Robot
        # We need the object's pose in G_Arena (object_pos, object_angle)
        dx = obj.pose.position.x - robot.pose.position.x
        dy = obj.pose.position.y - robot.pose.position.y

        object_pos = Node((cozmo_pos.x+dx, cozmo_pos.y+dy))
        object_angle = obj.pose.rotation.angle_z.radians

        # The goal cube is defined as robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id
        # CHANGE BACK TO 1
        if robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id == obj.object_id:

            # Calculate the approach position of the object
            local_goal_pos = Node((0, -cozmo_padding))
            goal_pos = get_global_node(object_angle, object_pos, local_goal_pos)

            # Check whether this goal location is valid
            if cmap.is_inside_obstacles(goal_pos) or (not cmap.is_inbound(goal_pos)):
                print("The goal position is not valid. Please remove the goal cube and place in another position.")
            else:
                cmap.clear_goals()
                cmap.add_goal(goal_pos)
                goal_center = object_pos

        # Define an obstacle by its four corners in clockwise order
        obstacle_nodes = []
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, cube_padding))))
        cmap.add_obstacle(obstacle_nodes)
        marked[obj.object_id] = obj
        update_cmap = True

    return update_cmap, goal_center


class RobotThread(threading.Thread):
    """Thread to run cozmo code separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        # Please refrain from enabling use_viewer since it uses tk, which must be in main thread
        cozmo.run_program(CozmoPlanning,use_3d_viewer=False, use_viewer=False)
        stopevent.set()


class RRTThread(threading.Thread):
    """Thread to run RRT separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        while not stopevent.is_set():
            RRT(cmap, cmap.get_start())
            time.sleep(100)
            cmap.reset()
        stopevent.set()


if __name__ == '__main__':
    global cmap, stopevent
    stopevent = threading.Event()
    robotFlag = False
    for i in range(0,len(sys.argv)):
        if (sys.argv[i] == "-robot"):
            robotFlag = True
    if (robotFlag):
        cmap = CozMap("maps/emptygrid.json", node_generator)
        robot_thread = RobotThread()
        robot_thread.start()
    else:
        cmap = CozMap("maps/map1.json", node_generator)
        sim = RRTThread()
        sim.start()
    visualizer = Visualizer(cmap)
    visualizer.start()
    stopevent.set();