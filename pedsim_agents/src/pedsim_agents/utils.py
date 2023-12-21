import dataclasses
import enum
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import pedsim_msgs.msg
import std_msgs.msg
import geometry_msgs.msg

import rospy

# INPUT

InputMsg = pedsim_msgs.msg.PedsimAgentsDataframe

@dataclasses.dataclass
class InputData:
    header: std_msgs.msg.Header
    agents: List[pedsim_msgs.msg.AgentState]
    robots: List[pedsim_msgs.msg.RobotState]
    groups: List[pedsim_msgs.msg.AgentGroup]
    waypoints: List[pedsim_msgs.msg.Waypoint]
    line_obstacles: List[pedsim_msgs.msg.Wall] #TODO rename to walls
    obstacles: List[pedsim_msgs.msg.Obstacle]


# FEEDBACK

@dataclasses.dataclass
class FeedbackDatum:
    feedback: pedsim_msgs.msg.AgentFeedback
    TODO: None = None

FeedbackData = List[FeedbackDatum]
FeedbackMsg = pedsim_msgs.msg.AgentFeedbacks


# SEMANTIC 

@enum.unique
class PedType(enum.IntEnum):
    NONE = 0
    adult = enum.auto()

@enum.unique
class SemanticAttribute(enum.Enum):
    IS_PEDESTRIAN = "pedestrian"
    IS_PEDESTRIAN_MOVING = "pedestrian_moving"
    IS_PEDESTRIAN_TALKING = "pedestrian_talking"
    IS_PEDESTRIAN_IDLING = "pedestrian_idling"
    IS_PEDESTRIAN_INTERACTING = "pedestrian_interacting"
    PEDESTRIAN_VEL_X = "pedestrian_vel_x"
    PEDESTRIAN_VEL_Y = "pedestrian_vel_y"
    PEDESTRIAN_TYPE = "pedestrian_type"

SemanticMsg = pedsim_msgs.msg.SemanticData
SemanticData = Dict[SemanticAttribute, List[Tuple[geometry_msgs.msg.Point, float]]]

T = TypeVar("T")

def NList(l: Optional[List[T]]) -> List[T]:
    return [] if l is None else l

# FUN

class SocialState(enum.IntEnum):
    IDLE = 0
    WALKING = 1
    TALKING = 2
    INTERACTING = 3

class Thingy():
    position: np.ndarray
    interaction_radius: float
    can_interact: bool
    can_talk: bool

    def __init__(self, position: np.ndarray = np.array([0.0,0.0])) -> None:
        self.position = position
        self.interaction_radius = 0.5
        self.can_interact = False
        self.can_talk = False

    def get_position(self) -> np.ndarray:
        return self.position

class Objecty(Thingy):

    def __init__(self, state: pedsim_msgs.msg.Obstacle) -> None:
        super().__init__(np.array([state.pose.position.x, state.pose.position.y]))
        self.can_interact = True
        self.interaction_radius = 1.5
        self.interacting = False
    
    def interact(self):
        self.interacting = True

class Groupy(): #TODO
    agents: List[Thingy]
    position: np.ndarray

    def __init__(self, *agents: Thingy):
        super().__init__()
        self.agents = list(agents)
        self.position = self.meet()
    
    def meet(self) -> np.ndarray: #TODO
        positions = []
        for agent in self.agents:
            positions.append(agent.get_position())
        return np.mean(np.array(positions), axis=0)
    
    def get_position(self):
        return self.position

    def join(self, agent: Thingy):
        self.agents.append(agent)
    
    def leave(self, agent: Thingy): #TODO
        self.agents.remove(agent)
        if len(self.agents) < 2:
            self.agents[0].group = None
            self.agents[0].social_state = SocialState.IDLE
    
class Agenty(Thingy):

    social_state: SocialState
    group: Groupy
    direction: float
    speed: float
    destination: Thingy
    probabilities: Dict[str, float]
    gen: np.random.Generator

    def __init__(self, agent_state: pedsim_msgs.msg.AgentState):
        super().__init__()
        self.can_interact = True

        self.can_talk = True
        self.social_state = SocialState.IDLE

        self.fov = 100
        self.range = 10
        self.interaction_radius = 1

        self.group = None

        self.gen = np.random.default_rng()
        self.probabilities = {"walk random": 0.01,
                            "interact with": 0.01,
                            "stop talking": 0.01,
                            "stop interacting": 0.01}
        
        self.update(agent_state)
        self.destination = Thingy(self.get_position())

    def update(self, agent_state: pedsim_msgs.msg.AgentState):
        self.position = np.array([agent_state.pose.position.x, agent_state.pose.position.y])
        self.direction = np.arctan2(agent_state.acceleration.y, agent_state.acceleration.x)
        self.speed = np.linalg.norm(np.array([agent_state.acceleration.x, agent_state.acceleration.y]))
    
    def check_probability(self, event: str):
        return False if event not in self.probabilities else self.gen.random() < self.probabilities[event]
    
    def next_state(self, agents: List[Thingy]): 
        if self.social_state == SocialState.IDLE:
            things = self.see(agents)
            np.random.shuffle(things)
            if things == []:
                if self.check_probability("walk random"):
                    self.destination = World.random_thing(self.get_position(), 3, 5, self.direction, self.fov)
                    self.social_state = SocialState.WALKING
                    return self.social_state
            else:
                for thing in things:
                    if self.check_probability("interact with"):
                        self.destination = thing
                        self.social_state = SocialState.WALKING
                        return self.social_state
            return self.social_state

        elif self.social_state == SocialState.WALKING:
            if self.reached_destination():
                self.social_state = SocialState.IDLE
                if self.destination.can_interact:
                    if self.destination.can_talk:
                        self.talk(self.destination)
                    else:
                        self.interact(self.destination)
        
            return self.social_state
            
        elif self.social_state == SocialState.TALKING: #TODO
            if self.check_probability("stop talking"):
                self.group.leave(self)
                self.group = None
                self.social_state = SocialState.IDLE
                self.destination = Thingy(self.get_position())

        elif self.social_state == SocialState.INTERACTING: #TODO
            if self.check_probability("stop interacting"):
                self.social_state = SocialState.IDLE
                self.destination = Thingy(self.get_position())
            
        return self.social_state

    def update_destination(self, agent_state: pedsim_msgs.msg.AgentState) -> pedsim_msgs.msg.AgentState:
        agent_state.destination.x = self.destination.get_position()[0]
        agent_state.destination.y = self.destination.get_position()[1]
        return agent_state
    
    def update_force(self, feedback: FeedbackDatum) -> FeedbackDatum: #TODO
        if self.social_state in [SocialState.TALKING, SocialState.INTERACTING] and self.speed < 0.05:

            vector = (self.destination.get_position()-self.get_position())
            be = np.linalg.norm(vector)

            feedback.feedback.force.x = 0.000001 * vector[0]/be
            feedback.feedback.force.y = 0.000001 * vector[1]/be

        return feedback
    
    def see(self, things: List[Thingy]) -> List[Thingy]:
        return list(filter((lambda thing: self.range > np.linalg.norm(self.get_position()-thing.get_position()) > 0 
                       and np.deg2rad(self.fov/2) < abs(self.direction-np.arctan2(*reversed(list(self.get_position()-thing.get_position()))))), things))
    
    def talk(self, thing): #TODO
        self.social_state = SocialState.TALKING
        if (thing.group):
            thing.group.join(self)
            self.group = thing.group
        else:
            self.group = Groupy(self, thing)
            thing.social_state = SocialState.TALKING
            thing.group = self.group
        self.destination = self.group
    
    def interact(self, thing: Objecty): #TODO
        self.social_state = SocialState.INTERACTING
        thing.interact()
    
    def reached_destination(self) -> bool:
        return self.destination.interaction_radius > abs(np.linalg.norm(self.get_position()-self.destination.get_position()))

class World():
    agents: List[Agenty]
    objects: List[Objecty]
    map_size = np.array([24,24])

    def __init__(self):
        self.agents = []
        self.objects = []
    
    def load_stuff(self, input_data: InputData):
        self.agents = [Agenty(state) for state in input_data.agents]
        self.objects = [Objecty(obstacle) for obstacle in input_data.obstacles]

    def update_agents(self, input_data: InputData) -> InputData:
        for agent, state in zip(self.agents, input_data.agents):
            agent.update(state)

        for agent in self.agents:
            agent.next_state(self.get_things())
        
        rospy.logerr([(agent.social_state) for agent in self.agents])
        rospy.logwarn([(agent.destination, agent.destination.get_position()) for agent in self.agents])

        input_data.agents = [agent.update_destination(state) for agent, state in zip(self.agents, input_data.agents)]

        return input_data
    
    def get_things(self) -> List[Thingy]:
        return self.agents+self.objects
    
    def random_thing(start: np.ndarray, a:float, b: float, direction: float, fov: float) -> Thingy:
        rng = np.random.default_rng()
        angle = direction+np.deg2rad(rng.random()*fov-fov/2)
        random_vector = ((b-a)*rng.random()+a)* np.array(np.cos(angle), np.sin(angle))
        new_vector = start+random_vector
        new_vector[0] = 0 if new_vector[0] < 0 else (World.map_size[0] if new_vector[0] > World.map_size[0] else new_vector[0])
        new_vector[1] = 0 if new_vector[1] < 0 else (World.map_size[1] if new_vector[1] > World.map_size[1] else new_vector[1])
        return Thingy(start+new_vector)
    
    def reset(self):
        self.agents = self.objects = []