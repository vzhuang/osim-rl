import opensim
import math
import numpy as np
import os
import random
import string
from itertools import chain
from .osim import OsimEnv

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

class RunEnv(OsimEnv):
    STATE_PELVIS_X = 1
    STATE_PELVIS_Y = 2
    MUSCLES_PSOAS_R = 3
    MUSCLES_PSOAS_L = 11

    num_obstacles = 0
    max_obstacles = None

    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc.osim')
    ligamentSet = []
    verbose = False
    pelvis = None
    env_desc = {"obstacles": [], "muscles": [1]*18}

    ninput = 41
    noutput = 18

    def __init__(self, visualize = True, max_obstacles = 3):
        self.max_obstacles = max_obstacles
        super(RunEnv, self).__init__(visualize = False, noutput = self.noutput)
        self.osim_model.model.setUseVisualizer(visualize)
        self.create_obstacles()
        state = self.osim_model.model.initSystem()

        self.total_steps = 0

        if visualize:
            manager = opensim.Manager(self.osim_model.model)
            manager.setInitialTime(-0.00001)
            manager.setFinalTime(0.0)
            manager.integrate(state)

        self.imitate_gait = False

    def setup(self, difficulty, seed=None):
        # create the new env
        # set up obstacles
        self.env_desc = self.generate_env(difficulty, seed, self.max_obstacles)
        
        self.clear_obstacles(self.osim_model.state)
        for x,y,r in self.env_desc['obstacles']:
            self.add_obstacle(self.osim_model.state,x,y,r)

        # set up muscle strength
        self.osim_model.set_strength(self.env_desc['muscles'])

    def reset(self, difficulty=2, seed=None):
        super(RunEnv, self).reset()
        self.istep = 0
        self.setup(difficulty, seed)
        self.last_state = self.get_observation()
        self.current_state = self.last_state
        return self.last_state

    def set_imitation(self, left_obs, right_obs, cycle_length, match_indices, x_indices, max_ep_len):
        self.imitate_gait = True
        self.left_obs = left_obs
        self.right_obs = right_obs
        self.cycle_length = cycle_length
        self.match_indices = match_indices
        self.x_indices = x_indices
        self.x_offset = self.current_state
        self.max_ep_len = max_ep_len
        self.ep_multiplier = 1

    def compute_reward(self):        
        if self.imitate_gait:
            cycle = (self.istep / self.cycle_length) % 2
            timestep = self.istep % self.cycle_length
            # set x_offset
            if timestep == 0:
                self.x_offset = self.current_state
            obs_arr = self.left_obs
            if cycle:
                obs_arr = self.right_obs
            # reward = 0.1 - (obs_arr[timestep][8] - self.current_state[8])**2 - (obs_arr[timestep][11] - self.current_state[11])**2
            reward = 0
            for idx in self.match_indices:
                if idx in self.x_indices:
                    reward += (self.x_offset[idx] + obs_arr[timestep][idx] - self.current_state[idx])**2
                else:
                    reward += (obs_arr[timestep][idx] - self.current_state[idx])**2
            with open('/home/ubuntu/imitate_log.txt', 'a') as f:
                
                f.write('timestep ' + str(timestep)+'\n')
                f.write('last action: ')
                f.write(" ".join([str(x) for x in self.last_action]) + '\n')
                for i in range(41):
                    f.write('feature ' + str(i) + ': ')
                    f.write(str(obs_arr[timestep][i]) + ' ' + str(self.current_state[i]) + '\n')
                f.close()                
            reward = np.exp(-reward)#0.1 - np.sqrt(reward) / len(self.match_indices)
            return reward
        # Compute ligaments penalty
            # lig_pen = 0
            # # Get ligaments
            # for j in range(20, 26):
            #     lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            #     lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

            # # Get the pelvis X delta
            # delta_x = self.current_state[self.STATE_PELVIS_X] - self.last_state[self.STATE_PELVIS_X]

            # reward += delta_x - math.sqrt(lig_pen) * 10e-8            
            # return reward
        else:
        # Compute ligaments penalty
            lig_pen = 0
            # Get ligaments
            for j in range(20, 26):
                lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
                lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

            # Get the pelvis X delta
            delta_x = self.current_state[self.STATE_PELVIS_X] - self.last_state[self.STATE_PELVIS_X]

            return delta_x - math.sqrt(lig_pen) * 10e-8

    def is_pelvis_too_low(self):
        return (self.current_state[self.STATE_PELVIS_Y] < 0.65)
    
    def is_done(self):
        return self.is_pelvis_too_low() or (self.istep >= self.spec.timestep_limit) #or self.istep >= self.max_ep_len * self.ep_multiplier

    def configure(self):
        super(RunEnv, self).configure()

        if self.verbose:
            print("JOINTS")
            for i in range(11):
                print(i,self.osim_model.jointSet.get(i).getName())
            print("\nBODIES")
            for i in range(13):
                print(i,self.osim_model.bodySet.get(i).getName())
            print("\nMUSCLES")
            for i in range(18):
                print(i,self.osim_model.muscleSet.get(i).getName())
            print("\nFORCES")
            for i in range(26):
                print(i,self.osim_model.forceSet.get(i).getName())
            print("")

        # for i in range(18):
        #     m = opensim.Thelen2003Muscle.safeDownCast(self.osim_model.muscleSet.get(i))
        #     m.setActivationTimeConstant(0.0001) # default 0.01
        #     m.setDeactivationTimeConstant(0.0001) # default 0.04

        # The only joint that has to be cast
        self.pelvis = opensim.PlanarJoint.safeDownCast(self.osim_model.get_joint("ground_pelvis"))

    def next_obstacle(self):
        obstacles = self.env_desc['obstacles']
        x = self.pelvis.getCoordinate(self.STATE_PELVIS_X).getValue(self.osim_model.state)
        for obstacle in obstacles:
            if obstacle[0] + obstacle[2] < x:
                continue
            else:
                ret = list(obstacle)
                ret[0] = ret[0] - x
                return ret
        return [100,0,0]
        
    def _step(self, action):
        self.last_action = action
        self.last_state = self.current_state
        self.total_steps += 1
        # if self.total_steps > 10 and self.total_steps % 10000 == 0:
        #     self.ep_multiplier += 1
        return super(RunEnv, self)._step(action)

    def get_observation(self):
        bodies = ['head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']

        pelvis_pos = [self.pelvis.getCoordinate(i).getValue(self.osim_model.state) for i in range(3)]
        pelvis_vel = [self.pelvis.getCoordinate(i).getSpeedValue(self.osim_model.state) for i in range(3)]

        jnts = ['hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l']
        joint_angles = [self.osim_model.get_joint(jnts[i]).getCoordinate().getValue(self.osim_model.state) for i in range(6)]
        joint_vel = [self.osim_model.get_joint(jnts[i]).getCoordinate().getSpeedValue(self.osim_model.state) for i in range(6)]

        mass_pos = [self.osim_model.model.calcMassCenterPosition(self.osim_model.state)[i] for i in range(2)]  
        mass_vel = [self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)[i] for i in range(2)]

        body_transforms = [[self.osim_model.get_body(body).getTransformInGround(self.osim_model.state).p()[i] for i in range(2)] for body in bodies]

        muscles = [ self.env_desc['muscles'][self.MUSCLES_PSOAS_L], self.env_desc['muscles'][self.MUSCLES_PSOAS_R] ]
    
        # see the next obstacle
        obstacle = self.next_obstacle()

#        feet = [opensim.HuntCrossleyForce.safeDownCast(self.osim_model.forceSet.get(j)) for j in range(20,22)]
        self.current_state = pelvis_pos + pelvis_vel + joint_angles + joint_vel + mass_pos + mass_vel + list(flatten(body_transforms)) + muscles + obstacle
        return self.current_state

    def create_obstacles(self):
        x = 0
        y = 0
        r = 0.1
        for i in range(self.max_obstacles):
            name = i.__str__()
            blockos = opensim.Body(name + '-block', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
            pj = opensim.PlanarJoint(name + '-joint',
                                  self.osim_model.model.getGround(), # PhysicalFrame
                                  opensim.Vec3(0, 0, 0),
                                  opensim.Vec3(0, 0, 0),
                                  blockos, # PhysicalFrame
                                  opensim.Vec3(0, 0, 0),
                                  opensim.Vec3(0, 0, 0))

            self.osim_model.model.addJoint(pj)
            self.osim_model.model.addBody(blockos)

            block = opensim.ContactSphere(r, opensim.Vec3(0,0,0), blockos)
            block.setName(name + '-contact')
            self.osim_model.model.addContactGeometry(block)

            force = opensim.HuntCrossleyForce()
            force.setName(name + '-force')
            
            force.addGeometry(name + '-contact')
            force.addGeometry("r_heel")
            force.addGeometry("l_heel")
            force.addGeometry("r_toe")
            force.addGeometry("l_toe")
        
            force.setStiffness(1.0e6/r)
            force.setDissipation(1e-5)
            force.setStaticFriction(0.0)
            force.setDynamicFriction(0.0)
            force.setViscousFriction(0.0)

            self.osim_model.model.addForce(force);

    def clear_obstacles(self, state):
        for j in range(0, self.max_obstacles):
            joint_generic = self.osim_model.get_joint("%d-joint" % j)
            joint = opensim.PlanarJoint.safeDownCast(joint_generic)
            joint.getCoordinate(1).setValue(state, 0)
            joint.getCoordinate(2).setValue(state, -0.1)

            contact_generic = self.osim_model.get_contact_geometry("%d-contact" % j)
            contact = opensim.ContactSphere.safeDownCast(contact_generic)
            contact.setRadius(0.0001)

            for i in range(3):
                joint.getCoordinate(i).setLocked(state, True)

        self.num_obstacles = 0
        pass
        
    def add_obstacle(self, state, x, y, r):
        # set obstacle number num_obstacles
        contact_generic = self.osim_model.get_contact_geometry("%d-contact" % self.num_obstacles)
        contact = opensim.ContactSphere.safeDownCast(contact_generic)
        contact.setRadius(r)

        force_generic = self.osim_model.get_force("%d-force" % self.num_obstacles)
        force = opensim.HuntCrossleyForce.safeDownCast(force_generic)
        force.setStiffness(1.0e6/r)

        joint_generic = self.osim_model.get_joint("%d-joint" % self.num_obstacles)
        joint = opensim.PlanarJoint.safeDownCast(joint_generic)
        
        newpos = [x,y] 
        for i in range(2):
            joint.getCoordinate(1 + i).setLocked(state, False)
            joint.getCoordinate(1 + i).setValue(state, newpos[i], False)
            joint.getCoordinate(1 + i).setLocked(state, True)

        self.num_obstacles += 1
        pass

    def generate_env(self, difficulty, seed, max_obstacles):
        if seed is not None:
            np.random.seed(seed) # seed the RNG if seed is provided

        # obstacles
        num_obstacles = 0
        xs = []
        ys = []
        rs = []
        
        if 0 < difficulty:
            num_obstacles = min(3, max_obstacles)
            xs = np.random.uniform(1.0, 5.0, num_obstacles)
            ys = np.random.uniform(-0.25, 0.25, num_obstacles)
            rs = [0.05 + r for r in np.random.exponential(0.05, num_obstacles)]

        if 0 < difficulty and 3 < max_obstacles:
            extra_obstacles = max(min(20, max_obstacles) - num_obstacles, 0)
            xs = np.concatenate([xs,(np.cumsum(np.random.uniform(2.0, 4.0, extra_obstacles)) + 5)])
            ys = np.concatenate([ys,np.random.uniform(-0.05, 0.25, extra_obstacles)])
            rs = rs + [0.05 + r for r in np.random.exponential(0.05, extra_obstacles)]
            num_obstacles = len(xs)

        ys = map(lambda xy: xy[0]*xy[1], list(zip(ys, rs)))

        # muscle strength
        rpsoas = 1
        lpsoas = 1
        if difficulty >= 2:
            rpsoas = 1 - np.random.normal(0, 0.1)
            lpsoas = 1 - np.random.normal(0, 0.1)
            rpsoas = max(0.5, rpsoas)
            lpsoas = max(0.5, lpsoas)

        muscles = [1] * 18
            
        # modify only psoas
        muscles[self.MUSCLES_PSOAS_R] = rpsoas
        muscles[self.MUSCLES_PSOAS_L] = lpsoas

        obstacles = list(zip(xs,ys,rs))
        obstacles.sort()

        return {
            'muscles': muscles,
            'obstacles': obstacles
        }

