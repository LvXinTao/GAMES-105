# 以下部分均为可更改部分
# Most of the codes are taken from Daniel Holden's Learned Motion Matching
# https://github.com/orangeduck/Motion-Matching
from answer_task1 import *
from smooth_utils import quat_to_avel,decay_spring_implicit_damping_pos,decay_spring_implicit_damping_rot
import scipy.signal as signal
from scipy.spatial.transform import Rotation as R

def clamp(x,min,max):
                    if x<min:
                        return min
                    elif x>max:
                        return max
                    else:
                        return x

length=np.linalg.norm
                    
class CharacterController():
    def __init__(self, controller,dt=1.0/60.0) -> None:
        self.motions = []
        self.motion_files=[
            'motion_material/idle.bvh',
            'motion_material/walk_forward.bvh',
            # 'motion_material/run_forward.bvh',
            'motion_material/walkF.bvh',
            'motion_material/walk_and_ture_right.bvh',
            'motion_material/walk_and_turn_left.bvh'
        ]
        for file in self.motion_files:
            self.motions.append(BVHMotion(file))
        # adjust joint names
        tgt_joint_name=self.motions[0].joint_name
        for motion in self.motions:
            motion.adjust_joint_name(tgt_joint_name)

        self.dt=dt # current fps:60

        self.db=MotionMatchingDatabase(self.motions,dt=self.dt)
        
        # search timer for our motion matching
        self.search_time=0.1
        self.search_timer=self.search_time
        self.force_search_timer=self.search_time
        # state change
        self.desired_velocity_prev=np.zeros(3)
        self.desired_velocity_change_prev=0.0
        self.desired_velocity_change_threshold=50.0
        self.desired_rotation_prev=np.array([0.0,0.0,0.0,1.0])
        self.desired_rotation_change_prev=np.zeros(3)
        self.desired_rotation_change_threshold=50.0
        # simulation speed
        self.simulation_walk_fwd_speed=1.75
        self.simulation_walk_side_speed=1.5
        self.simulation_walk_back_speed=1.25

        self.controller = controller
        self.cur_frame = self.db.range_starts[0]
        self.cur_root_pos = self.db.bone_positions[self.cur_frame,0]
        self.cur_root_rot = self.db.bone_rotations[self.cur_frame,0]
        self.cur_root_velocity=self.db.bone_velocities[self.cur_frame,0]
        self.cur_root_angular_velocity=self.db.bone_angular_velocities[self.cur_frame,0]

        # inertializers offset recordings
        self.bone_offset_positions=np.zeros((len(self.db.bone_names),3))
        self.bone_offset_velocities=np.zeros((len(self.db.bone_names),3))
        self.bone_offset_rotations=np.zeros((len(self.db.bone_names),4))
        self.bone_offset_rotations[:,-1]=1.0 # x,y,z,w for quaternion
        self.bone_offset_angular_velocities=np.zeros((len(self.db.bone_names),3))

        # transition recordings
        self.transition_src_position=self.cur_root_pos
        self.transition_src_rotation=self.cur_root_rot
        self.transition_dst_position=np.zeros(3)
        self.transition_dst_rotation=np.array([0.0,0.0,0.0,1.0])

        self.inertialize_blending_halflife=0.1

        self._inertialze_pose_update(
            self.db.bone_positions[self.cur_frame],
            self.db.bone_velocities[self.cur_frame],
            self.db.bone_rotations[self.cur_frame],
            self.db.bone_angular_velocities[self.cur_frame],
            self.transition_src_position,
            self.transition_src_rotation,
            self.transition_dst_position,
            self.transition_dst_rotation,
            self.inertialize_blending_halflife,
            self.dt
        )

        pass
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        # Check if we should force a search because input changed quickly
        desired_velocity_change_prev=self.desired_velocity_change_prev
        desired_velocity_change_curr=(desired_vel_list[0]-self.desired_velocity_prev)/self.dt

        desired_rotation_change_prev=self.desired_rotation_change_prev
        desired_rotation_change_curr=\
            (R.from_quat(desired_rot_list[0])*R.from_quat(self.desired_rotation_prev).inv()).as_rotvec()/self.dt

        force_search=False
        if(self.force_search_timer<=0.0 and 
           ((length(desired_velocity_change_prev)>=self.desired_velocity_change_threshold
             and length(desired_velocity_change_curr)<self.desired_velocity_change_threshold) or
             (length(desired_rotation_change_prev)>=self.desired_rotation_change_threshold
              and length(desired_rotation_change_curr)<self.desired_rotation_change_threshold))):
            force_search=True
            self.force_search_timer=self.search_time
        elif self.force_search_timer>0.0:
            self.force_search_timer-=self.dt
        
        self.desired_velocity_change_prev=desired_velocity_change_curr
        self.desired_rotation_change_prev=desired_rotation_change_curr

        # Make query vector for search
        # In theory this only needs to be done when a search is actually required 
        # However for visualizatio purposes it can be nice to do it every frame
        curr_denormalized_feature=self.db.get_denormalized_feature(self.cur_frame)
        # we need to change trajectory position 2D and trajectory directions 2D, which are the last 10 and 10 features
        traj1=R.from_quat(self.cur_root_rot).inv().apply(desired_pos_list[1]-self.cur_root_pos)
        traj2=R.from_quat(self.cur_root_rot).inv().apply(desired_pos_list[2]-self.cur_root_pos)
        traj3=R.from_quat(self.cur_root_rot).inv().apply(desired_pos_list[3]-self.cur_root_pos)
        traj4=R.from_quat(self.cur_root_rot).inv().apply(desired_pos_list[4]-self.cur_root_pos)
        traj5=R.from_quat(self.cur_root_rot).inv().apply(desired_pos_list[5]-self.cur_root_pos)
        curr_denormalized_feature[-20:-10]=np.array(
            [traj1[0],traj1[2],
            traj2[0],traj2[2],
            traj3[0],traj3[2],
            traj4[0],traj4[2],
            traj5[0],traj5[2]
            ]
        )
        rot1=R.from_quat(self.cur_root_rot).inv().apply(
            R.from_quat(desired_rot_list[1]).apply(np.array([0.0,0.0,1.0]))
        )
        rot2=R.from_quat(self.cur_root_rot).inv().apply(
            R.from_quat(desired_rot_list[2]).apply(np.array([0.0,0.0,1.0]))
        )
        rot3=R.from_quat(self.cur_root_rot).inv().apply(
            R.from_quat(desired_rot_list[3]).apply(np.array([0.0,0.0,1.0]))
        )
        rot4=R.from_quat(self.cur_root_rot).inv().apply(
            R.from_quat(desired_rot_list[4]).apply(np.array([0.0,0.0,1.0]))
        )
        rot5=R.from_quat(self.cur_root_rot).inv().apply(
            R.from_quat(desired_rot_list[5]).apply(np.array([0.0,0.0,1.0]))
        )
        curr_denormalized_feature[-10:]=np.array(
              [rot1[0],rot1[2],
                rot2[0],rot2[2],
                rot3[0],rot3[2],
                rot4[0],rot4[2],
                rot5[0],rot5[2]
            ]
        )
        assert curr_denormalized_feature.shape[0]==self.db.n_features

        # Check if we reached the end of the current anim
        end_of_anim=(self.db._trajectory_index_clamp(self.cur_frame,1)==self.cur_frame)
        # Do we need to search?
        if force_search or self.search_timer<=0.0 or end_of_anim:
            # Search
            best_index=-1 if end_of_anim else self.cur_frame
            best_cost=np.inf
            
            best_index,best_cost=self.db.database_search(
                best_index,
                best_cost,
                curr_denormalized_feature
            )
            # Transition if better frame found
            if best_index!=self.cur_frame:
                curr_bone_positions=self.db.bone_positions[self.cur_frame]
                curr_bone_velocities=self.db.bone_velocities[self.cur_frame]
                curr_bone_rotations=self.db.bone_rotations[self.cur_frame]
                curr_bone_angular_velocities=self.db.bone_angular_velocities[self.cur_frame]

                trns_bone_positions=self.db.bone_positions[best_index]
                trns_bone_velocities=self.db.bone_velocities[best_index]
                trns_bone_rotations=self.db.bone_rotations[best_index]
                trns_bone_angular_velocities=self.db.bone_angular_velocities[best_index]

                self._inertialize_pose_transition(
                    self.cur_root_pos,
                    self.cur_root_velocity,
                    self.cur_root_rot,
                    self.cur_root_angular_velocity,
                    curr_bone_positions,
                    curr_bone_velocities,
                    curr_bone_rotations,
                    curr_bone_angular_velocities,
                    trns_bone_positions,
                    trns_bone_velocities,
                    trns_bone_rotations,
                    trns_bone_angular_velocities
                )
                self.cur_frame=best_index
            else:
                trns_bone_positions=self.db.bone_positions[best_index]
                trns_bone_velocities=self.db.bone_velocities[best_index]
                trns_bone_rotations=self.db.bone_rotations[best_index]
                trns_bone_angular_velocities=self.db.bone_angular_velocities[best_index]
            # Reset search timer
            self.search_timer=self.search_time
            


        # Tick down search timer
        self.search_timer-=self.dt

        # Tick frame
        self.cur_frame+=1
        # Look up next pose
        curr_bone_positions=self.db.bone_positions[self.cur_frame]
        curr_bone_velocities=self.db.bone_velocities[self.cur_frame]
        curr_bone_rotations=self.db.bone_rotations[self.cur_frame]
        curr_bone_angular_velocities=self.db.bone_angular_velocities[self.cur_frame]

        # Update inertializers
        curr_bone_positions,curr_bone_velocities,\
        curr_bone_rotations,curr_bone_angular_velocities=self._inertialze_pose_update(
            curr_bone_positions,curr_bone_velocities,
            curr_bone_rotations,curr_bone_angular_velocities,
            self.transition_src_position,self.transition_src_rotation,
            self.transition_dst_position,self.transition_dst_rotation,
            self.inertialize_blending_halflife,self.dt
        )
        # Update root information
        self.cur_root_pos,self.cur_root_rot,\
        self.cur_root_velocity,self.cur_root_angular_velocity=\
            curr_bone_positions[0],curr_bone_rotations[0],\
            curr_bone_velocities[0],curr_bone_angular_velocities[0]
        
        # Return the global orientation and position of the joints
        joint_translation,joint_orientation=self.db.forward_kinematics_full(
            curr_bone_positions,curr_bone_rotations,
            self.db.bone_parents
        )
        return self.db.bone_names,joint_translation,joint_orientation
             

    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        
        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)
        
        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.

    def _inertialize_pose_transition(self,
                                     root_position,
                                     root_velocity,
                                     root_rotation,
                                     root_angular_velocity,
                                     curr_bone_positions,
                                     curr_bone_velocities,
                                     curr_bone_rotations,
                                     curr_bone_angular_velocities,
                                     trns_bone_positions,
                                     trns_bone_velocities,
                                     trns_bone_rotations,
                                     trns_bone_angular_velocities,
                                     ):
        """
        Update the offset inertializers for the pose transition
        """
        # First we record the root position and rotation in the animation data 
        # for the source and destination animation
        self.transition_dst_position=root_position
        self.transition_dst_rotation=root_rotation
        self.transition_src_position=trns_bone_positions[0]
        self.transition_src_rotation=trns_bone_rotations[0]

        # we then find the velocities so we can transition the root inertializers
        world_space_dst_velocity=R.from_quat(self.transition_dst_rotation).apply(
            R.from_quat(self.transition_src_rotation).inv().apply(trns_bone_velocities[0])
        )
        world_space_dst_angular_velocity=R.from_quat(self.transition_dst_rotation).apply(
            R.from_quat(self.transition_src_rotation).inv().apply(trns_bone_angular_velocities[0])
        )

        # Transition inertializers recording the offsets for the root
        self.bone_offset_positions[0],self.bone_offset_velocities[0]=self._inertialize_transition(
            self.bone_offset_positions[0],self.bone_offset_velocities[0],
            root_position,root_velocity,
            root_position,world_space_dst_velocity
        )
        self.bone_offset_rotations[0],self.bone_offset_angular_velocities[0]=self._inertialize_transition(
            self.bone_offset_rotations[0],self.bone_offset_angular_velocities[0],
            root_rotation,root_angular_velocity,
            root_rotation,world_space_dst_angular_velocity
        )
        # Transition all the inertializers for the rest of the bones
        for i in range(1,curr_bone_positions.shape[0]):
            self.bone_offset_positions[i],self.bone_offset_velocities[i]=self._inertialize_transition(
                self.bone_offset_positions[i],self.bone_offset_velocities[i],
                curr_bone_positions[i],curr_bone_velocities[i],
                trns_bone_positions[i],trns_bone_velocities[i]
            )
            self.bone_offset_rotations[i],self.bone_offset_angular_velocities[i]=self._inertialize_transition(
                self.bone_offset_rotations[i],self.bone_offset_angular_velocities[i],
                curr_bone_rotations[i],curr_bone_angular_velocities[i],
                trns_bone_rotations[i],trns_bone_angular_velocities[i])

    def _inertialize_transition(self,
                                off_x,off_v,
                                src_x,src_v,
                                dst_x,dst_v):
        off_x=(src_x+off_x)-dst_x
        off_v=(src_v+off_v)-dst_v
        return off_x,off_v

    def _inertialze_pose_update(self,
                                bone_input_positions,
                                bone_input_velocities,
                                bone_input_rotations,
                                bone_input_angular_velocities,
                                transitions_src_position,
                                transitions_src_rotation,
                                transition_dst_position,
                                transition_dst_rotation,
                                halflife,
                                dt):
        """
        Update the inertializer states and output the smoothed animation
        """
        smoothed_bone_positions=np.empty_like(bone_input_positions)
        smoothed_bone_velocities=np.empty_like(bone_input_velocities)
        smoothed_bone_rotations=np.empty_like(bone_input_rotations)
        smoothed_bone_angular_velocities=np.empty_like(bone_input_angular_velocities)

        # First we find the next root position, velocity, rotation and angular velocity
        # in the world space by transforming the input animation from its animation space
        # into the space of the currently playing animation.
        world_space_position=R.from_quat(transition_dst_rotation).apply(
            R.from_quat(transitions_src_rotation).inv().apply(bone_input_positions[0]-transitions_src_position)
        )+transition_dst_position
        world_space_velocity=R.from_quat(transition_dst_rotation).apply(
            R.from_quat(transitions_src_rotation).inv().apply(bone_input_velocities[0])
        )
        # Normalize here
        world_space_rotation=(R.from_quat(transition_dst_rotation)*R.from_quat(transitions_src_rotation).inv()\
                            *R.from_quat(bone_input_rotations[0])).as_quat()
        world_space_rotation=world_space_rotation/length(world_space_rotation)

        world_space_angular_velocity=R.from_quat(transition_dst_rotation).apply(
            R.from_quat(transitions_src_rotation).inv().apply(bone_input_angular_velocities[0])
        )
        # Then we update these two inertializers with these new world space inputs
        smoothed_bone_positions[0],smoothed_bone_velocities[0],\
        self.bone_offset_positions[0],self.bone_offset_velocities[0]=self._inertialize_update(
             self.bone_offset_positions[0],self.bone_offset_velocities[0],
                world_space_position,world_space_velocity,
                halflife,dt)
        smoothed_bone_rotations[0],smoothed_bone_angular_velocities[0],\
        self.bone_offset_rotations[0],self.bone_offset_angular_velocities[0]=self._inertialize_update(
            self.bone_offset_rotations[0],self.bone_offset_angular_velocities[0],
            world_space_rotation,world_space_angular_velocity,
            halflife,dt
        )
        # Then we update the inertializers for the rest of the bones
        for i in range(1,bone_input_positions.shape[0]):
            smoothed_bone_positions[i],smoothed_bone_velocities[i],\
            self.bone_offset_positions[i],self.bone_offset_velocities[i]=self._inertialize_update(
                self.bone_offset_positions[i],self.bone_offset_velocities[i],
                bone_input_positions[i],bone_input_velocities[i],
                halflife,dt
            )
            smoothed_bone_rotations[i],smoothed_bone_angular_velocities[i],\
            self.bone_offset_rotations[i],self.bone_offset_angular_velocities[i]=self._inertialize_update(
                self.bone_offset_rotations[i],self.bone_offset_angular_velocities[i],
                bone_input_rotations[i],bone_input_angular_velocities[i],
                halflife,dt
            )
        
        return smoothed_bone_positions,smoothed_bone_velocities,\
            smoothed_bone_rotations,smoothed_bone_angular_velocities
    
    def _inertialize_update(self,
                            off_x,off_v,
                            in_x,in_v,
                            halflife,dt
                            ):
        if len(off_x)==3:
            # for position and velocity
            off_x,off_v=decay_spring_implicit_damping_pos(off_x,off_v,halflife,dt)
            out_x=in_x+off_x
            out_v=in_v+off_v
            return out_x,out_v,off_x,off_v
        elif len(off_x)==4:
            # for rotation and angular velocity
            off_x=R.from_quat(off_x).as_rotvec()
            off_x,off_v=decay_spring_implicit_damping_rot(off_x,off_v,halflife,dt)
            off_x=R.from_rotvec(off_x).as_quat()

            out_x=(R.from_quat(off_x)*R.from_quat(in_x)).as_quat()
            out_v=R.from_quat(off_x).apply(in_v)+off_v

            return out_x,out_v,off_x,off_v
        else:
            assert False
        

        

class MotionMatchingDatabase:

    BOUND_SM_SIZE=16
    BOUND_LR_SIZE=64

    def __init__(self,motions,dt=1.0/60.0):
        self.motions=motions
        self.n_features=3+3+3+3+3+10+10 # Left foot position, Right foot position, Left foot velocity, Right foot velocity,\
                                        # Hip velocity, Trajectory Positions 2D, Trajectory Directions 2D
        self.dt=dt
        self.bone_positions,self.bone_velocities,self.bone_rotations,self.bone_angular_velocities=\
                                                                                None,None,None,None
        self.bone_parents=None
        self.bone_names=None
        self.range_starts,self.range_stops=None,None
        self.nframes=None
        self._bone_info_from_motions()


        self.features=np.empty((self.nframes,self.n_features),dtype=np.float32)
        self.features_offset=np.empty((self.n_features,),dtype=np.float32)
        self.features_scale=np.empty((self.n_features,),dtype=np.float32)

        self.feature_weight_foot_position=0.75
        self.feature_weight_foot_velocity=1.0
        self.feature_weight_hip_velocity=1.0
        self.feature_weight_trajectory_positions=1.0
        self.feature_weight_trajectory_directions=1.5

        self.bound_sm_min=None
        self.bound_sm_max=None
        self.bound_lr_min=None
        self.bound_lr_max=None
        
        self._build_matching_features()
        print('--------Motion Matching Database Initialized Done!!!----------')

    def database_search(self,
                        best_index,
                        best_cost,
                        query,
                        transition_cost=0.0,
                        ignore_range_end=20,
                        ignore_surrounding=20):
        """
        Search the database and find the nearst neighbour in the database.
        Here we implement the K-D tree nearsest neighbour search.
        return the best index and best cost.
        """
        # Normalize query
        query_normalized=(query-self.features_offset)/self.features_scale
        
        curr_index=best_index
        # Find cost for current frame
        if best_index!=-1:
            best_cost=np.sum(np.square(query_normalized-self.features[best_index]))

        curr_cost=0.0
        # search rest of database
        for r in range(len(self.range_starts)):
            i=self.range_starts[r]
            range_end=self.range_stops[r]-ignore_range_end
            while i<range_end:
                # Find index of current and next large box
                i_lr=i//self.BOUND_LR_SIZE
                i_lr_next=(i_lr+1)*self.BOUND_LR_SIZE
                # Find distance to box
                curr_cost=transition_cost
                curr_cost+=np.sum(np.square(query_normalized-\
                                            np.clip(query_normalized,self.bound_lr_min[i_lr],self.bound_lr_max[i_lr])))
                # if distance is greater than current best jump to next box
                if curr_cost>=best_cost:
                    i=i_lr_next
                    continue
                # Check against small box
                while i<i_lr_next and i<range_end:
                    # Find index of current and next small box
                    i_sm=i//self.BOUND_SM_SIZE
                    i_sm_next=(i_sm+1)*self.BOUND_SM_SIZE
                    # Find distance to box
                    curr_cost=transition_cost
                    curr_cost+=np.sum(np.square(query_normalized-\
                                                np.clip(query_normalized,self.bound_sm_min[i_sm],self.bound_sm_max[i_sm])))
                    if curr_cost>=best_cost:
                        i=i_sm_next
                        continue
                    # Search inside small box
                    while i<i_sm_next and i<range_end:
                        # Skip surrounding frames
                        if curr_index!=-1 and abs(i-curr_index)<ignore_surrounding:
                            i+=1
                            continue
                        # Check against each frame inside small box
                        curr_cost=transition_cost
                        curr_cost+=np.sum(np.square(query_normalized-self.features[i]))
                        if curr_cost<best_cost:
                            best_index=i
                            best_cost=curr_cost
                        i+=1

        return best_index,best_cost

    def _bone_info_from_motions(self):
        """
        Extract bone information from motion files,
        and generate simulation bones.
        """
        bone_positions=[]
        bone_velocities=[]
        bone_rotations=[]
        bone_angular_velocities=[]
        bone_parents=[]
        bone_names=[]

        range_starts=[]
        range_stops=[]
        for motion in self.motions:
            positions=motion.joint_position
            rotations=motion.joint_rotation
            parents=motion.joint_parent
            names=motion.joint_name

            nframes=positions.shape[0]
            nbones=positions.shape[1]
            # Extract Simulation Bone
            global_translation,global_orientation=motion.batch_forward_kinematics()
            # specify joints to use for simulation bone
            sim_position_joint=motion.joint_name.index('pelvis_lowerback')
            sim_rotation_joint=motion.joint_name.index('RootJoint')

            # Position comes from pelvis_lowerback
            sim_position=np.array([1.0,0.0,1.0])*global_translation[:,sim_position_joint,:]
            sim_position=signal.savgol_filter(sim_position, 31, 3, axis=0,mode='interp')
            # Direction comes from projected hip forward direction
            sim_direction=np.array([1.0,0.0,1.0])*\
                        (R.from_quat(global_orientation[:,sim_rotation_joint]).apply(np.array([0.0,1.0,0.0])))
            # re-normalize direction
            sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[...,np.newaxis] 
            sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
            sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[...,np.newaxis]) # (N,3)
            # Extract rotation from direction
            rot_axis=np.array([0.0,1.0,0.0])[np.newaxis,...].repeat(nframes,axis=0) # (N,3)
            rot_theta=np.arccos(np.clip(np.sum(sim_direction*np.array([0.0,0.0,1.0]),axis=-1),-1.0,1.0))[...,np.newaxis].repeat(3,axis=-1) # (N,3)
            rotvec=np.multiply(rot_axis,rot_theta) # (N,3)
            sim_rotation=R.from_rotvec(rotvec).as_quat() # (N,4)
            
            # Transform first joints to be local to sim and append sim as root bone
            positions[:,0]=R.from_quat(sim_rotation).inv().apply(positions[:,0]-sim_position)
            rotations[:,0]=(R.from_quat(sim_rotation).inv()*R.from_quat(rotations[:,0])).as_quat()

            positions=np.concatenate([sim_position[:,None,:],positions],axis=1)
            rotations=np.concatenate([sim_rotation[:,None,:],rotations],axis=1)

            
            parents=np.concatenate([[-1],np.array(parents)+1])
            names=['Simulation']+names
            if len(bone_parents)==0:
                bone_parents=parents
                bone_names=names
            else:
                assert np.all(bone_parents==parents)
                assert bone_names==names

            # Compute velocities via central difference
            velocities=np.empty_like(positions) # (N,J,3)
            velocities[1:-1]=(
                0.5*(positions[2:]-positions[1:-1])/self.dt+
                0.5*(positions[1:-1]-positions[:-2])/self.dt
            )
            velocities[0]=velocities[1]-(velocities[3]-velocities[2])
            velocities[-1]=velocities[-2]+(velocities[-2]-velocities[-3])

            # Angular velocities
            angular_velocities=np.zeros_like(velocities) # (N,J,3)
            for j in range(angular_velocities.shape[1]):
                angular_velocities[1:-1,j]=(
                    0.5*((R.from_quat(rotations[2:,j])*(R.from_quat(rotations[1:-1,j]).inv())).as_rotvec()/self.dt)+
                    0.5*((R.from_quat(rotations[1:-1,j])*(R.from_quat(rotations[:-2,j]).inv())).as_rotvec()/self.dt)
                )
            angular_velocities[0]=angular_velocities[1]-(angular_velocities[3]-angular_velocities[2])
            angular_velocities[-1]=angular_velocities[-2]+(angular_velocities[-2]-angular_velocities[-3])

            # append to database
            bone_positions.append(positions)
            bone_velocities.append(velocities)
            bone_rotations.append(rotations)
            
            bone_angular_velocities.append(angular_velocities)

            offset=0 if len(range_starts)==0 else range_stops[-1]
            range_starts.append(offset)
            range_stops.append(offset+len(positions))

    
        self.bone_positions=np.concatenate(bone_positions,axis=0).astype(np.float32)
        self.bone_velocities=np.concatenate(bone_velocities,axis=0).astype(np.float32)
        self.bone_rotations=np.concatenate(bone_rotations,axis=0).astype(np.float32)
        self.bone_angular_velocities=np.concatenate(bone_angular_velocities,axis=0).astype(np.float32)
        self.bone_parents=np.array(bone_parents).astype(np.int32)
        self.bone_names=bone_names
        self.range_starts=np.array(range_starts).astype(np.int32)
        self.range_stops=np.array(range_stops).astype(np.int32)
        self.nframes=self.bone_positions.shape[0]
    
    def _build_matching_features(self):
        offset=0
        offset=self._compute_bone_position_feature(offset,'lToeJoint',self.feature_weight_foot_position)
        offset=self._compute_bone_position_feature(offset,'rToeJoint',self.feature_weight_foot_position)
        offset=self._compute_bone_velocity_feature(offset,'lToeJoint',self.feature_weight_foot_velocity)
        offset=self._compute_bone_velocity_feature(offset,'rToeJoint',self.feature_weight_foot_velocity)
        offset=self._compute_bone_velocity_feature(offset,'RootJoint',self.feature_weight_hip_velocity)
        offset=self._compute_trajectory_position_feature(offset,self.feature_weight_trajectory_positions)
        offset=self._compute_trajectory_direction_feature(offset,self.feature_weight_trajectory_directions)

        assert offset==self.n_features
        self._build_bounds()
    
    def _build_bounds(self):
        nbound_sm=(self.nframes+self.BOUND_SM_SIZE-1)//self.BOUND_SM_SIZE
        nbound_lr=(self.nframes+self.BOUND_LR_SIZE-1)//self.BOUND_LR_SIZE

        self.bound_sm_min=np.full((nbound_sm,self.n_features),np.inf,dtype=np.float32)
        self.bound_sm_max=np.full((nbound_sm,self.n_features),-np.inf,dtype=np.float32)
        self.bound_lr_min=np.full((nbound_lr,self.n_features),np.inf,dtype=np.float32)
        self.bound_lr_max=np.full((nbound_lr,self.n_features),-np.inf,dtype=np.float32)

        for i in range(self.nframes):
            i_sm=i//self.BOUND_SM_SIZE
            i_lr=i//self.BOUND_LR_SIZE
            self.bound_sm_min[i_sm]=np.minimum(self.bound_sm_min[i_sm],self.features[i])
            self.bound_sm_max[i_sm]=np.maximum(self.bound_sm_max[i_sm],self.features[i])
            self.bound_lr_min[i_lr]=np.minimum(self.bound_lr_min[i_lr],self.features[i])
            self.bound_lr_max[i_lr]=np.maximum(self.bound_lr_max[i_lr],self.features[i])

    def _compute_bone_position_feature(self,offset,bone_name,weight):
        for i in range(self.nframes):
            bone_position,bone_rotation=self.forward_kinematics(
                self.bone_positions[i],
                self.bone_rotations[i],
                self.bone_parents,
                self.bone_names.index(bone_name)
            )
            # position relative to the simulation root
            bone_position=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                bone_position-self.bone_positions[i,0]
            )
            self.features[i,offset:offset+3]=bone_position
        self._normalize_feature(offset,3,weight)
        offset+=3
        
        return offset
            
    def _compute_bone_velocity_feature(self,offset,bone_name,weight):
        for i in range(self.nframes):
            bone_position,bone_velocity,bone_rotation,bone_angular_velocity=\
                self.forward_kinematics_velocity(
                self.bone_positions[i],
                self.bone_velocities[i],
                self.bone_rotations[i],
                self.bone_angular_velocities[i],
                self.bone_parents,
                self.bone_names.index(bone_name)
            )
            # velocity relative to the simulation root
            bone_velocity=R.from_quat(self.bone_rotations[i,0]).inv().apply(bone_velocity)
            
            self.features[i,offset:offset+3]=bone_velocity
        self._normalize_feature(offset,3,weight)
        offset+=3
        
        return offset

    def _compute_trajectory_position_feature(self,offset,weight):
        for i in range(self.nframes):
            t0=self._trajectory_index_clamp(i,20)
            t1=self._trajectory_index_clamp(i,40)
            t2=self._trajectory_index_clamp(i,60)
            t3=self._trajectory_index_clamp(i,80)
            t4=self._trajectory_index_clamp(i,100)

            traj_pos0=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                self.bone_positions[t0,0]-self.bone_positions[i,0]
            )
            traj_pos1=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                self.bone_positions[t1,0]-self.bone_positions[i,0]
            )
            traj_pos2=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                self.bone_positions[t2,0]-self.bone_positions[i,0]
            )
            traj_pos3=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                self.bone_positions[t3,0]-self.bone_positions[i,0]
            )
            traj_pos4=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                self.bone_positions[t4,0]-self.bone_positions[i,0]
            )
            self.features[i,offset:offset+10]=np.array(
                [
                    traj_pos0[0],traj_pos0[2],
                    traj_pos1[0],traj_pos1[2],
                    traj_pos2[0],traj_pos2[2],
                    traj_pos3[0],traj_pos3[2],
                    traj_pos4[0],traj_pos4[2]
                ]
            )
        self._normalize_feature(offset,10,weight)
        offset+=10
        return offset

    def _compute_trajectory_direction_feature(self,offset,weight):
        for i in range(self.nframes):
            t0=self._trajectory_index_clamp(i,20)
            t1=self._trajectory_index_clamp(i,40)
            t2=self._trajectory_index_clamp(i,60)
            t3=self._trajectory_index_clamp(i,80)
            t4=self._trajectory_index_clamp(i,100)

            traj_dir0=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                R.from_quat(self.bone_rotations[t0,0]).apply(np.array([0.0,0.0,1.0]))
            )
            traj_dir1=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                R.from_quat(self.bone_rotations[t1,0]).apply(np.array([0.0,0.0,1.0]))
            )
            traj_dir2=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                R.from_quat(self.bone_rotations[t2,0]).apply(np.array([0.0,0.0,1.0]))
            )
            traj_dir3=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                R.from_quat(self.bone_rotations[t3,0]).apply(np.array([0.0,0.0,1.0]))
            )
            traj_dir4=R.from_quat(self.bone_rotations[i,0]).inv().apply(
                R.from_quat(self.bone_rotations[t4,0]).apply(np.array([0.0,0.0,1.0]))
            )

            self.features[i,offset:offset+10]=np.array(
                [
                    traj_dir0[0],traj_dir0[2],
                    traj_dir1[0],traj_dir1[2],
                    traj_dir2[0],traj_dir2[2],
                    traj_dir3[0],traj_dir3[2],
                    traj_dir4[0],traj_dir4[2]
                ]
            )
        self._normalize_feature(offset,10,weight)
        offset+=10
        return offset

    def _normalize_feature(self,offset,size,weight):
        """
        normalize each feature dimension
        """
        # mean
        self.features_offset[offset:offset+size]=np.mean(self.features[:,offset:offset+size],axis=0)

        # std
        self.features_scale[offset:offset+size]=np.std(self.features[:,offset:offset+size],axis=0)/weight

        # normalize
        self.features[:,offset:offset+size]-=self.features_offset[offset:offset+size]
        self.features[:,offset:offset+size]/=self.features_scale[offset:offset+size]

    def _trajectory_index_clamp(self,frame,offset):
        """
        When we add an offset to a frame in the database there is a chance
        it will go out of the relevant range so here we can clamp it to 
        the last frame of that range.
        """
        for i in range(len(self.range_starts)):
            if frame>=self.range_starts[i] and frame<self.range_stops[i]:
                return clamp(frame+offset,self.range_starts[i],self.range_stops[i]-1)
        # Should never come here
        assert False

    def get_denormalized_feature(self,frame_index):
        """
        get the denormalized feature of frame_index
        """
        return self.features[frame_index]*self.features_scale+self.features_offset
    @staticmethod
    def forward_kinematics(
            bone_positions,
            bone_rotations,
            bone_parents,
            bone_index
        ):
        """
        do FK for i-th frame,return the global position and orientation of bone
        """
        global_positions=[]
        global_orientations=[]
        for i in range(len(bone_parents)):
            if bone_parents[i]==-1:
                global_positions.append(bone_positions[i])
                global_orientations.append(bone_rotations[i])
            else:
                orientation=(R.from_quat(global_orientations[bone_parents[i]])*R.from_quat(bone_rotations[i])).as_quat()
                position=R.from_quat(global_orientations[bone_parents[i]]).apply(bone_positions[i])+\
                        global_positions[bone_parents[i]]
                global_positions.append(position)
                global_orientations.append(orientation)
            if i==bone_index:
                break
        return global_positions[bone_index],global_orientations[bone_index]

    @staticmethod
    def forward_kinematics_velocity(
        bone_positions,
        bone_velocities,
        bone_rotations,
        bone_angular_velocities,
        bone_parents,
        bone_index
    ):
        """
        do FK for i-th frame,return global position,velocity,orientation and angular velocity of bone
        """
        global_positions=[]
        global_velocities=[]
        global_orientations=[]
        global_angular_velocities=[]
        for i in range(len(bone_parents)):
            if bone_parents[i]==-1:
                global_positions.append(bone_positions[i])
                global_velocities.append(bone_velocities[i])
                global_orientations.append(bone_rotations[i])
                global_angular_velocities.append(bone_angular_velocities[i])
            else:
                orientation=(R.from_quat(global_orientations[bone_parents[i]])*R.from_quat(bone_rotations[i])).as_quat()
                position=R.from_quat(global_orientations[bone_parents[i]]).apply(bone_positions[i])+\
                        global_positions[bone_parents[i]]
                velocity=global_velocities[bone_parents[i]]+\
                        R.from_quat(global_orientations[bone_parents[i]]).apply(bone_velocities[i])+\
                        np.cross(global_angular_velocities[bone_parents[i]],
                                 R.from_quat(global_orientations[bone_parents[i]]).apply(bone_positions[i]))
                angular_velocity=global_angular_velocities[bone_parents[i]]+\
                        R.from_quat(global_orientations[bone_parents[i]]).apply(bone_angular_velocities[i])
                global_positions.append(position)
                global_velocities.append(velocity)
                global_orientations.append(orientation)
                global_angular_velocities.append(angular_velocity)
            if i==bone_index:
                break
        return global_positions[bone_index],global_velocities[bone_index],\
                global_orientations[bone_index],global_angular_velocities[bone_index]
    
    @staticmethod
    def forward_kinematics_full(
        bone_positions,
        bone_rotations,
        bone_parents
    ):
        """
        do FK for i-th frame,return global position and orientation of all bones
        """
        global_positions=[]
        global_orientations=[]
        for i in range(len(bone_parents)):
            if bone_parents[i]==-1:
                global_positions.append(bone_positions[i])
                global_orientations.append(bone_rotations[i])
            else:
                orientation=(R.from_quat(global_orientations[bone_parents[i]])*R.from_quat(bone_rotations[i])).as_quat()
                position=R.from_quat(global_orientations[bone_parents[i]]).apply(bone_positions[i])+\
                        global_positions[bone_parents[i]]
                global_positions.append(position)
                global_orientations.append(orientation)
        global_positions=np.array(global_positions)
        global_orientations=np.array(global_orientations)
        return global_positions,global_orientations
    
