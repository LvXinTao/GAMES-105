import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def inv_safe(data):
    # return R.from_quat(data).inv()
    if np.allclose(data, [0, 0, 0, 0]):
        return np.eye(3)
    else:
        return np.linalg.inv(R.from_quat(data).as_matrix())

def from_quat_safe(data):
    # return R.from_quat(data)
    if np.allclose(data, [0, 0, 0, 0]):
        return np.eye(3)
    else:
        return R.from_quat(data).as_matrix()
    
class MetaData:
    def __init__(self, joint_name, joint_parent, joint_initial_position, root_joint, end_joint):
        """
        一些固定信息，其中joint_initial_position是T-pose下的关节位置，可以用于计算关节相互的offset
        root_joint是固定节点的索引，并不是RootJoint节点
        """
        self.joint_name = joint_name
        self.joint_parent = joint_parent
        self.joint_initial_position = joint_initial_position
        self.root_joint = root_joint
        self.end_joint = end_joint

        # 从end节点开始，一直往上找，直到找到腰部节点
        path1 = [self.joint_name.index(self.end_joint)]
        while self.joint_parent[path1[-1]] != -1:
            path1.append(self.joint_parent[path1[-1]])
            
        # 从root节点开始，一直往上找，直到找到腰部节点
        path2 = [self.joint_name.index(self.root_joint)]
        while self.joint_parent[path2[-1]] != -1:
            path2.append(self.joint_parent[path2[-1]])
        
        # 合并路径，消去重复的节点
        while path1 and path2 and path2[-1] == path1[-1]:
            path1.pop()
            a = path2.pop()
            
        path2.append(a)
        path = path2 + list(reversed(path1))
        path_name = [self.joint_name[i] for i in path]

        self.path=path
        self.path_name=path_name
        self.path1=path1
        self.path2=path2

    def get_path_from_root_to_end(self):
        """
        辅助函数，返回从root节点到end节点的路径
        
        输出：
            path: 各个关节的索引
            path_name: 各个关节的名字
        Note: 
            如果root_joint在脚，而end_joint在手，那么此路径会路过RootJoint节点。
            在这个例子下path2返回从脚到根节点的路径，path1返回从根节点到手的路径。
            你可能会需要这两个输出。
        """
          
        return self.path, self.path_name, self.path1, self.path2
    
def get_path_info(meta_data:MetaData,joint_positions,joint_orientations):
    """
    get infomation along the kinematic chain
    return 
    path_offsets
    path_positions
    path_orientations
    """
    path_len=len(meta_data.path)
    path_positions = []
    path_orientations = []
    path_offsets = []
    for joint in meta_data.path:
        path_positions.append(joint_positions[joint])
        path_orientations.append(R.from_quat(joint_orientations[joint]))
    path_offsets.append((0,0,0)) # zero offset for root
    for i in range(path_len-1):
        path_offsets.append(
            meta_data.joint_initial_position[meta_data.path[i+1]]-\
            meta_data.joint_initial_position[meta_data.path[i]]
        )
    return path_offsets,path_positions,path_orientations
    
def CCD(meta_data,path_offsets,path_positions,path_orientations,target_pos,thresh=0.01,max_iter=50):
    cnt=0
    end_index=len(meta_data.path)-1
    while(np.linalg.norm(path_positions[end_index]-target_pos,2)>=thresh and cnt<=max_iter):
        # do CCD from end to root
        for cur_index in range(end_index-1,-1,-1): # from end_index-1 to 0
            end_pos=path_positions[end_index]
            cur_pos=path_positions[cur_index]

            vec_cur2end=end_pos-cur_pos
            vec_cur2tgt=target_pos-cur_pos
            cur2end=vec_cur2end/np.linalg.norm(vec_cur2end)
            cur2tgt=vec_cur2tgt/np.linalg.norm(vec_cur2tgt)

            rotation_radius=np.arccos(np.clip(np.dot(cur2end,cur2tgt),-1,1))/2
            rotation_axis=np.cross(cur2end,cur2tgt)
            rotation_axis=rotation_axis/np.linalg.norm(rotation_axis)
            rotation_vector=R.from_rotvec(rotation_axis*rotation_radius)

            # Update current orientation
            path_orientations[cur_index]=rotation_vector*path_orientations[cur_index]

            # update cur_index to end_index 's orientations and positions
            # first get local rotation in the path
            path_rotations=[]
            path_rotations.append(path_orientations[0])
            for i in range(end_index):
                path_rotations.append(R.inv(path_orientations[i])*path_orientations[i+1])
            # update positions and orientations
            for j in range(cur_index,end_index):
                path_positions[j+1]=path_positions[j]+path_orientations[j].apply(path_offsets[j+1])
                if j+1<end_index:
                    path_orientations[j+1]=path_orientations[j]*path_rotations[j+1]
                else:
                    path_orientations[j+1]=path_orientations[j]
        cnt+=1
    return path_positions,path_orientations

def calculateJointAngle(path_orientations):
    path_rotations = []
    path_rotations.append(path_orientations[0])
    # update joint rotations R_{i} = Q_{i-1}^T Q_{i}
    for j in range(len(path_orientations) - 1):
        rot = R.inv(path_orientations[j]) * path_orientations[j + 1]
        path_rotations.append(rot)
    # decompose euler angle
    joint_angle = []
    for r in path_rotations:
        eula = R.from_matrix(r.as_matrix()).as_euler('XYZ', degrees=True)
        joint_angle.append(eula)
    return joint_angle

def calculateJacobian(end_position, joint_angle, path_positions, path_orientations):
    # fill jacobian matrix
    # i'th column = a_{i} x r{i}
    # for XYZ ball joint, use Euler angle to decomposite: R = Rx Ry Rz
    jacobian = []

    for i in range(len(joint_angle)):
        # print(path_positions[i], joint_angle[i+1])
        current_position = path_positions[i]
        current_angle = joint_angle[i]
        r = current_position - end_position
        rx = R.from_euler('XYZ', [current_angle[0],  0., 0.], degrees=True)
        rxy = R.from_euler('XYZ', [current_angle[0],  current_angle[1], 0.], degrees=True)
        q_prev = None
        if i == 0:
            q_prev = R.from_quat([ 0., 0., 0., 1.])
        else:
            q_prev = path_orientations[i-1]
        ex = np.array([1., 0., 0.]).reshape(-1, 3)
        ey = np.array([0., 1., 0.]).reshape(-1, 3)
        ez = np.array([0., 0., 1.]).reshape(-1, 3)
        ax = q_prev.apply(ex)
        ay = q_prev.apply(rx.apply(ey))
        az = q_prev.apply(rxy.apply(ez))
        jacobian.append(np.cross(ax, r))
        jacobian.append(np.cross(ay, r))
        jacobian.append(np.cross(az, r))
    jacobian = np.concatenate(jacobian, axis=0).transpose()
    return jacobian

def calculateJointPathInJacobian(theta, end_index, path_offsets, path_positions, path_orientations):
    path_rotations = []
    theta = theta.reshape(-1,3)
    for i in range(len(theta)):
        eula = theta[i]
        path_rotations.append(R.from_euler('XYZ', eula, degrees=True))

    # update joint rotations R_{i} = Q_{i-1}^T Q_{i}
    path_orientations[0] = path_rotations[0]
    for j in range(len(path_positions) - 1):
        path_positions[j + 1] = path_positions[j] + path_orientations[j].apply(path_offsets[j + 1])
        if j + 1 < end_index:
            path_orientations[j + 1] = path_orientations[j] * path_rotations[j + 1]
        else:
            path_orientations[j + 1] = path_orientations[j]
    return path_positions, path_orientations

def gradientDescent(meta_data, path_offsets,path_positions,path_orientations, target_pose,thresh=0.01,max_iter=50):
 
    end_index = meta_data.path_name.index(meta_data.end_joint)
    count = 0
    while (np.linalg.norm(path_positions[end_index] - target_pose) >= thresh and count <= max_iter):
        end_position = path_positions[end_index]
        joint_angle = calculateJointAngle(path_orientations)
        jacobian = calculateJacobian(end_position, joint_angle, path_positions, path_orientations)
        delta = np.array(target_pose - end_position).reshape(3, -1)

        # get all path rotations, convert to XYZ euler angle
        theta = np.concatenate(joint_angle, axis=0).transpose().reshape(-1, 1)

        t1 = np.dot(jacobian, jacobian.transpose())
        t2 = np.dot(t1, delta)
        alpha = 32 * np.sum(t2 * delta) / (np.linalg.norm(t2) * np.linalg.norm(t2))

        # print(t2, delta, alpha)

        # theta_i+1 = theta_i - alpha J^T delta 
        delta = alpha * np.dot(jacobian.transpose(), delta)
        theta = theta - delta

        # convert theta back to rotations
        path_positions, path_orientations = calculateJointPathInJacobian(theta, end_index, path_offsets, path_positions, path_orientations)

        alpha = alpha * 0.8
        count = count + 1

    return path_positions, path_orientations

def apply_fullbody_ik(meta_data:MetaData,joint_positions,joint_orientations,path_positions,path_orientations):
    """
    apply fullbody ik to get joint_positions and joint_orientations
    """
    # get local rotations
    joint_rotations=[]
    for i in range(len(meta_data.joint_name)):
        if i==0:
            joint_rotations.append(R.from_quat(joint_orientations[i]))
        else:
            joint_rotations.append(R.from_matrix(from_quat_safe(joint_orientations[meta_data.joint_parent[i]])).inv()*R.from_matrix(from_quat_safe(joint_orientations[i])))
    # apply IK rotation result
    if len(meta_data.path2)>1:
        for i in range(len(meta_data.path2)-1):
            joint_orientations[meta_data.path2[i+1]]=path_orientations[i].as_quat() # reverse from foot to root
        joint_orientations[meta_data.path2[-1]]=path_orientations[len(meta_data.path2)-1].as_quat() # for hips
        for i in range(len(meta_data.path1)-1):
            joint_orientations[meta_data.path[i+len(meta_data.path2)]]=path_orientations[i+len(meta_data.path2)].as_quat()
    else:
        for i in range(len(meta_data.path)):
            joint_orientations[meta_data.path[i]]=path_orientations[i].as_quat()
    # apply IK position result
    for i in range(len(meta_data.path)):
        joint_positions[meta_data.path[i]]=path_positions[i]
    # FK to get other joints
    for i in range(len(meta_data.joint_name)):
        if i==0:
            continue
        if meta_data.joint_name[i] not in meta_data.path_name:
            joint_positions[i]=joint_positions[meta_data.joint_parent[i]]+R.from_quat(joint_orientations[meta_data.joint_parent[i]])\
                .apply(meta_data.joint_initial_position[i]-meta_data.joint_initial_position[meta_data.joint_parent[i]])
            joint_orientations[i]=(R.from_quat(joint_orientations[meta_data.joint_parent[i]])*joint_rotations[i]).as_quat()
    return joint_positions,joint_orientations
 
def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pos):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pos: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    # two steps
    # CCD or Jacobian or FABRIK... along the kinematic chain
    # apply FullBody IK
    path_offsets,path_positions,path_orientations=get_path_info(meta_data,joint_positions,joint_orientations)
    path_positions,path_orientations=CCD(meta_data,path_offsets,path_positions,path_orientations,target_pos)
    # path_positions,path_orientations=gradientDescent(meta_data,path_offsets,path_positions,path_orientations,target_pos)
    joint_positions,joint_orientations=apply_fullbody_ik(meta_data,joint_positions,joint_orientations,path_positions,path_orientations)

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    target_pos=np.array([joint_positions[0][0]+relative_x,target_height,joint_positions[0][2]+relative_z])
    path_offsets,path_positions,path_orientations=get_path_info(meta_data,joint_positions,joint_orientations)
    # path_positions,path_orientations=CCD(meta_data,path_offsets,path_positions,path_orientations,target_pos)
    path_positions,path_orientations=gradientDescent(meta_data,path_offsets,path_positions,path_orientations,target_pos)
    joint_positions,joint_orientations=apply_fullbody_ik(meta_data,joint_positions,joint_orientations,path_positions,path_orientations)
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    cnt=0
    meta_data_l=MetaData(meta_data.joint_name,meta_data.joint_parent,meta_data.joint_initial_position,'lToeJoint_end','lWrist_end')
    meta_data_r=MetaData(meta_data.joint_name,meta_data.joint_parent,meta_data.joint_initial_position,'lToeJoint_end','rWrist_end')
    while cnt<=100:
        
        path_offsets_l,path_positions_l,path_orientations_l=get_path_info(meta_data_l,joint_positions,joint_orientations)
        # path_positions_l,path_orientations_l=CCD(meta_data_l,path_offsets_l,path_positions_l,path_orientations_l,left_target_pose)
        path_positions_l,path_orientations_l=gradientDescent(meta_data_l,path_offsets_l,path_positions_l,path_orientations_l,left_target_pose)
        joint_positions,joint_orientations=apply_fullbody_ik(meta_data_l,joint_positions,joint_orientations,path_positions_l,path_orientations_l)

        
        path_offsets_r,path_positions_r,path_orientations_r=get_path_info(meta_data_r,joint_positions,joint_orientations)
        # path_positions_r,path_orientations_r=CCD(meta_data_r,path_offsets_r,path_positions_r,path_orientations_r,right_target_pose)
        path_positions_r,path_orientations_r=gradientDescent(meta_data_r,path_offsets_r,path_positions_r,path_orientations_r,right_target_pose)
        joint_positions,joint_orientations=apply_fullbody_ik(meta_data_r,joint_positions,joint_orientations,path_positions_r,path_orientations_r)
        cnt+=1
    
    return joint_positions, joint_orientations