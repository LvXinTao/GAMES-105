from bvh_utils import *
#---------------你的代码------------------#
# translation 和 orientation 都是全局的
def skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = T_pose_vertex_translation.copy()
    
    #---------------你的代码------------------#

    # # Get the local coordinate of each vertex
    # The looping version is too slow!! (3.0 fps)
    # for vertex_id in range(skinning_idx.shape[0]):
    #     skinning_idx_vertex=skinning_idx[vertex_id]
    #     skinning_weight_vertex=skinning_weight[vertex_id] # (4,)
    #     T_pose_vertex_position=T_pose_vertex_translation[vertex_id]
    #     T_pose_joint_position=T_pose_joint_translation[skinning_idx_vertex]
    #     # Get the local coordinate of each vertex
    #     T_pose_vertex_local=T_pose_vertex_position-T_pose_joint_position
    #     # Get the position in the world coordinate
    #     global_position=R.from_quat(joint_orientation[skinning_idx_vertex]).apply(T_pose_vertex_local)+joint_translation[skinning_idx_vertex] # (4,3)
    #     vertex_translation[vertex_id]=np.sum(global_position*skinning_weight_vertex[:,np.newaxis],axis=0)

    # The Numpy Vectorized version is way much faster (60 fps or more)
    T_pose_vertex_local=T_pose_vertex_translation[:,np.newaxis,:]-T_pose_joint_translation[skinning_idx] # (N,4,3)
    global_position=R.from_quat(joint_orientation[skinning_idx].reshape(-1,4)).apply(T_pose_vertex_local.reshape(-1,3)).reshape(-1,4,3)+joint_translation[skinning_idx] # (N,4,3)
    vertex_translation=np.sum(global_position*skinning_weight[:,:,np.newaxis],axis=1).squeeze() # (N,3)

    return vertex_translation