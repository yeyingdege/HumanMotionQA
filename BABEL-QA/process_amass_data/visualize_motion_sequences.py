import argparse
import os.path as osp
import numpy as np
import time
import os
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def vis_joints(babel_id, data_dir, center_subject=True):

    joints_path = osp.join(data_dir, 'motion_sequences', babel_id, 'joints.npy')
    joints = np.load(joints_path)
    if center_subject: joints[:,:,:2] =  joints[:,:,:2] - joints[:,:1,:2]
    joints = np.expand_dims(joints, axis=0)

    visualize_out_dir = osp.join(data_dir, 'vis', 'joint_vis', f'{babel_id}_{time.strftime("%Y-%m-%d-%H-%M-%S")}')

    if not osp.exists(visualize_out_dir):
        os.makedirs(visualize_out_dir)

    if center_subject:
        fig = plt.figure(figsize=(9, 7))
    else:
        fig = plt.figure(figsize=(9*3, 7*3))
    ax = Axes3D(fig) 
    ax.grid(False)
    plt.axis('off')

    color_list = ['#27AE60', '#E74C3C', '#E76F51', '#F4A261'] # green, red, red orange, soft orange
    if joints.shape[0] ==1:
        color_list = ['#E74C3C'] # Prediction lnly setting, use red color for the predicted skeleton 
        color_list = ['#004385'] # dark blue
        color_list = ['#000000'] # black
    vals = joints # K X T X 24 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]
    # SMPL connections 22 joints 
    connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]]

    lines = []
    for cmp_idx in range(num_cmp):
        cur_line = []
        for ind, (i,j) in enumerate(connections):
            cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2*3, c=color_list[cmp_idx])[0])
        lines.append(cur_line)
  
    def animate(i):
        changed = []
        for ai in range(len(vals)):
            for ind, (p_idx, j_idx) in enumerate(connections):
                lines[ai][ind].set_data_3d([vals[ai][i, j_idx, 0], vals[ai][i, p_idx, 0]],
                                           [vals[ai][i, j_idx, 1], vals[ai][i, p_idx, 1]],
                                           [vals[ai][i, j_idx, 2], vals[ai][i, p_idx, 2]])
            changed += lines
        return changed

    if center_subject:
        RADIUS = 2 / 3
    else:
        RADIUS = 2  # space around the subject
    xroot, yroot, zroot = vals[0, 0, 0, 0], vals[0, 0, 0, 1], vals[0, 0, 0, 2]
    # xroot, yroot, zroot = 0, 0, 0 # For debug
       
    ax.view_init(0, 120) # Used in training AMASS dataset
          
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # ax.set_axis_off()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    ani = FuncAnimation(fig,                                            
                        animate,                                        
                        np.arange(len(vals[0])),                  
                        interval=33.33)  

    # ani.save(ospj(out_dir, 'joints.mp4'),                       
    #         writer="ffmpeg",                                                
    #         fps=30) 

    ani.save(osp.join(visualize_out_dir, f'joints.gif'),                       
        writer="ffmpeg",                                                
        fps=30, savefig_kwargs={"transparent": True}) 

    plt.cla()
    plt.close()

    # create_gif_from_video(ospj(out_dir, 'joints.mp4'), out_path=ospj(out_dir, 'joints.gif'), fps=30)


def vis_joints_fig(babel_id, data_dir, center_subject=True, sample_seq=False):
    
    joints_path = osp.join(data_dir, 'motion_sequences', babel_id, 'joints.npy')
    joints = np.load(joints_path)
    if center_subject: joints[:,:,:2] =  joints[:,:,:2] - joints[:,:1,:2]
    # randomly sample 12 sequence
    if sample_seq:
        seq_idx = random.sample(range(joints.shape[0]), 12)
        joints = joints[seq_idx]
    joints = np.expand_dims(joints, axis=0)

    # visualize_out_dir = osp.join(data_dir, 'vis', 'joint_vis', f'{babel_id}_{time.strftime("%Y-%m-%d-%H-%M-%S")}')
    visualize_out_dir = osp.join(data_dir, 'vis', 'joint_vis')

    if not osp.exists(visualize_out_dir):
        os.makedirs(visualize_out_dir)

    color_list = ['#27AE60', '#E74C3C', '#E76F51', '#F4A261'] # green, red, red orange, soft orange
    if joints.shape[0] == 1:
        color_list = ['#E74C3C'] # Prediction lnly setting, use red color for the predicted skeleton 
        color_list = ['#004385'] # dark blue
        color_list = ['#000000'] # black
    vals = joints # K X T X 24 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]
    # SMPL connections 22 joints 
    connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]]

    if center_subject:
        figsize=(12, 3)
        line_w = 1.5
    else:
        figsize=(9*3, 3*3)
        line_w = 2.5
    fig, axs = plt.subplots(num_cmp, joints.shape[1], figsize=figsize, sharey=True)
    plt.subplots_adjust(wspace=0)

    for cmp_idx in range(num_cmp):
        for ind, ax in enumerate(axs):
            for p_idx, j_idx in connections:
                curr_x = [vals[cmp_idx][ind, j_idx, 1], vals[cmp_idx][ind, p_idx, 1]]
                curr_y = [vals[cmp_idx][ind, j_idx, 2], vals[cmp_idx][ind, p_idx, 2]]
                ax.plot(curr_x, curr_y, lw=line_w, c=color_list[cmp_idx])
            ax.set_axis_off()
            ax.set_ylim(0, 2)
            ax.set_xlim(-0.6, 0.6)
            ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig.savefig(osp.join(visualize_out_dir, f'joints_{babel_id}.png'))
    plt.cla()
    plt.close()


def vis_mesh():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_babel_ids_path', default="BABEL-QA/process_amass_data/vis_babel_ids.txt", help='Path to BABEL ids that you want to visualize.')
    parser.add_argument('--data_dir', default="data/babel-qa", help='Root directory of BABEL-QA dataset')
    parser.add_argument('--vis_mesh', action='store_true', help='Create mesh visualization')
    parser.add_argument('--vis_joints', action='store_true', default=True, help='Create joint visualization')

    args = parser.parse_args()

    vis_babel_ids = list(open(args.vis_babel_ids_path, 'r').read().split('\n'))

    for babel_id in vis_babel_ids:
        if args.vis_joints:
            vis_joints_fig(babel_id, args.data_dir, sample_seq=True)
        if args.vis_mesh:
            vis_mesh()
    
if __name__ == '__main__':
    main()