import os
import torch
import numpy as np
import random


class AMASSMotionLoader:
    def __init__(
        self, base_dir, fps, disable: bool = False, nfeats=None, umin_s=0.5, umax_s=3.0
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.disable = disable
        self.nfeats = nfeats

        # unconditional, sampling the duration from [umin, umax]
        self.umin = int(self.fps * umin_s)
        assert self.umin > 0
        self.umax = int(self.fps * umax_s)

    def __call__(self, path, start, end, drop_motion_perc=None, load_transition=False):
        if self.disable:
            return {"x": path, "length": int(self.fps * (end - start))}

        # load the motion
        if path not in self.motions:
            motion_path = os.path.join(self.base_dir, path + ".npy")
            motion = np.load(motion_path)
            motion = torch.from_numpy(motion).to(torch.float)
            self.motions[path] = motion

        if load_transition:
            motion = self.motions[path]
            # take a random crop
            duration = random.randint(self.umin, min(self.umax, len(motion)))
            # random start
            start = random.randint(0, len(motion) - duration)
            motion = motion[start : start + duration]
        else:
            begin = int(start * self.fps)
            end = int(end * self.fps)

            motion = self.motions[path][begin:end]

            # crop max X% of the motion randomly beginning and end
            if drop_motion_perc is not None:
                max_frames_to_drop = int(len(motion) * drop_motion_perc)
                # randomly take a number of frames to drop
                n_frames_to_drop = random.randint(0, max_frames_to_drop)

                # split them between left and right
                n_frames_left = random.randint(0, n_frames_to_drop)
                n_frames_right = n_frames_to_drop - n_frames_left

                # crop the motion
                motion = motion[n_frames_left:-n_frames_right]

        x_dict = {"x": motion, "length": len(motion)}
        return x_dict
