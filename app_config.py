RUN_DIR = "pretrained_models/mdm-smpl_clip_smplrifke_humanml3d"
FPS = 20.0
CKPT = "last"
SHARE = True

CONFIG = {
    "joints_renderer": {
        "_target_": "src.renderer.matplotlib.MatplotlibRender",
        "jointstype": "guoh3djoints",
        "fps": FPS,
        "colors": ["black", "magenta", "red", "green", "blue"],
        "figsize": 4,
        "canonicalize": True,
    },
    "smpl_renderer": {
        "_target_": "src.renderer.humor.HumorRenderer",
        "fps": FPS,
        "imw": 720,
        "imh": 720,
    },
    "logger_level": "INFO",
    "ckpt": CKPT,
    "run_dir": RUN_DIR,
}
