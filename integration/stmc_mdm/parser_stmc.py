from utils.parser_util import (
    ArgumentParser,
    add_base_options,
    # add_sampling_options,
    add_generate_options,
    # parse_and_load_from_model,
    add_data_options,
    add_model_options,
    add_diffusion_options,
    get_args_per_group_name,
    get_model_path_from_args,
    os,
    json,
    #
    get_cond_mode,
)


# same function as in utils.parser_util
# but does not reparse the model_path for some reason
# otherwise cannot use a default model_path
def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    ### Only changed this line
    model_path = args.model_path  # get_model_path_from_args()
    ###
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif "cond_mode" in model_args:  # backward compitability
            unconstrained = model_args["cond_mode"] == "no_cond"
            setattr(args, "unconstrained", unconstrained)

        else:
            print(
                "Warning: was not able to load [{}], using default value [{}] instead.".format(
                    a, args.__dict__[a]
                )
            )

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


# same function, but take a model path by default
# fmt: off
def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path",
                       default="./save/humanml_trans_enc_512/model000200000.pt",
                       type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
# fmt: on


# Same function as in utils/parser_util.py
# but I added stmc options
def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_stmc_options(parser)

    args = parse_and_load_from_model(parser)

    cond_mode = get_cond_mode(args)
    assert cond_mode == "text"
    assert args.dataset == "humanml"
    # only tested for this dataset

    if args.input_text or args.text_prompt:
        raise Exception(
            "Arguments input_text and text_prompt should not be used for timeline conditionning."
        )
    return args


def add_stmc_options(parser):
    group = parser.add_argument_group("stmc")
    group.add_argument(
        "--interval_overlap",
        default=0.5,
        type=float,
        help="Overlap (in seconds) for the diffcollage per body part",
    )
    group.add_argument(
        "--input_timeline",
        default="",
        type=str,
        help="Path to a timeline file.",
    )
    group.add_argument(
        "--stmc_baseline",
        default="none",
        type=str,
        choices=["none", "sinc", "sinc_lerp", "onetext", "singletrack"],
        help="Use sinc baseline or not.",
    )
