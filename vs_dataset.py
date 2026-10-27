import glob
import os

import torch

from vs_config import (
    LENGTH_MIN,
    LENGTH_MAX,
    DEPTH_MIN,
    DEPTH_MAX,
)

from vs_representation import (
    actual_to_canonical,
    make_state_representation,
)


# ============================================================
# SHAPE PARAM NORMALIZATION
#
# Paper:
#   x, y, theta -> pose / warp
#   l, z        -> param2deform shape parameters
#
# We normalize only:
#
#   [length, depth]
#
# to [-1, 1].
# ============================================================

def normalize_shape_params(
    length,
    depth,
):

    length_range = (
        LENGTH_MAX
        -
        LENGTH_MIN
    )

    depth_range = (
        DEPTH_MAX
        -
        DEPTH_MIN
    )


    if length_range <= 0.0:

        raise ValueError(
            "LENGTH_MAX must be greater than LENGTH_MIN."
        )


    if depth_range <= 0.0:

        raise ValueError(
            "DEPTH_MAX must be greater than DEPTH_MIN."
        )


    length_norm = (
        2.0
        *
        (
            length
            -
            LENGTH_MIN
        )
        /
        length_range
        -
        1.0
    )


    depth_norm = (
        2.0
        *
        (
            depth
            -
            DEPTH_MIN
        )
        /
        depth_range
        -
        1.0
    )


    return torch.stack(
        [
            length_norm,
            depth_norm,
        ],
        dim=-1,
    )


# ============================================================
# OPTIONAL DENORMALIZATION
#
# Useful later for debugging / planner inspection.
# ============================================================

def denormalize_shape_params(
    params,
):

    if params.ndim != 2:

        raise ValueError(
            f"Expected params shape [B,2], "
            f"got {tuple(params.shape)}"
        )


    if params.shape[
        1
    ] != 2:

        raise ValueError(
            f"Expected params shape [B,2], "
            f"got {tuple(params.shape)}"
        )


    length_norm = params[
        :,
        0
    ]


    depth_norm = params[
        :,
        1
    ]


    length = (
        (
            length_norm
            +
            1.0
        )
        *
        0.5
        *
        (
            LENGTH_MAX
            -
            LENGTH_MIN
        )
        +
        LENGTH_MIN
    )


    depth = (
        (
            depth_norm
            +
            1.0
        )
        *
        0.5
        *
        (
            DEPTH_MAX
            -
            DEPTH_MIN
        )
        +
        DEPTH_MIN
    )


    return torch.stack(
        [
            length,
            depth,
        ],
        dim=-1,
    )


# ============================================================
# CHUNK HELPERS
# ============================================================

def _get_required_key(
    chunk,
    primary_key,
    alias_key,
    path,
):

    if primary_key in chunk:

        return chunk[
            primary_key
        ]


    if alias_key in chunk:

        return chunk[
            alias_key
        ]


    raise KeyError(
        f"{path}: neither '{primary_key}' "
        f"nor '{alias_key}' was found."
    )


# ============================================================
# GPU DATASET
# ============================================================

class GPUDataset:

    def __init__(
        self,
        root,
        device,
        limit=None,
    ):

        self.root = root

        self.device = device


        files = sorted(
            glob.glob(
                os.path.join(
                    root,
                    "chunk_*.pt"
                )
            )
        )


        if len(
            files
        ) == 0:

            raise RuntimeError(
                f"No dataset chunks in {root}"
            )


        # ----------------------------------------------------
        # CPU accumulation
        # ----------------------------------------------------

        states = []

        actions = []

        next_states = []


        rollout_ids = []

        rollout_steps = []


        has_rollout_metadata = True


        action_types = set()

        action_dims = set()


        print()
        print(
            "==============================================="
        )

        print(
            "LOADING PARAM2DEFORM DATASET"
        )

        print(
            "==============================================="
        )

        print(
            "root   =",
            root
        )

        print(
            "chunks =",
            len(
                files
            )
        )

        print()


        # ====================================================
        # LOAD CHUNKS
        # ====================================================

        for path in files:

            chunk = torch.load(
                path,
                map_location="cpu",
            )


            state = _get_required_key(
                chunk,
                "state",
                "states",
                path,
            )


            action = _get_required_key(
                chunk,
                "action",
                "actions",
                path,
            )


            next_state = _get_required_key(
                chunk,
                "next_state",
                "next_states",
                path,
            )


            # ------------------------------------------------
            # TYPE
            # ------------------------------------------------

            state = state.float()

            action = action.float()

            next_state = next_state.float()


            # ------------------------------------------------
            # SHAPE CHECK
            # ------------------------------------------------

            if state.ndim != 3:

                raise ValueError(
                    f"{path}: expected state [N,H,W], "
                    f"got {tuple(state.shape)}"
                )


            if next_state.ndim != 3:

                raise ValueError(
                    f"{path}: expected next_state [N,H,W], "
                    f"got {tuple(next_state.shape)}"
                )


            if state.shape != next_state.shape:

                raise ValueError(
                    f"{path}: state shape "
                    f"{tuple(state.shape)} "
                    f"!= next_state shape "
                    f"{tuple(next_state.shape)}"
                )


            if action.ndim != 2:

                raise ValueError(
                    f"{path}: expected action [N,5], "
                    f"got {tuple(action.shape)}"
                )


            if action.shape[
                1
            ] != 5:

                raise ValueError(
                    f"{path}: expected Visual Sculpting "
                    f"action [x,y,theta,length,depth], "
                    f"got {tuple(action.shape)}"
                )


            if action.shape[
                0
            ] != state.shape[
                0
            ]:

                raise ValueError(
                    f"{path}: action count "
                    f"{action.shape[0]} "
                    f"!= state count "
                    f"{state.shape[0]}"
                )


            # ------------------------------------------------
            # METADATA CHECK
            # ------------------------------------------------

            if "action_type" in chunk:

                action_types.add(
                    str(
                        chunk[
                            "action_type"
                        ]
                    )
                )


            if "action_dim" in chunk:

                action_dims.add(
                    int(
                        chunk[
                            "action_dim"
                        ]
                    )
                )


            # ------------------------------------------------
            # ROLLOUT METADATA
            #
            # Needed later for rollout-level train/val split.
            # ------------------------------------------------

            if (
                "rollout_id" in chunk
                and
                "rollout_step" in chunk
            ):

                rid = chunk[
                    "rollout_id"
                ].long()


                rstep = chunk[
                    "rollout_step"
                ].long()


                if rid.ndim != 1:

                    raise ValueError(
                        f"{path}: rollout_id must be [N]."
                    )


                if rstep.ndim != 1:

                    raise ValueError(
                        f"{path}: rollout_step must be [N]."
                    )


                if rid.shape[
                    0
                ] != state.shape[
                    0
                ]:

                    raise ValueError(
                        f"{path}: rollout_id length mismatch."
                    )


                if rstep.shape[
                    0
                ] != state.shape[
                    0
                ]:

                    raise ValueError(
                        f"{path}: rollout_step length mismatch."
                    )


                rollout_ids.append(
                    rid
                )


                rollout_steps.append(
                    rstep
                )


            else:

                has_rollout_metadata = False


            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            states.append(
                state
            )


            actions.append(
                action
            )


            next_states.append(
                next_state
            )


            print(
                f"[loaded] "
                f"{os.path.basename(path)} | "
                f"N={state.shape[0]} | "
                f"state={tuple(state.shape)} | "
                f"action={tuple(action.shape)}"
            )


        # ====================================================
        # CONCATENATE
        # ====================================================

        states = torch.cat(
            states,
            dim=0,
        )


        actions = torch.cat(
            actions,
            dim=0,
        )


        next_states = torch.cat(
            next_states,
            dim=0,
        )


        if (
            has_rollout_metadata
            and
            len(
                rollout_ids
            )
            ==
            len(
                files
            )
        ):

            rollout_ids = torch.cat(
                rollout_ids,
                dim=0,
            )


            rollout_steps = torch.cat(
                rollout_steps,
                dim=0,
            )


        else:

            rollout_ids = None

            rollout_steps = None


        # ====================================================
        # OPTIONAL LIMIT
        # ====================================================

        if limit is not None:

            limit = int(
                limit
            )


            if limit <= 0:

                raise ValueError(
                    f"limit must be > 0, got {limit}"
                )


            limit = min(
                limit,
                states.shape[
                    0
                ],
            )


            states = states[
                :limit
            ]


            actions = actions[
                :limit
            ]


            next_states = next_states[
                :limit
            ]


            if rollout_ids is not None:

                rollout_ids = rollout_ids[
                    :limit
                ]


            if rollout_steps is not None:

                rollout_steps = rollout_steps[
                    :limit
                ]


        # ====================================================
        # FINAL CHECK
        # ====================================================

        if states.shape[
            0
        ] == 0:

            raise RuntimeError(
                "Dataset contains zero samples."
            )


        # ====================================================
        # ONE GPU TRANSFER
        # ====================================================

        self.states = states.to(
            device,
            non_blocking=True,
        )


        self.actions = actions.to(
            device,
            non_blocking=True,
        )


        self.next_states = next_states.to(
            device,
            non_blocking=True,
        )


        if rollout_ids is not None:

            self.rollout_ids = rollout_ids.to(
                device,
                non_blocking=True,
            )


        else:

            self.rollout_ids = None


        if rollout_steps is not None:

            self.rollout_steps = rollout_steps.to(
                device,
                non_blocking=True,
            )


        else:

            self.rollout_steps = None


        self.size = self.states.shape[
            0
        ]


        self.has_rollout_metadata = (
            self.rollout_ids is not None
            and
            self.rollout_steps is not None
        )


        self.action_types = sorted(
            action_types
        )


        self.action_dims = sorted(
            action_dims
        )


        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print(
            "==============================================="
        )

        print(
            "PARAM2DEFORM DATASET SUMMARY"
        )

        print(
            "==============================================="
        )


        print(
            "samples          =",
            self.size
        )


        print(
            "state shape      =",
            tuple(
                self.states.shape
            )
        )


        print(
            "action shape     =",
            tuple(
                self.actions.shape
            )
        )


        print(
            "action definition= "
            "[x,y,theta,length,depth]"
        )


        print(
            "action types     =",
            self.action_types
        )


        print(
            "action dims      =",
            self.action_dims
        )


        print(
            "rollout metadata =",
            (
                "YES"
                if self.has_rollout_metadata
                else
                "NO"
            )
        )


        if self.has_rollout_metadata:

            unique_rollouts = torch.unique(
                self.rollout_ids
            )


            print(
                "rollouts         =",
                unique_rollouts.numel()
            )


            print(
                "rollout step min =",
                int(
                    self.rollout_steps.min().item()
                )
            )


            print(
                "rollout step max =",
                int(
                    self.rollout_steps.max().item()
                )
            )


        print(
            "==============================================="
        )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(
        self,
    ):

        return self.size


    # ========================================================
    # ROLLOUT ACCESS
    # ========================================================

    def get_rollout_ids(
        self,
    ):

        return self.rollout_ids


    def get_rollout_steps(
        self,
    ):

        return self.rollout_steps


    # ========================================================
    # GET RAW BATCH
    #
    # Useful for debugging / evaluation.
    # ========================================================

    def get_raw_batch(
        self,
        indices,
    ):

        return {

            "state":
            self.states[
                indices
            ],

            "action":
            self.actions[
                indices
            ],

            "next_state":
            self.next_states[
                indices
            ],
        }


    # ========================================================
    # GET PARAM2DEFORM TRAINING BATCH
    #
    # Paper Figure 4:
    #
    # actual state
    #   -> inverse rotate/translate by x,y,theta
    #   -> canonical state
    #
    # canonical state + spatial gradients
    #   -> state representation
    #
    # l,z
    #   -> shape parameters
    #
    # target:
    #
    # canonical delta
    #   = canonical(next_state)
    #   - canonical(state)
    #
    # ========================================================

    def get_batch(
        self,
        indices,
    ):

        # ----------------------------------------------------
        # RAW WORLD-FRAME DATA
        # ----------------------------------------------------

        state_actual = self.states[
            indices
        ]


        action = self.actions[
            indices
        ]


        next_actual = self.next_states[
            indices
        ]


        # ----------------------------------------------------
        # VISUAL SCULPTING 5D ACTION
        #
        # a = [x, y, theta, l, z]
        # ----------------------------------------------------

        x = action[
            :,
            0
        ]


        y = action[
            :,
            1
        ]


        theta = action[
            :,
            2
        ]


        length = action[
            :,
            3
        ]


        depth = action[
            :,
            4
        ]


        # ----------------------------------------------------
        # CANONICALIZE
        #
        # Same inverse transform must be used for:
        #
        #   S_t
        #   S_{t+1}
        #
        # so that the network predicts deformation in one
        # constant pose.
        # ----------------------------------------------------

        state_can = actual_to_canonical(
            state_actual,
            x,
            y,
            theta,
        )


        next_can = actual_to_canonical(
            next_actual,
            x,
            y,
            theta,
        )


        # ----------------------------------------------------
        # STATE REPRESENTATION
        #
        # Expected output:
        #
        #   [B,3,H,W]
        #
        # channels:
        #
        #   0 = current height/depth
        #   1 = spatial gradient dx
        #   2 = spatial gradient dy
        #
        # ----------------------------------------------------

        state_rep = make_state_representation(
            state_can
        )


        if state_rep.ndim != 4:

            raise RuntimeError(
                f"make_state_representation() must return "
                f"[B,C,H,W], got {tuple(state_rep.shape)}"
            )


        if state_rep.shape[
            1
        ] != 3:

            raise RuntimeError(
                f"Expected state representation with "
                f"3 channels [height,dx,dy], "
                f"got {tuple(state_rep.shape)}"
            )


        # ----------------------------------------------------
        # TARGET DEFORMATION
        #
        # param2deform predicts:
        #
        #   delta_can
        #
        # not the absolute next state.
        # ----------------------------------------------------

        delta_can = (
            next_can
            -
            state_can
        )


        # ----------------------------------------------------
        # SHAPE PARAMETERS
        #
        # Only:
        #
        #   l
        #   z
        #
        # enter param2deform.
        #
        # x,y,theta are pose parameters and have already been
        # consumed by the canonical transform.
        # ----------------------------------------------------

        params = normalize_shape_params(
            length,
            depth,
        )


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        result = {

            # canonical current state
            "state":
            state_can,

            # [height, grad_x, grad_y]
            "state_rep":
            state_rep,

            # canonical deformation target
            "delta":
            delta_can,

            # normalized [length, depth]
            "params":
            params,

            # full original 5D action
            "action":
            action,

            # raw world-frame states
            "state_actual":
            state_actual,

            "next_actual":
            next_actual,

            # canonical ground-truth next state
            "next_state":
            next_can,
        }


        # ----------------------------------------------------
        # OPTIONAL ROLLOUT METADATA
        # ----------------------------------------------------

        if self.has_rollout_metadata:

            result[
                "rollout_id"
            ] = self.rollout_ids[
                indices
            ]


            result[
                "rollout_step"
            ] = self.rollout_steps[
                indices
            ]


        return result