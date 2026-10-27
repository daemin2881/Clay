import torch
import torch.nn as nn
import torch.nn.functional as F

from vs_config import (
    MAP_H,
    MAP_W,
)


class ConvAct(
    nn.Module
):

    def __init__(
        self,
        cin,
        cout,
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            cin,
            cout,
            3,
            padding=1,
        )

        self.act = nn.SiLU()


    def forward(
        self,
        x,
    ):

        return self.act(
            self.conv(
                x
            )
        )


class Param2Deform(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        # ----------------------------------------------------
        # l,z
        # 3 FC
        # ----------------------------------------------------

        self.param_fc = nn.Sequential(

            nn.Linear(
                2,
                128,
            ),

            nn.SiLU(),

            nn.Linear(
                128,
                256,
            ),

            nn.SiLU(),

            nn.Linear(
                256,
                32
                *
                16
                *
                16,
            ),

            nn.SiLU(),
        )


        # ----------------------------------------------------
        # parameter conv x3
        # ----------------------------------------------------

        self.param_conv = nn.Sequential(

            ConvAct(
                32,
                32,
            ),

            ConvAct(
                32,
                32,
            ),

            ConvAct(
                32,
                32,
            ),
        )


        # ----------------------------------------------------
        # state + gradient
        # ----------------------------------------------------

        self.state_encoder = nn.Sequential(

            ConvAct(
                3,
                32,
            ),

            ConvAct(
                32,
                32,
            ),

            ConvAct(
                32,
                32,
            ),
        )


        # ----------------------------------------------------
        # current feature + raw current + params
        #
        # 32 + 1 + 32 = 65
        #
        # 5 conv
        # ----------------------------------------------------

        self.final = nn.Sequential(

            ConvAct(
                65,
                64,
            ),

            ConvAct(
                64,
                64,
            ),

            ConvAct(
                64,
                64,
            ),

            ConvAct(
                64,
                32,
            ),

            nn.Conv2d(
                32,
                1,
                3,
                padding=1,
            ),
        )


    def forward(
        self,
        state_rep,
        shape_params,
    ):

        B = state_rep.shape[
            0
        ]

        p = self.param_fc(
            shape_params
        )

        p = p.reshape(
            B,
            32,
            16,
            16,
        )

        p = F.interpolate(
            p,
            size=(
                MAP_H,
                MAP_W,
            ),
            mode="bilinear",
            align_corners=False,
        )

        p = self.param_conv(
            p
        )

        s = self.state_encoder(
            state_rep
        )

        raw_height = state_rep[
            :,
            0:1
        ]

        x = torch.cat(
            [
                s,
                raw_height,
                p,
            ],
            dim=1,
        )

        return self.final(
            x
        )