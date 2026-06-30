from typing import Literal

from mjlab.managers.curriculum_manager import CurriculumTermCfg


def make_curriculum_cfg(terrain_type: Literal["flat"] = "flat") -> dict[str, CurriculumTermCfg]:
    if terrain_type != "flat":
        raise ValueError("k1_kick curriculum is flat-only")
    return {}
