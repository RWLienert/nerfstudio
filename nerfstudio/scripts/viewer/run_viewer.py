# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
Starts viewer in eval mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from threading import Lock
from typing import Literal, Optional, List

import tyro

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState


@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewer:
    """Load a checkpoint and start the viewer."""

    load_config: List[Path]
    
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""
    vis: Literal["viewer", "viewer_legacy"] = "viewer"
    """Type of viewer"""

    def main(self) -> None:
        """Main function."""
        if len(self.load_config) not in [1, 2]:
            raise ValueError("You must provide 1 or 2 config paths")       
        
        config1, pipeline1, _, step1 = eval_setup(
            self.load_config[0],
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        
        num_rays_per_chunk = config1.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config1.vis = self.vis
        config1.viewer = self.viewer.as_viewer_config()
        config1.viewer.num_rays_per_chunk = num_rays_per_chunk
        
        if len(self.load_config) == 2:
            config2, pipeline2, _, step2 = eval_setup(
                self.load_config[1],
                eval_num_rays_per_chunk=None,
                test_mode="test",
            )
            config2.vis = self.vis
            config2.viewer = self.viewer.as_viewer_config()
            config2.viewer.num_rays_per_chunk = num_rays_per_chunk

            _start_viewer(config1, pipeline1, step1, config2, pipeline2, step2)
        else:
            _start_viewer(config1, pipeline1, step1)

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """


def _start_viewer(config1: TrainerConfig, pipeline1: Pipeline, step1: int, config2: Optional[TrainerConfig] = None, pipeline2: Optional[Pipeline] = None, step2: Optional[int] = None):
    """Starts the viewer

    Args:
        config: Configuration of pipeline to load
        pipeline: Pipeline instance of which to load weights
        step: Step at which the pipeline was saved
    """
    base_dir = config1.get_base_dir()
    viewer_log_path = base_dir / config1.viewer.relative_log_filename
    banner_messages = None
    viewer_state = None
    viewer_callback_lock = Lock()
    if config1.vis == "viewer_legacy":
        viewer_state = ViewerLegacyState(
            config1.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline1.datamanager.get_datapath(),
            pipeline=pipeline1,
            train_lock=viewer_callback_lock,
        )
        banner_messages = [f"Legacy viewer at: {viewer_state.viewer_url}"]
    if config1.vis == "viewer":
        viewer_state = ViewerState(
            config1.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline1.datamanager.get_datapath(),
            datapath2=pipeline2.datamanager.get_datapath() if pipeline2 else None,
            pipeline=pipeline1,
            pipeline2=pipeline2 if pipeline2 else None,
            share=config1.viewer.make_share_url,
            train_lock=viewer_callback_lock,
        )
        banner_messages = viewer_state.viewer_info

    # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
    config1.logging.local_writer.enable = False
    writer.setup_local_writer(config1.logging, max_iter=config1.max_num_iterations, banner_messages=banner_messages)

    assert viewer_state and pipeline1.datamanager.train_dataset
    viewer_state.init_scene(
        train_dataset=pipeline1.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline1.datamanager.eval_dataset,
    )
    if isinstance(viewer_state, ViewerLegacyState):
        viewer_state.viser_server.set_training_state("completed")
    viewer_state.update_scene(step=step1)
    while True:
        time.sleep(0.01)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RunViewer]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(tyro.conf.FlagConversionOff[RunViewer])  # noqa
