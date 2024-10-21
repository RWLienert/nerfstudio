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

"""This file contains the render state machine, which is responsible for deciding when to render the image"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, get_args

import numpy as np
import cv2
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F
from viser import ClientHandle

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components.renderers import background_color_override_context
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils import colormaps, writer, plotly_utils
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.utils import CameraState, get_camera
from nerfstudio.viewer_legacy.server import viewer_utils

if TYPE_CHECKING:
    from nerfstudio.viewer.viewer import Viewer

RenderStates = Literal["low_move", "low_static", "high"]
RenderActions = Literal["rerender", "move", "static", "step"]


@dataclass
class RenderAction:
    """Message to the render state machine"""

    action: RenderActions
    """The action to take """
    camera_state: CameraState
    """The current camera state """


class RenderStateMachine(threading.Thread):
    """The render state machine is responsible for deciding how to render the image.
    It decides the resolution and whether to interrupt the current render.

    Args:
        viewer: the viewer state
    """

    def __init__(self, viewer: Viewer, viser_scale_ratio: float, client: ClientHandle):
        threading.Thread.__init__(self)
        self.transitions: Dict[RenderStates, Dict[RenderActions, RenderStates]] = {
            s: {} for s in get_args(RenderStates)
        }
        # by default, everything is a self-transition
        for a in get_args(RenderActions):
            for s in get_args(RenderStates):
                self.transitions[s][a] = s
        # then define the actions between states
        self.transitions["low_move"]["static"] = "low_static"
        self.transitions["low_static"]["static"] = "high"
        self.transitions["low_static"]["step"] = "high"
        self.transitions["low_static"]["move"] = "low_move"
        self.transitions["high"]["move"] = "low_move"
        self.transitions["high"]["rerender"] = "low_static"
        self.next_action: Optional[RenderAction] = None
        self.state: RenderStates = "low_static"
        self.render_trigger = threading.Event()
        self.target_fps = 30
        self.viewer = viewer
        self.interrupt_render_flag = False
        self.daemon = True
        self.output_keys = {}
        self.viser_scale_ratio = viser_scale_ratio
        self.client = client
        self.running = True
        self.past_angle = 0
        self.last_percentage = 0.0
        self.highest_error_percentage = 0.0
        self.error_checked = False
        self.current_position = ((0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)))
        self.error_values = []
        self.collecting_angle_error = False

    def action(self, action: RenderAction):
        """Takes an action and updates the state machine

        Args:
            action: the action to take
        """

        if self.next_action is None:
            self.next_action = action
        elif action.action == "step" and (self.state == "low_move" or self.next_action.action in ("move", "rerender")):
            # ignore steps if:
            #  1. we are in low_moving state
            #  2. the current next_action is move, static, or rerender
            return
        elif self.next_action.action == "rerender":
            # never overwrite rerenders
            pass
        elif action.action == "static" and self.next_action.action == "move":
            # don't overwrite a move action with a static: static is always self-fired
            return
        else:
            #  monimal use case, just set the next action
            self.next_action = action

        # handle interrupt logic
        if self.state == "high" and self.next_action.action in ("move", "rerender"):
            self.interrupt_render_flag = True
        self.render_trigger.set()

    def _render_img(self, camera_state: CameraState):
        """Takes the current camera, generates rays, and renders the image

        Args:
            camera_state: the current camera state
        """
        
        # initialize the camera ray bundle
        if self.viewer.control_panel.crop_viewport:
            obb = self.viewer.control_panel.crop_obb
        else:
            obb = None

        image_height, image_width = self._calculate_image_res(camera_state.aspect)

        # These 2 lines make the control panel's time option independent from the render panel's.
        # When outside of render preview, it will use the control panel's time.
        if not self.viewer.render_tab_state.preview_render and self.viewer.include_time:
            camera_state.time = self.viewer.control_panel.time
        camera = get_camera(camera_state, image_height, image_width)
        camera = camera.to(self.viewer.get_model().device)
        assert isinstance(camera, Cameras)
        assert camera is not None, "render called before viewer connected"

        with TimeWriter(None, None, write=False) as vis_t:
            with self.viewer.train_lock if self.viewer.train_lock is not None else contextlib.nullcontext():
                if isinstance(self.viewer.get_model(), SplatfactoModel):
                    color = self.viewer.control_panel.background_color
                    background_color = torch.tensor(
                        [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0],
                        device=self.viewer.get_model().device,
                    )
                    self.viewer.get_model().set_background(background_color)
                self.viewer.get_model().eval()
                step = self.viewer.step
                try:
                    if self.viewer.control_panel.crop_viewport:
                        color = self.viewer.control_panel.background_color
                        if color is None:
                            background_color = torch.tensor([0.0, 0.0, 0.0], device=self.viewer.pipeline.model.device)
                        else:
                            background_color = torch.tensor(
                                [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0],
                                device=self.viewer.get_model().device,
                            )
                        with background_color_override_context(
                            background_color
                        ), torch.no_grad(), viewer_utils.SetTrace(self.check_interrupt):
                            outputs1 = self.viewer.get_model().get_outputs_for_camera(camera, obb_box=obb)
                        if self.viewer.num_pipelines == 2:
                            with background_color_override_context(
                                background_color
                            ), torch.no_grad(), viewer_utils.SetTrace(self.check_interrupt):
                                outputs2 = self.viewer.get_model2().get_outputs_for_camera(camera, obb_box=obb)
                    else:
                        with torch.no_grad(), viewer_utils.SetTrace(self.check_interrupt):
                            outputs1 = self.viewer.get_model().get_outputs_for_camera(camera, obb_box=obb)
                        if self.viewer.num_pipelines == 2:
                            with torch.no_grad(), viewer_utils.SetTrace(self.check_interrupt):
                                outputs2 = self.viewer.get_model2().get_outputs_for_camera(camera, obb_box=obb)
                except viewer_utils.IOChangeException:
                    self.viewer.get_model().train()
                    raise
                self.viewer.get_model().train()

            if self.viewer.control_panel.num_pipelines == 2:
                # Initialises the rendering and calculates the error values from different viewpoints
                if self.viewer.control_panel._angle_based_error.value == True and self.error_checked == False:
                    self.error_checked = True
                    self.current_position = self.viewer.control_panel.return_location()
                    self.collecting_angle_error = True
                    new_coordinates = self.return_camera_coordinates()
                    
                    self._render_img(self.viewer.get_camera_state(self.client))
                    
                    for coordinate in new_coordinates:
                        self.viewer.control_panel.change_location(
                            (coordinate[0], coordinate[1], coordinate[2], coordinate[3])
                        )
                        
                        self._render_img(self.viewer.get_camera_state(self.client))
                    
                    self.collecting_angle_error = False
                    self.viewer.control_panel.change_location(
                        (self.current_position[0], self.current_position[1], self.current_position[2], self.current_position[3])
                    )
                
                # Reset values when the angle-based error is unchecked
                if self.viewer.control_panel._angle_based_error.value == False and self.error_checked == True:
                    self.error_checked = False
                    self.current_position = ((0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)))
                    self.error_values = []
                
                # Preview particular viewpoints
                if self.past_angle != self.viewer.control_panel._viewpoint_slider.value:
                    self.past_angle = self.viewer.control_panel._viewpoint_slider.value
                    self.simulate_cameras(self.viewer.control_panel._viewpoint_slider.value)
                    
                # Compute the difference between the two outputs and calculate percentage of pixels with significant colour differences
                rgb_diff = outputs1["rgb"] - outputs2["rgb"]
        
                threshold = self.viewer.control_panel.error_threshold
                error_mask = torch.abs(rgb_diff) > threshold

                total_pixels = error_mask.numel()
                significant_pixels = torch.sum(error_mask).item()
                colour_percentage = round((significant_pixels / total_pixels) * 100, 2)
                
                # Logic for updating errors
                if self.collecting_angle_error == True:
                    self.error_values.append(colour_percentage)
                    
                if colour_percentage != self.last_percentage:
                    self.viewer.update_percentage(colour_percentage)
                    self.last_percentage = colour_percentage
                    
                # Code for finding the highest error viewpoint
                if self.viewer.control_panel.angle_error_button_clicked == True:
                    self.viewer.control_panel.angle_error_button_clicked = False
                    self.viewer.control_panel._coloured_error_visual.value = False
                    max_error = 0
                    
                    for i in range(0,len(self.error_values)):
                        if self.error_values[i] > max_error:
                            max_error = self.error_values[i]
                            specific_angle = i
                    self.highest_error_percentage = self.error_values[specific_angle]
                    self.simulate_cameras(specific_angle)
                                
                # Settings for adjusting colour rendering
                if self.viewer.control_panel._visualise_error.value == True:
                    emphasis_value = self.viewer.control_panel.error_emphasis
                    emphasis_scaling = emphasis_value / 10.0
                    
                    error_colour = self.viewer.control_panel.error_colour                    
                    error_overlay = torch.zeros_like(rgb_diff)
                    error_overlay[..., 0] = error_colour[0]
                    error_overlay[..., 1] = error_colour[1]
                    error_overlay[..., 2] = error_colour[2]

                    emphasized_diff = rgb_diff * emphasis_scaling
                    blended_diff = (1 - error_mask.float()) * outputs1["rgb"] + error_mask.float() * error_overlay * emphasis_scaling

                    outputs1["rgb"] = blended_diff

            num_rays = (camera.height * camera.width).item()
            if self.viewer.control_panel.layer_depth:
                if isinstance(self.viewer.get_model(), SplatfactoModel):
                    assert len(outputs1["depth"].shape) == 3
                    assert outputs1["depth"].shape[-1] == 1

                    desired_depth_pixels = {"low_move": 128, "low_static": 128, "high": 512}[self.state] ** 2
                    current_depth_pixels = outputs1["depth"].shape[0] * outputs1["depth"].shape[1]

                    scale = min(desired_depth_pixels / max(1, current_depth_pixels), 1.0)

                    outputs1["gl_z_buf_depth"] = F.interpolate(
                        outputs1["depth"].squeeze(dim=-1)[None, None, ...],
                        size=(int(outputs1["depth"].shape[0] * scale), int(outputs1["depth"].shape[1] * scale)),
                        mode="bilinear",
                    )[0, 0, :, :, None]
                else:
                    R = camera.camera_to_worlds[0, 0:3, 0:3].T
                    camera_ray_bundle = camera.generate_rays(camera_indices=0, obb_box=obb)
                    pts = camera_ray_bundle.directions * outputs1["depth"]
                    pts = (R @ (pts.view(-1, 3).T)).T.view(*camera_ray_bundle.directions.shape)
                    outputs1["gl_z_buf_depth"] = -pts[..., 2:3]  # negative z axis is the coordinate convention

        render_time = vis_t.duration
        if writer.is_initialized() and render_time != 0:
            writer.put_time(
                name=EventName.VIS_RAYS_PER_SEC, duration=num_rays / render_time, step=step, avg_over_steps=True
            )
        return outputs1
    
    def return_camera_coordinates(self, num_simulations: int = 15):
        new_coordinates = []
        # Get the current camera position and rotation
        origin_x, origin_y, origin_z, quaternion = self.current_position
        original_position = torch.tensor([origin_x, origin_y, origin_z])
        radius = self.viewer.control_panel._radius_value.value  

        # Normalize the quaternion and extract values
        quaternion = quaternion / np.linalg.norm(quaternion)
        w, x, y, z = quaternion

        # Calculate the forward vector directly from the quaternion
        forward_vector = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x**2 + y**2)
        ])

        # Normalize the forward vector
        unit_forward_vector = forward_vector / np.linalg.norm(forward_vector)

        arbitrary_vector = np.array([0, 0, 1])  # Global up vector
        right_vector = np.cross(unit_forward_vector, arbitrary_vector)

        # If the forward vector is aligned with the up vector, switch to a different vector
        if np.linalg.norm(right_vector) == 0:
            arbitrary_vector = np.array([1, 0, 0])
            right_vector = np.cross(unit_forward_vector, arbitrary_vector)

        right_vector /= np.linalg.norm(right_vector)  # Normalize the right vector

        # The up vector can be found by crossing right and forward vectors
        up_vector = np.cross(unit_forward_vector, right_vector)
        
        # Adjust the circle's center to be _origin_value away from the current position along the forward vector
        origin_offset = self.viewer.control_panel._origin_value.value
        circle_center = original_position + unit_forward_vector * origin_offset

        # Generate points in a circle perpendicular to the forward vector
        angles = np.linspace(0, 2 * np.pi, num_simulations, endpoint=False)
        
        for angle in angles:
            # Calculate the point on the circle in the plane
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)
            point_on_circle = circle_center + np.array([x_offset, y_offset, 0])

            # Calculate the target point where the camera should look
            target_point = original_position + unit_forward_vector * origin_offset * self.viewer.control_panel._angle_origin.value

            forward_vector = target_point - point_on_circle
            forward_vector /= np.linalg.norm(forward_vector)
            up_vector = np.array([0, 0, -1])
            right_vector = np.cross(up_vector, forward_vector)
            right_vector /= np.linalg.norm(right_vector)
            true_up_vector = np.cross(forward_vector, right_vector)

            # Create a rotation matrix using the right, true_up, and forward vectors
            rotation_matrix = np.vstack([right_vector, true_up_vector, forward_vector]).T
            quaternion = R.from_matrix(rotation_matrix).as_quat()
            x, y, z, w = quaternion
            new_quaternion = [w, x, y, z]
            
            new_coordinates.append((point_on_circle[0], point_on_circle[1], point_on_circle[2], new_quaternion))
            
        return new_coordinates

    def simulate_cameras(self, specific_angle, num_simulations: int = 15):
        """
        Simulate cameras moving in a circular path on a plane that is perpendicular 
        to the camera's viewing direction (forward vector), while keeping the camera
        pointed at a target point.

        Args:
            num_simulations: Number of simulated camera positions to create.
            radius: Distance to move the cameras from the original position (radius of the circle).
        """

        # Get the current camera position and rotation
        origin_x, origin_y, origin_z, quaternion = self.current_position
        original_position = torch.tensor([origin_x, origin_y, origin_z])
        radius = self.viewer.control_panel._radius_value.value  

        # Normalize the quaternion and extract values
        quaternion = quaternion / np.linalg.norm(quaternion)
        w, x, y, z = quaternion

        # Calculate the forward vector directly from the quaternion
        forward_vector = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x**2 + y**2)
        ])

        # Normalize the forward vector
        unit_forward_vector = forward_vector / np.linalg.norm(forward_vector)

        arbitrary_vector = np.array([0, 0, 1])  # Global up vector
        right_vector = np.cross(unit_forward_vector, arbitrary_vector)

        # If the forward vector is aligned with the up vector, switch to a different vector
        if np.linalg.norm(right_vector) == 0:
            arbitrary_vector = np.array([1, 0, 0])
            right_vector = np.cross(unit_forward_vector, arbitrary_vector)

        right_vector /= np.linalg.norm(right_vector)

        # The up vector can be found by crossing right and forward vectors
        up_vector = np.cross(unit_forward_vector, right_vector)
        
        # Adjust the circle's center to be _origin_value away from the current position along the forward vector
        origin_offset = self.viewer.control_panel._origin_value.value
        circle_center = original_position + unit_forward_vector * origin_offset

        # Generate points in a circle perpendicular to the forward vector
        angles = np.linspace(0, 2 * np.pi, num_simulations, endpoint=False)
        
        if specific_angle is not None:
            if specific_angle == 0:
                self.viewer.control_panel.change_location(
                    (self.current_position[0], self.current_position[1], self.current_position[2], self.current_position[3])
                )
                
                # Render the image for the new camera position
                self._render_img(self.viewer.get_camera_state(self.client))
            else:
                specific_angle -= 1
                x_offset = radius * np.cos(angles[specific_angle])
                y_offset = radius * np.sin(angles[specific_angle])
                point_on_circle = circle_center + np.array([x_offset, y_offset, 0])

                # Calculate the target point where the camera should look
                target_point = original_position + unit_forward_vector * origin_offset * self.viewer.control_panel._angle_origin.value
                forward_vector = target_point - point_on_circle
                forward_vector /= np.linalg.norm(forward_vector)
                up_vector = np.array([0, 0, -1])
                right_vector = np.cross(up_vector, forward_vector)
                right_vector /= np.linalg.norm(right_vector)
                true_up_vector = np.cross(forward_vector, right_vector)

                # Create a rotation matrix using the right, true_up, and forward vectors
                rotation_matrix = np.vstack([right_vector, true_up_vector, forward_vector]).T
                quaternion = R.from_matrix(rotation_matrix).as_quat()
                x, y, z, w = quaternion
                new_quaternion = [w, x, y, z]
                                    
                self.viewer.control_panel.change_location(
                    (point_on_circle[0], point_on_circle[1], point_on_circle[2], new_quaternion)
                )
                
                # Render the image for the new camera position
                self._render_img(self.viewer.get_camera_state(self.client))


    def run(self):
        """Main loop for the render thread"""
        while self.running:
            if not self.viewer.ready:
                time.sleep(0.1)
                continue
            if not self.render_trigger.wait(0.2):
                # if we haven't received a trigger in a while, send a static action
                self.action(RenderAction(action="static", camera_state=self.viewer.get_camera_state(self.client)))
            action = self.next_action
            self.render_trigger.clear()
            if action is None:
                continue
            self.next_action = None
            if self.state == "high" and action.action == "static":
                # if we are in high res and we get a static action, we don't need to do anything
                continue
            self.state = self.transitions[self.state][action.action]
            try:
                outputs = self._render_img(action.camera_state)
            except viewer_utils.IOChangeException:
                # if we got interrupted, don't send the output to the viewer
                continue
            self._send_output_to_viewer(outputs, static_render=(action.action in ["static", "step"]))

    def check_interrupt(self, frame, event, arg):
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.interrupt_render_flag:
                self.interrupt_render_flag = False
                raise viewer_utils.IOChangeException
        return self.check_interrupt

    def _send_output_to_viewer(self, outputs: Dict[str, Any], static_render: bool = True):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the model
        """
        output_keys = set(outputs.keys())
        if self.output_keys != output_keys:
            self.output_keys = output_keys
            self.viewer.control_panel.update_output_options(list(outputs.keys()))

        output_render = self.viewer.control_panel.output_render
        self.viewer.update_colormap_options(
            dimensions=outputs[output_render].shape[-1], dtype=outputs[output_render].dtype
        )
        selected_output = colormaps.apply_colormap(
            image=outputs[self.viewer.control_panel.output_render],
            colormap_options=self.viewer.control_panel.colormap_options,
        )

        if self.viewer.control_panel.split:
            split_output_render = self.viewer.control_panel.split_output_render
            self.viewer.update_split_colormap_options(
                dimensions=outputs[split_output_render].shape[-1], dtype=outputs[split_output_render].dtype
            )
            split_output = colormaps.apply_colormap(
                image=outputs[self.viewer.control_panel.split_output_render],
                colormap_options=self.viewer.control_panel.split_colormap_options,
            )
            split_index = min(
                int(self.viewer.control_panel.split_percentage * selected_output.shape[1]),
                selected_output.shape[1] - 1,
            )
            selected_output = torch.cat([selected_output[:, :split_index], split_output[:, split_index:]], dim=1)
            selected_output[:, split_index] = torch.tensor([0.133, 0.157, 0.192], device=selected_output.device)

        selected_output = (selected_output * 255).type(torch.uint8)
        depth = (
            outputs["gl_z_buf_depth"].cpu().numpy() * self.viser_scale_ratio if "gl_z_buf_depth" in outputs else None
        )

        # Convert to numpy.
        selected_output = selected_output.cpu().numpy()
        assert selected_output.shape[-1] == 3
        
        if self.viewer.control_panel._coloured_error_visual.value == True and self.viewer.control_panel._angle_based_error.value == True:
            # Draw a circle on the selected_output image
            height, width, _ = selected_output.shape
            centre = (width // 2, height // 2)
            radius = min(width, height) // 10
            thickness = 5
            
            values = self.error_values[1:]
            max_value = max(values)
            min_value = min(values)
            
            # Normalize values to the range [0, 1]
            norm_values = [(value - min_value) / (max_value - min_value) for value in values]

            # Initialize the start angle for drawing pie segments
            angle_step = (2 * np.pi) / len(values)
            start_angle = 0

            # Loop over all values and draw pie segments
            for i in range(len(norm_values)):
                value = norm_values[i]
                next_value = norm_values[(i + 1) % len(norm_values)]

                def get_colour(value):
                    # Ensure the value is clamped between 0 and 1
                    value = max(0, min(1, value))

                    if value <= 0.33:
                        # Map to green
                        r = int(value * 3 * 255)
                        g = 255
                        b = 0
                    elif value <= 0.66:
                        # Map to orange
                        r = int((value - 0.33) * 3 * 255)
                        g = 165
                        b = 0
                    else:
                        # Map to red
                        r = 255
                        g = int((1 - value) * 3 * 255)
                        b = 0

                    return (r, g, b)

                colour = get_colour(value)
                next_colour = get_colour(next_value)

                # Break the segment into smaller sub-segments for blending
                sub_angle_step = angle_step / 5
                for j in range(5):
                    end_angle = start_angle - sub_angle_step

                    # Compute the alpha (blending) value
                    alpha = j / 5.0

                    # Blend the current color with the next color
                    blended_colour = (1 - alpha) * np.array(colour) + alpha * np.array(next_colour)
                    blended_colour = blended_colour.astype(int).tolist()

                    # Draw the blended pie segment
                    cv2.ellipse(
                        selected_output,
                        centre,
                        (radius, radius),
                        0,
                        np.degrees(start_angle),
                        np.degrees(end_angle),
                        blended_colour,
                        thickness
                    )

                    # Move to the next sub-segment
                    start_angle = end_angle

                # Move to the next segment
                start_angle -= sub_angle_step

        # Pad image if the aspect ratio (W/H) doesn't match the client!
        current_h, current_w = selected_output.shape[:2]
        desired_aspect = self.client.camera.aspect
        pad_width = int(max(0, (desired_aspect * current_h - current_w) // 2))
        pad_height = int(max(0, (current_w / desired_aspect - current_h) // 2))
        if pad_width > 5 or pad_height > 5:
            selected_output = np.pad(
                selected_output,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        jpg_quality = (
            self.viewer.config.jpeg_quality
            if static_render
            else 75
            if self.viewer.render_tab_state.preview_render
            else 40
        )
        self.client.scene.set_background_image(
            selected_output,
            format=self.viewer.config.image_format,
            jpeg_quality=jpg_quality,
            depth=depth,
        )
        res = f"{selected_output.shape[1]}x{selected_output.shape[0]}px"
        self.viewer.stats_markdown.content = self.viewer.make_stats_markdown(None, res)

    def _calculate_image_res(self, aspect_ratio: float) -> Tuple[int, int]:
        """Calculate the maximum image height that can be rendered in the time budget

        Args:
            apect_ratio: the aspect ratio of the current view
        Returns:
            image_height: the maximum image height that can be rendered in the time budget
            image_width: the maximum image width that can be rendered in the time budget
        """
        max_res = self.viewer.control_panel.max_res
        if self.state == "high":
            # high res is always static
            image_height = max_res
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)
        elif self.state in ("low_move", "low_static"):
            if writer.is_initialized() and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                vis_rays_per_sec = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
            else:
                vis_rays_per_sec = 100000
            target_fps = self.target_fps
            num_vis_rays = vis_rays_per_sec / target_fps
            image_height = (num_vis_rays / aspect_ratio) ** 0.5
            image_height = int(round(image_height, -1))
            image_height = max(min(max_res, image_height), 30)
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)
        else:
            raise ValueError(f"Invalid state: {self.state}")

        return image_height, image_width
