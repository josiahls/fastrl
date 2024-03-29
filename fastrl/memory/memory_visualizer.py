# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/04_Memory/01_memory_visualizer.ipynb.

# %% auto 0
__all__ = ['MemoryBufferViewer']

# %% ../../nbs/04_Memory/01_memory_visualizer.ipynb 2
# Python native modules
import io
from typing import List
# Third party libs
from PIL import Image
from ipywidgets import Button, HBox, VBox, Output, IntText, Label
from ipywidgets import widgets
from IPython.display import display
import numpy as np
import torch
# Local modules
from ..core import StepTypes

# %% ../../nbs/04_Memory/01_memory_visualizer.ipynb 4
class MemoryBufferViewer:
    def __init__(self, memory:List[StepTypes.types], agent=None, ignore_image:bool=False):
        # Assuming memory contains SimpleStep instances or None
        self.memory = memory
        self.agent = agent
        self.current_index = 0
        self.ignore_image = ignore_image
        # Add a label for displaying the number of elements in memory
        self.memory_size_label = Label(value=f"Number of Elements in Memory: {len([x for x in memory if x is not None])}")

        # Create the widgets
        self.out = Output()
        self.next_button = Button(description="Next")
        self.prev_button = Button(description="Previous")
        self.goto_text = IntText(value=0, description='Index:')
        # Button to jump to the desired index
        self.goto_button = Button(description="Go")
        self.goto_button.on_click(self.goto_step)
        self.action_value_label = Label()
        
        # Setup event handlers
        self.next_button.on_click(self.next_step)
        self.prev_button.on_click(self.prev_step)
        self.manual_navigation = False
        self.goto_text.observe(self.jump_to_index, names='value')

        # Display the widgets
        # Update the widget layout
        self.display_content_placeholder = VBox([])
        self.layout = VBox([
            self.memory_size_label,
            HBox([self.prev_button, self.next_button, self.goto_text, self.goto_button]),
            self.action_value_label,
            self.out,
            self.display_content_placeholder
        ])
        self.show_current()
        display(self.layout)
        
    def jump_to_index(self, change):
        if not self.manual_navigation:
            idx = change['new']
            if 0 <= idx < len(self.memory):
                self.current_index = idx
                self.show_current()
        else:
            self.manual_navigation = False

    def next_step(self, change):
        self._toggle_buttons_state(False)  # Disable buttons
        self.current_index = min(len(self.memory) - 1, self.current_index + 1)
        self.manual_navigation = True
        self.goto_text.value = self.current_index
        self.show_current()
        self._toggle_buttons_state(True)  # Enable buttons

    def prev_step(self, change):
        self._toggle_buttons_state(False)  # Disable buttons
        self.current_index = max(0, self.current_index - 1)
        self.manual_navigation = True
        self.goto_text.value = self.current_index
        self.show_current()
        self._toggle_buttons_state(True)  # Enable buttons

    def goto_step(self, change):
        self._toggle_buttons_state(False)  # Disable buttons
        target_idx = self.goto_text.value
        if 0 <= target_idx < len(self.memory):
            self.current_index = target_idx
        self.show_current()
        self._toggle_buttons_state(True)  # Enable buttons

    def _toggle_buttons_state(self, state):
        """Helper function to toggle button states."""
        self.prev_button.disabled = not state
        self.next_button.disabled = not state
        self.goto_button.disabled = not state

    def tensor_to_pil(self, tensor_image):
        """Convert a tensor to a PIL Image."""
        # Convert the tensor to numpy
        img_np = tensor_image.numpy()

        # Check if the tensor was in C, H, W format and convert it to H, W, C for PIL
        if img_np.ndim == 3 and img_np.shape[2] != 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Make sure the data type is right
        if img_np.dtype in (np.float32,np.float64):
            img_np = (img_np * 255).astype(np.uint8)

        return Image.fromarray(img_np)

    def pil_image_to_byte_array(self, pil_image):
        """Helper function to convert PIL image to byte array."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        return buffer.getvalue()

    def show_current(self):
        self.out.clear_output(wait=True)
        with self.out:
            if self.memory[self.current_index] is not None:
                step = self.memory[self.current_index]
                
                # Prepare the right-side content (step details)
                details_list = []
                details_list.append(Label(f"Action Value: {step.action.item()}"))
                # If agent is provided, predict the action based on step.state
                if self.agent is not None:
                    with torch.no_grad():
                        for predicted_action in self.agent([step]):pass
                        details_list.append(Label(f"Agent Predicted Action: {predicted_action}"))
                
                for field, value in step.to_tensordict().items():
                    if field not in ['state', 'next_state', 'image']:
                        details_list.append(Label(f"{field.capitalize()}: {value}"))
                
                details_display = VBox(details_list)

                # If the image is present, prepare left-side content
                if torch.is_tensor(step.image) and step.image.nelement() > 1 and not self.ignore_image:
                    pil_image = self.tensor_to_pil(step.image)
                    img_display = widgets.Image(value=self.pil_image_to_byte_array(pil_image), format='jpeg')
                    display_content = HBox([img_display, details_display])
                else:
                    # If image is not present, use the entire space for details
                    # You can expand this to include 'state' and 'next_state' as desired
                    # If image is not present, display 'state' and 'next_state' along with other details
                    state_label = Label(f"State: {step.state}")
                    next_state_label = Label(f"Next State: {step.next_state}")
                    display_content = VBox([details_display, state_label, next_state_label])
                
                self.display_content_placeholder.children = [display_content]
            else:
                print(f"Step {self.current_index}: Empty")
                self.action_value_label.value = ""
