BackseatDriver: Semantic Segmentation for Safety Monitoring
===========================================================

*This project represents our submission for the Spring 2020 16.412 Grand Challenge.*

A pressing problem in the design of autonomous driving architectures is the need to constantly monitor the safety of planned trajectories. In this project, we propose using semantic segmentation (using RefineNet) with RGBd data to construct 3D point clouds with semantic labels of the vehicle's environment, against which planned trajectories can be checked for collision. Of course, the safety of autonomous systems is a complex topic that cannot be solved by a single innovation, but we present this approach as a possible tool in a broader safety-engineering toolbox.

Dependencies and Installation
-----------------------------

This project is intended for use within the [Carla driving simulator](https://carla.org/). Please follow the instructions [here](https://carla.readthedocs.io/en/latest/) to install Carla.

Additionally, our project requires the following dependencies:
    - Numpy
    - PIL
    - PyTorch
