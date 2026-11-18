"""From drives to a network that drives, one module per stage.

    collect    the teacher drives in the simulator, with disturbed steering -> episodes/
    label      a plan for every frame: the teacher's, or the actions that followed
    validate   is the dataset complete, consistent and varied enough?
    browse     look at the frames with their labels drawn on the road
    train      fit the network -> models/<version>.pt
    lrfind     which learning rates make the loss fall, before a long training
    evaluate   drive the models on every track -> reports/
    probe      look inside a trained CNN: layers, curves, errors, activations, saliency, behaviour
    status     what the project holds, and the next step
    export     the episodes as a LeRobotDataset, to fine-tune a VLA (needs lerobot)

A stage is a module with `add_parser(subparsers)` and `run(project, args)`; adding one
is writing that module and listing it here. Files are laid out by `project.py`.
"""
from . import browse, collect, evaluate, export, label, lrfind, probe, status, train, validate

STAGES = [collect, label, validate, browse, lrfind, train, evaluate, probe, status, export]
