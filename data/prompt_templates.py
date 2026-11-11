"""Single source of truth for the 64 fixed RemoteCLIP text templates."""

REMOTECLIP_PROMPT_TEMPLATES = [
    # Basic remote-sensing viewpoints.
    "a remote sensing image of {}.",
    "a satellite image of {}.",
    "an aerial image of {}.",
    "an overhead image of {}.",
    "a top-down view of {}.",
    "a bird's-eye view of {}.",
    "remote sensing imagery showing {}.",
    "an overhead scene containing {}.",

    # Natural caption variants.
    "the remote sensing image shows {}.",
    "the satellite image shows {}.",
    "the aerial image shows {}.",
    "an overhead image in which {} can be seen.",
    "a satellite image in which {} is visible.",
    "an aerial scene with {}.",
    "a geographic scene containing {}.",
    "a view from above showing {}.",

    # Resolution and acquisition platforms.
    "a high-resolution remote sensing image of {}.",
    "a high-resolution satellite image of {}.",
    "a high-resolution aerial image of {}.",
    "a very high-resolution overhead image of {}.",
    "a drone image of {}.",
    "a UAV image of {}.",
    "an orthophoto showing {}.",
    "a wide-area satellite view containing {}.",

    # Region and spatial extent.
    "a close overhead view of {}.",
    "a local overhead view containing {}.",
    "a visible region of {} from above.",
    "the spatial extent of {} in an overhead image.",
    "the visible footprint of {} in an aerial image.",
    "a land-cover area of {} in satellite imagery.",
    "a land-use area of {} in aerial imagery.",
    "a remote sensing region corresponding to {}.",

    # Shape and boundary.
    "the shape of {} in an overhead image.",
    "the outline of {} in satellite imagery.",
    "the boundary of {} in aerial imagery.",
    "the edge pattern of {} in an overhead image.",
    "the geometric structure of {} from above.",
    "the spatial layout of {} in remote sensing imagery.",
    "the visible form of {} in a satellite image.",
    "the surface structure of {} in an aerial image.",

    # Texture and appearance.
    "the texture of {} in satellite imagery.",
    "the fine-grained texture of {} in aerial imagery.",
    "the visual pattern of {} in an overhead image.",
    "the spatial pattern of {} in remote sensing imagery.",
    "the color pattern of {} in an aerial image.",
    "a homogeneous region of {} from above.",
    "a heterogeneous region containing {} from above.",
    "an overhead view where {} is distinguished from its surroundings.",

    # Scale, count, and distribution.
    "a small instance of {} in an aerial image.",
    "a large instance of {} in a satellite image.",
    "multiple instances of {} in overhead imagery.",
    "a group of {} visible from above.",
    "clustered {} in remote sensing imagery.",
    "scattered {} in remote sensing imagery.",
    "densely distributed {} in an overhead image.",
    "sparsely distributed {} in an overhead image.",

    # Spatial morphology and scene context.
    "a compact region of {} in remote sensing imagery.",
    "an elongated region of {} in remote sensing imagery.",
    "a linear structure of {} in an overhead image.",
    "a continuous region of {} viewed from above.",
    "a fragmented region of {} viewed from above.",
    "urban remote sensing imagery containing {}.",
    "rural remote sensing imagery containing {}.",
    "{} within its surrounding landscape in satellite imagery.",
]

assert len(REMOTECLIP_PROMPT_TEMPLATES) == 64
