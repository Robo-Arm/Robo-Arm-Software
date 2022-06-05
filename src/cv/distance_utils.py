def get_focal_length(perceived_width: float, known_distance: float, known_width: float)->float:
    # F = (P*D) / W
    return (perceived_width * known_distance) / known_width

def get_distance_to_camera(known_width: float, focal_length: float,  perceived_width: float)->float:
    # Let perceived_width be the camera's measurement of the object in pixels
    return (known_width * focal_length) / perceived_width

