import math

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset


def shrink_bbox(bbox, coef=0.3, inplace=False):
    """
    Shrinks a bounding box polygon by a given coefficient.

    This function reduces the size of a quadrilateral bounding box by moving each edge inward,
    effectively shrinking the box. This is useful for creating tighter regions around detected text.

    Args:
        bbox (np.ndarray): A 4x2 array representing the four vertices of the bounding box.
        coef (float, optional): Shrink coefficient determining how much to reduce the box size.
                                 Default is 0.3.
        inplace (bool, optional): If True, modifies the input bbox array directly.
                                  If False, returns a new shrunk bbox. Default is False.

    Returns:
        np.ndarray: The shrunk bounding box as a 4x2 array.
    """
    # Calculate the length of each side of the bounding box
    lens = [np.linalg.norm(bbox[i] - bbox[(i + 1) % 4], ord=2) for i in range(4)]
    # Determine the minimum adjacent lengths for each vertex
    r = [min(lens[(i - 1) % 4], lens[i]) for i in range(4)]

    # If not modifying in place, create a copy of the bbox
    if not inplace:
        bbox = bbox.copy()

    # Determine offset based on the sum of opposite sides to handle skewed boxes
    offset = 0 if lens[0] + lens[2] > lens[1] + lens[3] else 1
    # Iterate over the vertices in a specific order to shrink the box
    for idx in [0, 2, 1, 3]:
        p1_idx, p2_idx = (idx + offset) % 4, (idx + 1 + offset) % 4
        p1p2 = bbox[p2_idx] - bbox[p1_idx]
        dist = np.linalg.norm(p1p2)
        if dist <= 1:
            continue  # Skip if the distance is too small to avoid division by zero
        # Move the vertices inward based on the shrink coefficient and distance
        bbox[p1_idx] += p1p2 / dist * r[p1_idx] * coef
        bbox[p2_idx] -= p1p2 / dist * r[p2_idx] * coef
    return bbox


def get_rotated_coords(h, w, theta, anchor):
    """
    Computes rotated coordinates for each pixel in an image.

    This function generates the rotated x and y coordinates for each pixel in an image
    given a rotation angle and an anchor point. It is useful for applying geometric transformations.

    Args:
        h (int): Height of the image.
        w (int): Width of the image.
        theta (float): Rotation angle in radians.
        anchor (np.ndarray): A 2-element array representing the anchor point (x, y) for rotation.

    Returns:
        tuple: Two 2D arrays representing the rotated x and y coordinates.
    """
    anchor = anchor.reshape(2, 1)  # Ensure anchor is a column vector
    rotate_mat = get_rotate_mat(theta)  # Get the rotation matrix
    # Create meshgrid for x and y coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_lin = x.reshape((1, x.size))  # Flatten the x coordinates
    y_lin = y.reshape((1, y.size))  # Flatten the y coordinates
    # Combine x and y into a coordinate matrix
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    # Apply rotation to the coordinates
    rotated_coord = np.dot(rotate_mat, coord_mat - anchor) + anchor
    # Reshape back to image dimensions
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def get_rotate_mat(theta):
    """
    Generates a 2D rotation matrix for a given angle.

    Args:
        theta (float): Rotation angle in radians.

    Returns:
        np.ndarray: A 2x2 rotation matrix.
    """
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])


def calc_error_from_rect(bbox):
    """
    Calculates the cumulative error between a rotated bounding box and an axis-aligned rectangle.

    The default orientation is defined as:
        - x1y1: left-top
        - x2y2: right-top
        - x3y3: right-bottom
        - x4y4: left-bottom

    Args:
        bbox (np.ndarray): A 4x2 array representing the four vertices of the bounding box.

    Returns:
        float: The sum of Euclidean distances between corresponding vertices of the rotated bbox and the default rectangle.
    """
    x_min, y_min = np.min(bbox, axis=0)
    x_max, y_max = np.max(bbox, axis=0)
    # Define the default axis-aligned rectangle
    rect = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    dtype=np.float32)
    # Compute the Euclidean distance between each vertex of bbox and rect
    return np.linalg.norm(bbox - rect, axis=0).sum()


def rotate_bbox(bbox, theta, anchor=None):
    """
    Rotates a bounding box by a given angle around an anchor point.

    Args:
        bbox (np.ndarray): A 4x2 array representing the four vertices of the bounding box.
        theta (float): Rotation angle in radians.
        anchor (np.ndarray, optional): A 2x1 array representing the anchor point (x, y) for rotation.
                                       If None, the first point of the bbox is used as the anchor.

    Returns:
        np.ndarray: The rotated bounding box as a 4x2 array.
    """
    points = bbox.T  # Transpose to shape (2, 4) for matrix multiplication
    if anchor is None:
        anchor = points[:, :1]  # Use the first point as the anchor
    rotated_points = np.dot(get_rotate_mat(theta), points - anchor) + anchor
    return rotated_points.T  # Transpose back to shape (4, 2)


def find_min_rect_angle(bbox, rank_num=10):
    """
    Finds the best rotation angle for a bounding box to minimize the difference from an axis-aligned rectangle.

    This function iterates over a range of angles, rotates the bounding box, and calculates the error
    between the rotated bbox and the default axis-aligned rectangle. It selects the angle that
    results in the smallest cumulative error.

    Args:
        bbox (np.ndarray): A 4x2 array representing the four vertices of the bounding box.
        rank_num (int, optional): Number of top candidates to consider based on area for error calculation.
                                   Default is 10.

    Returns:
        float: The best rotation angle in radians that minimizes the error.
    """
    areas = []
    # Generate angles from -90 to 90 degrees in radians
    angles = np.arange(-90, 90) / 180 * math.pi
    for theta in angles:
        rotated_bbox = rotate_bbox(bbox, theta)
        x_min, y_min = np.min(rotated_bbox, axis=0)
        x_max, y_max = np.max(rotated_bbox, axis=0)
        # Calculate the area of the axis-aligned bounding rectangle
        areas.append((x_max - x_min) * (y_max - y_min))

    best_angle, min_error = -1, float('inf')
    # Select the top `rank_num` smallest areas to consider for minimal error
    for idx in np.argsort(areas)[:rank_num]:
        rotated_bbox = rotate_bbox(bbox, angles[idx])
        error = calc_error_from_rect(rotated_bbox)
        if error < min_error:
            best_angle, min_error = angles[idx], error

    return best_angle


def generate_score_geo_maps(image, word_bboxes, map_scale=0.5):
    """
    Generates score and geometry maps for a given image and its word bounding boxes.

    These maps are used as ground truth targets for training the EAST model. The score map indicates
    the presence of text regions, while the geometry map encodes the geometry information of the text.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        word_bboxes (list or np.ndarray): A list of word bounding boxes, each represented as a 4x2 array.
        map_scale (float, optional): Scaling factor to reduce the resolution of the maps. Default is 0.5.

    Returns:
        tuple: A tuple containing:
            - score_map (np.ndarray): A 2D array indicating text regions.
            - geo_map (np.ndarray): A 3D array encoding geometric information of text regions.
    """
    img_h, img_w = image.shape[:2]
    map_h, map_w = int(img_h * map_scale), int(img_w * map_scale)
    inv_scale = int(1 / map_scale)

    # Initialize score and geometry maps with zeros
    score_map = np.zeros((map_h, map_w, 1), np.float32)
    geo_map = np.zeros((map_h, map_w, 5), np.float32)

    word_polys = []

    for bbox in word_bboxes:
        # Shrink the bounding box to create tighter text regions
        poly = np.around(map_scale * shrink_bbox(bbox)).astype(np.int32)
        word_polys.append(poly)

        # Create a mask for the center of the text region
        center_mask = np.zeros((map_h, map_w), np.float32)
        cv2.fillPoly(center_mask, [poly], 1)

        # Find the optimal rotation angle for the bounding box
        theta = find_min_rect_angle(bbox)
        # Rotate the bounding box by the optimal angle
        rotated_bbox = rotate_bbox(bbox, theta) * map_scale
        x_min, y_min = np.min(rotated_bbox, axis=0)
        x_max, y_max = np.max(rotated_bbox, axis=0)

        # Define the anchor point for rotation
        anchor = bbox[0] * map_scale
        rotated_x, rotated_y = get_rotated_coords(map_h, map_w, theta, anchor)

        # Calculate distances from the rotated coordinates to the rectangle boundaries
        d1, d2 = rotated_y - y_min, y_max - rotated_y
        d1[d1 < 0] = 0
        d2[d2 < 0] = 0
        d3, d4 = rotated_x - x_min, x_max - rotated_x
        d3[d3 < 0] = 0
        d4[d4 < 0] = 0

        # Update geometry maps with the calculated distances and angle
        geo_map[:, :, 0] += d1 * center_mask * inv_scale
        geo_map[:, :, 1] += d2 * center_mask * inv_scale
        geo_map[:, :, 2] += d3 * center_mask * inv_scale
        geo_map[:, :, 3] += d4 * center_mask * inv_scale
        geo_map[:, :, 4] += theta * center_mask

    # Fill the score map with polygons indicating text regions
    cv2.fillPoly(score_map, word_polys, 1)

    return score_map, geo_map


class EASTDataset(Dataset):
    """
    Custom Dataset class for the EAST text detection model.

    This dataset wraps around an existing dataset and generates score and geometry maps
    required for training the EAST model.

    Args:
        dataset (torch.utils.data.Dataset): The base dataset containing images and their annotations.
        map_scale (float, optional): Scaling factor to reduce the resolution of the maps. Default is 0.5.
        to_tensor (bool, optional): If True, converts the outputs to PyTorch tensors. Default is True.
    """

    def __init__(self, dataset, map_scale=0.5, to_tensor=True):
        self.dataset = dataset
        self.map_scale = map_scale
        self.to_tensor = to_tensor

    def __getitem__(self, idx):
        """
        Retrieves the processed data sample at the specified index.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The input image tensor.
                - score_map (torch.Tensor): The score map tensor.
                - geo_map (torch.Tensor): The geometry map tensor.
                - roi_mask (torch.Tensor): The region of interest mask tensor.
        """
        # Retrieve the image, word bounding boxes, and ROI mask from the base dataset
        image, word_bboxes, roi_mask = self.dataset[idx]
        # Generate score and geometry maps for the image
        score_map, geo_map = generate_score_geo_maps(image, word_bboxes, map_scale=self.map_scale)

        # Resize the ROI mask to match the map dimensions
        mask_size = int(image.shape[0] * self.map_scale), int(image.shape[1] * self.map_scale)
        roi_mask = cv2.resize(roi_mask, dsize=mask_size)
        if roi_mask.ndim == 2:
            roi_mask = np.expand_dims(roi_mask, axis=2)  # Add channel dimension if missing

        if self.to_tensor:
            # Convert image and maps to PyTorch tensors and rearrange dimensions to (C, H, W)
            image = torch.Tensor(image).permute(2, 0, 1)
            score_map = torch.Tensor(score_map).permute(2, 0, 1)
            geo_map = torch.Tensor(geo_map).permute(2, 0, 1)
            roi_mask = torch.Tensor(roi_mask).permute(2, 0, 1)

        return image, score_map, geo_map, roi_mask

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)

