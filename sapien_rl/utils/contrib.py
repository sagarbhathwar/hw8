import numpy as np
import transforms3d
import trimesh
from sapien.core import Pose
from shapely.geometry import Polygon
from .o3d_utils import merge_mesh, mesh2pcd, np2mesh


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def compute_relative_vel(frame_pose, frame_vel, frame_ang_vel, p_world, p_world_vel):
    p_frame = frame_pose.inv().transform(Pose(p_world)).p
    H = frame_pose.to_transformation_matrix()
    R = H[:3, :3]
    o = H[:3, 3]
    S = skew(frame_ang_vel)
    return S @ (R @ p_frame) + frame_vel - p_world_vel

def build_pose(forward, flat):
    extra = np.cross(forward, flat)
    ans = np.eye(4)
    ans[:3, :3] = np.array([forward, flat, extra])
    return Pose.from_transformation_matrix(ans)

def get_unit_box_corners():
    corners = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        ])
    return corners - [0.5, 0.5, 0.5]


def to_generalized(x):
    # [n, 3] -> [n, 4]
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=-1)


def to_normal(x):
    # [n, 4] -> [n, 3]
    return x[:, :3] / x[:, 3:]


def apply_pose_to_points(x, pose):
    return to_normal(to_generalized(x) @ pose.to_transformation_matrix().T)


def quaternion_distance(q1, q2):
    q1, q2 = Pose(q=q1), Pose(q=q2)
    q = q1.inv() * q2
    # print(q.q)
    return 1 - np.abs(q.q[0])


def pose_vec_distance(pose1, pose2):
    dist_p = np.linalg.norm(pose1.p - pose2.p)
    dist_q = quaternion_distance(pose1.q, pose2.q)
    # print(dist_p, dist_q)
    # exit(0)
    return dist_p + 0.01 * dist_q

    # return

    '''
    unit_box = get_unit_box_corners()
    t1 = get_transformation(pose1)
    t2 = get_transformation(pose2)

    corner1 = to_generalized(unit_box) @ t1.T
    corner2 = to_generalized(unit_box) @ t2.T
    #print(corner1.shape, corner2.shape)
    return np.mean(np.linalg.norm(corner1 - corner2, axis=-1))
    '''


def pose_corner_distance(pose1, pose2):
    unit_box = get_unit_box_corners()
    t1 = pose1.to_transformation_matrix()
    t2 = pose2.to_transformation_matrix()

    corner1 = to_generalized(unit_box) @ t1.T
    corner2 = to_generalized(unit_box) @ t2.T
    # print(corner1.shape, corner2.shape)
    return np.mean(np.linalg.norm(corner1 - corner2, axis=-1))


def generate_ducttape_mesh(
        inner_radius_range, width_range, height_range, n_polygons, num
):
    # Range: [low, high], low>=0, high>=low
    duct_tapes = []
    for _ in range(num):
        for i in range(n_polygons):
            r1 = np.random.uniform(inner_radius_range[0], inner_radius_range[0])
            r2 = r1 + np.random.uniform(width_range[0], width_range[1])
            height = np.random.uniform(height_range[0], height_range[1])
            scene = trimesh.Scene()
            theta1 = 2 * np.pi * i / n_polygons
            theta2 = 2 * np.pi * (i + 1) / n_polygons
            coord1 = np.array([np.cos(theta1), np.sin(theta1)])
            coord2 = np.array([np.cos(theta2), np.sin(theta2)])
            p = [coord1 * r2, coord1 * r1, coord2 * r1, coord2 * r2]
            g = trimesh.creation.extrude_polygon(Polygon(p), height)
            scene.add_geometry(g)
            duct_tapes.append(scene)
    return duct_tapes


def compute_dist2pcd(triangle_vertices, point):
    return np.min(np.linalg.norm(triangle_vertices - point, axis=-1))


def compute_dist2surface(triangle_vertices, triangle_indices, point):
    triangles = triangle_vertices[triangle_indices.reshape(-1, 3)]
    p = trimesh.triangles.closest_point(
        triangles, np.tile(point, (triangles.shape[0], 1))
    )
    return np.min(np.linalg.norm(p - point, axis=1))


def compute_dist2object(obj, point):
    point = obj.get_pose().inv().transform(Pose(point)).p
    ds = [
        compute_dist2surface(
            g.geometry.vertices, g.geometry.indices, point
        )
        for g in obj.get_collision_shapes()
    ]
    return np.min(ds)
