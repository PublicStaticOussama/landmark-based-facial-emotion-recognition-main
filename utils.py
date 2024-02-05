import numpy as np
import math


def getX(point):
    return point.x
def getY(point):
    return point.y
def getZ(point):
    return point.z

getX_vec = np.vectorize(getX)
getY_vec = np.vectorize(getY)
getZ_vec = np.vectorize(getZ)

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    
    arr2 = upper_tri_masking(arr)

    return arr2

def euclidean_dist_sqrd(p1x, p2x, p1y, p2y):
    return (p1x-p2x)**2 + (p1y-p2y)**2

def get_face_dimensions_squared(all_landmarks, width, height):
    lx = all_landmarks[234].x * width
    ly = all_landmarks[234].y * height
    rx = all_landmarks[454].x * width
    ry = all_landmarks[454].y * height
    # -----------------------
    ux = all_landmarks[10].x * width
    uy = all_landmarks[10].y * height
    dx = all_landmarks[152].x * width
    dy = all_landmarks[152].y * height
    
    w = euclidean_dist_sqrd(lx, rx, ly, ry)
    h = euclidean_dist_sqrd(ux, dx, uy, dy)
    return w, h

def calculate_all_distances(landmark_list, width, height, face_width_squared, face_height_squared):
    scaler = math.sqrt(face_width_squared + face_height_squared)
    landmark_pairs = cartesian_product(landmark_list,landmark_list)
    distances = np.empty((len(landmark_pairs)))

    landmark_pairs_x = getX_vec(landmark_pairs)*width
    landmark_pairs_y = getY_vec(landmark_pairs)*height
    # landmark_pairs_z = getZ_vec(landmark_pairs)

    landmark_pairs_x_2 = np.square(landmark_pairs_x[:,0] - landmark_pairs_x[:,1])
    landmark_pairs_y_2 = np.square(landmark_pairs_y[:,0] - landmark_pairs_y[:,1])
    # landmark_pairs_z_2 = np.square(landmark_pairs_z[:,0] - landmark_pairs_z[:,1])

    # distances = np.sqrt(landmark_pairs_x_2 + landmark_pairs_y_2 + landmark_pairs_z_2)
    distances = np.sqrt(landmark_pairs_x_2 + landmark_pairs_y_2) / scaler

    return distances


def draw_styled_landmarks(image, mp_drawing, multi_face_landmarks, mp_face_mesh, mp_drawing_styles):
    for face_landmarks in multi_face_landmarks: # loop through all facial landmarks --------------------------
      mp_drawing.draw_landmarks( # drawing lines between points of the landmarks --------------------------
          image=image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())

      mp_drawing.draw_landmarks( # drawing face contour --------------------------
          image=image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())

      mp_drawing.draw_landmarks( # drawing the irises of the eye --------------------------
          image=image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())

