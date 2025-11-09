# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# This script is based on an original implementation by True Price.

import sqlite3
import sys

import numpy as np
from pathlib import Path

from pycolmap import (
    Camera,
    RANSACOptions,
    estimate_essential_matrix,
    estimate_fundamental_matrix,
    estimate_homography_matrix,
)

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(
    MAX_IMAGE_ID
)

CREATE_POSE_PRIORS_TABLE = """CREATE TABLE IF NOT EXISTS pose_priors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    position BLOB,
    coordinate_system INTEGER NOT NULL,
    position_covariance BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_POSE_PRIORS_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = int((pair_id - image_id2) / MAX_IMAGE_ID)
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if blob is None:  # idk why this happens
        return np.array([], dtype=dtype)
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_pose_priors_table = lambda: self.executescript(
            CREATE_POSE_PRIORS_TABLE
        )
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

        # * added
        self.feat_shape = 6  # must be 2, 4, or 6 (default)
        self.ransac_opt = RANSACOptions(
            max_error=4.0,
            min_inlier_ratio=0.25,
            confidence=0.999,
            min_num_trials=100,
            max_num_trials=10000,
        )  # default COLMAP options (two_view_geometry.h)
        # Lazy-load tables only if they already exist
        try:
            self.image_ids = self.read_all_image_ids()
            self.image_names = self.read_all_image_names()
            self.keypoints = self.read_all_keypoints()
            self.matches = self.read_all_matches()
            self.two_view_geom = self.read_all_two_view_geometries()
        except sqlite3.OperationalError:
            # Tables haven't been created yet
            self.image_ids = {}
            self.image_names = {}
            self.keypoints = {}
            self.matches = {}
            self.two_view_geom = {}


    def add_camera(
        self,
        model,
        width,
        height,
        params,
        prior_focal_length=False,
        camera_id=None,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self,
        name,
        camera_id,
        image_id=None,
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?)", (image_id, name, camera_id)
        )
        return cursor.lastrowid

    def add_pose_prior(
        self, image_id, position, coordinate_system=-1, position_covariance=None
    ):
        position = np.asarray(position, dtype=np.float64)
        if position_covariance is None:
            position_covariance = np.full((3, 3), np.nan, dtype=np.float64)
        self.execute(
            "INSERT INTO pose_priors VALUES (?, ?, ?, ?)",
            (
                image_id,
                array_to_blob(position),
                coordinate_system,
                array_to_blob(position_covariance),
            ),
        )

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )
        self.keypoints[image_id] = keypoints

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )
        self.matches[(image_id1, image_id2)] = matches

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
                array_to_blob(qvec),
                array_to_blob(tvec),
            ),
        )

    # * added to template
    def append_keypoints(self, image_id, keypoints):
        result = self.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,)
        ).fetchone()
        if result is None:
            self.add_keypoints(image_id, keypoints)
            return

        old_rows, old_cols, old_data_blob = result
        assert old_cols == keypoints.shape[1], "Column mismatch in keypoints!"

        old_keypoints = blob_to_array(old_data_blob, np.float32, (old_rows, old_cols))
        combined_keypoints = np.vstack((old_keypoints, keypoints.astype(np.float32)))
        combined_blob = array_to_blob(combined_keypoints)

        # Update database with combined keypoints
        self.execute(
            "UPDATE keypoints SET rows = ?, cols = ?, data = ? WHERE image_id = ?",
            (
                combined_keypoints.shape[0],
                combined_keypoints.shape[1],
                combined_blob,
                image_id,
            ),
        )
        self.commit()

    def append_descriptors(self, image_id, descriptors):
        result = self.execute(
            "SELECT data, rows, cols FROM descriptors WHERE image_id = ?", (image_id,)
        ).fetchone()

        if result is None:
            self.add_descriptors(image_id, descriptors)
            return

        existing_blob, rows, cols = result
        # Convert blob to numpy array
        existing_descriptors = np.frombuffer(existing_blob, dtype=np.uint8).reshape(
            rows, cols
        )

        # Concatenate existing and dummy descriptors
        combined = np.vstack((existing_descriptors, descriptors))

        # Update descriptors row with combined data
        self.execute(
            "UPDATE descriptors SET rows = ?, cols = ?, data = ? WHERE image_id = ?",
            (combined.shape[0], combined.shape[1], array_to_blob(combined), image_id),
        )

    def append_matches(self, image_id1, image_id2, new_matches):
        assert len(new_matches.shape) == 2
        assert new_matches.shape[1] == 2

        if image_id1 > image_id2:
            new_matches = new_matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        new_matches = np.asarray(new_matches, np.uint32)

        result = self.execute(
            "SELECT rows, cols, data FROM matches WHERE pair_id = ?", (pair_id,)
        ).fetchone()

        if result is None:
            self.add_matches(image_id1, image_id2, new_matches)
            return

        old_rows, old_cols, old_data_blob = result
        old_matches = blob_to_array(old_data_blob, np.uint32, (old_rows, old_cols))
        assert old_cols == 2, "Existing matches should have 2 columns"

        # Append new matches
        if len(old_matches) == 0:
            combined_matches = new_matches
        else:
            combined_matches = np.vstack((old_matches, new_matches))

        # Update database with combined matches blob
        combined_blob = array_to_blob(combined_matches)
        self.execute(
            "UPDATE matches SET rows = ?, cols = ?, data = ? WHERE pair_id = ?",
            (
                combined_matches.shape[0],
                combined_matches.shape[1],
                combined_blob,
                pair_id,
            ),
        )
        self.matches[(image_id1, image_id2)] = combined_matches

    def clear_keypoints(self):
        try:
            self.execute("DELETE FROM keypoints")
            return True
        except:
            return False
        
    def clear_descriptors(self):
        try:
            self.execute("DELETE FROM descriptors")
            return True
        except:
            return False

    def clear_matches(self):
        try:
            self.execute("DELETE FROM matches")
            return True
        except:
            return False

    def clear_two_view_geometries(self):
        try:
            self.execute("DELETE FROM two_view_geometries")
            return True
        except:
            return False

    def read_cameras(self):
        res = self.execute("SELECT * FROM cameras")
        cameras = res.fetchall()
        assert len(cameras) == 1  # ! I think ?
        camera_id, model, width, height, params, prior_focal_length = cameras[0]
        intrinsics = blob_to_array(params, np.float64, (4,))

        cam = Camera.create(camera_id, model, prior_focal_length, width, height)
        cam.params = intrinsics
        return cam

    def read_all_keypoints(self):
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, self.feat_shape)))
            for image_id, data in self.execute("SELECT image_id, data FROM keypoints")
        )
        return keypoints

    def read_all_matches(self):
        matches = dict(
            (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
            for pair_id, data in self.execute("SELECT pair_id, data FROM matches")
        )
        return matches

    def idx2kps(self, image_id, indexes):
        kps = self.get_keypoints(image_id)
        # O(n) to preserve order
        ordered_kps = np.array([kps[i] for i in indexes])
        return ordered_kps

    def get_essential_matrix(self, pts1, pts2):
        cam = self.read_cameras()
        res = estimate_essential_matrix(pts1, pts2, cam, cam, self.ransac_opt)
        try:
            return res["E"]
        except:
            return np.eye(3)

    def get_fundamental_matrix(self, pts1, pts2):
        res = estimate_fundamental_matrix(pts1, pts2, self.ransac_opt)
        try:
            return res["F"]
        except:
            return np.eye(3)

    def get_homography_matrix(self, pts1, pts2):
        res = estimate_homography_matrix(pts1, pts2, self.ransac_opt)
        try:
            return res["H"]
        except:
            return np.eye(3)

    def compute_matrices(self, image_id1, image_id2, matches):
        pts1 = self.idx2kps(image_id1, matches[:, 0])[:, :2]
        pts2 = self.idx2kps(image_id2, matches[:, 1])[:, :2]

        F = self.get_fundamental_matrix(pts1, pts2)
        E = self.get_essential_matrix(pts1, pts2)
        H = self.get_homography_matrix(pts1, pts2)

        return F, E, H

    # for debugging (not currently using)
    def read_all_two_view_geometries(self):
        two_view_geom = dict()

        for pair_id, data, F_blob, E_blob, H_blob, qvec_blob, tvec_blob in self.execute(
            "SELECT pair_id, data, F, E, H, qvec, tvec FROM two_view_geometries"
        ):
            img_id1, img_id2 = pair_id_to_image_ids(pair_id)
            matches = blob_to_array(data, np.uint32, (-1, 2))
            F = blob_to_array(F_blob, np.float64, (3, 3))
            E = blob_to_array(E_blob, np.float64, (3, 3))
            H = blob_to_array(H_blob, np.float64, (3, 3))
            qvec = blob_to_array(
                qvec_blob,
                np.float64(
                    4,
                ),
            )
            tvec = blob_to_array(
                tvec_blob,
                np.float64(
                    3,
                ),
            )
            two_view_geom[pair_id] = {
                "matches": matches,
                "F": F,
                "E": E,
                "H": H,
                "qvec": qvec,
                "tvec": tvec,
            }
        return two_view_geom

    def read_all_image_ids(self):
        ids = dict(
            (name, image_id)
            for image_id, name in self.execute("SELECT image_id, name FROM images")
        )
        return ids

    def read_all_image_names(self):
        names = dict(
            (image_id, name)
            for image_id, name in self.execute("SELECT image_id, name FROM images")
        )
        return names

    def get_n_imgs(self):
        names = dict(
            (name, image_id)
            for image_id, name in self.execute("SELECT image_id, name FROM images")
        )
        return len(names.keys())

    def get_image_id(self, image_name):
        if self.image_ids is None:
            self.image_ids = self.read_all_image_ids()
        if image_name not in self.image_ids.keys():
            print(f"Image {image_name} does not exist in database.")
            return None
        else:
            return self.image_ids[image_name]

    def get_image_name(self, image_id):
        if self.image_names is None:
            self.image_names = self.read_all_image_names()
        if image_id not in self.image_names.keys():
            print(f"Invalid image id ({image_id})")
            return None
        return self.image_names[image_id]

    def get_keypoints(self, img_id):
        if self.keypoints is None:
            self.keypoints = self.read_all_keypoints()
        if img_id not in self.keypoints.keys():
            return None
        return self.keypoints[img_id]

    def get_matches(self, img_id_1, img_id_2):
        # ! maybe less ugly idk
        if self.matches is None:
            self.matches = self.read_all_matches()
        if (img_id_1, img_id_2) not in self.matches.keys() and (
            img_id_2,
            img_id_1,
        ) not in self.matches.keys():
            # print(
            #     f"Matches for images ({img_id_1}, {img_id_2}) does not exist in database."
            # )
            return None
        elif (img_id_1, img_id_2) in self.matches.keys():
            if len(self.matches[(img_id_1, img_id_2)]) != 0:
                return self.matches[(img_id_1, img_id_2)]
            else:
                return None
        else:
            if len(self.matches[(img_id_2, img_id_1)]) != 0:
                return np.flip(self.matches[(img_id_2, img_id_1)], axis=1)
            else:  # ! idk why this is in db
                return None

    def delete_keypoints(self, image_id):
        try:
            self.execute("DELETE FROM keypoints WHERE image_id = ?", (image_id,))
            return True
        except:
            return False

    def delete_matches(self, img_id_1, img_id_2):
        try:
            self.execute(
                "DELETE FROM matches WHERE pair_id = ?",
                (image_ids_to_pair_id(img_id_1, img_id_2),),
            )
            return True
        except:
            return False

    def delete_two_view_geometries(self, img_id_1, img_id_2):
        try:
            self.execute(
                "DELETE FROM two_view_geometries WHERE pair_id = ?",
                (image_ids_to_pair_id(img_id_1, img_id_2),),
            )
            return True
        except:
            return False
        
        
    #added
    def add_keypoints_npy(self, image_id, kp_path):
        kp_array = np.load(kp_path)
        self.add_keypoints(image_id, kp_array)

    def add_descriptors_npy(self, image_id, desc_path):
        descriptors = np.load(desc_path)
        self.save_image_descriptors_npy(image_id, descriptors)
    
    def save_image_descriptors_npy(self, image_id, descriptors):
        self.add_descriptors(image_id, descriptors)
        self.commit()




def example_usage():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()

    # Create dummy cameras.

    model1, width1, height1, params1 = (
        0,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0)),
    )
    model2, width2, height2, params2 = (
        2,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0, 0.1)),
    )

    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images.

    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.
    #
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches.

    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)

    # Create dummy pose_priors.

    pos1 = np.random.rand(3, 1) * np.random.randint(10)
    pos2 = np.random.rand(3, 1) * np.random.randint(10)
    pos3 = np.random.rand(3, 1) * np.random.randint(10)

    cov3 = np.random.rand(3, 3) * np.random.randint(10)

    pose_prior1 = [image_id1, pos1, 1, None]
    pose_prior2 = [image_id2, pos2, -1, None]
    pose_prior3 = [image_id3, pos3, 0, cov3]

    db.add_pose_prior(*pose_prior1)
    db.add_pose_prior(*pose_prior2)
    db.add_pose_prior(*pose_prior3)

    # Convert unset covariance to nan matrix for later check
    pose_prior1[3] = np.full((3, 3), np.nan, dtype=np.float64)
    pose_prior2[3] = np.full((3, 3), np.nan, dtype=np.float64)

    # Commit the data to the file.

    db.commit()

    # Read and check cameras.

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints.

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
    )

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    matches = dict(
        (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Read and check pose_priors

    rows = db.execute("SELECT * FROM pose_priors")

    img_id1, pos1, coord_sys1, cov1 = next(rows)
    img_id2, pos2, coord_sys2, cov2 = next(rows)
    img_id3, pos3, coord_sys3, cov3 = next(rows)

    assert pose_prior1[0] == img_id1
    assert pose_prior2[0] == img_id2
    assert pose_prior3[0] == img_id3

    assert pose_prior1[1].all() == blob_to_array(pos1, np.float64, (3, 1)).all()
    assert pose_prior2[1].all() == blob_to_array(pos2, np.float64, (3, 1)).all()
    assert pose_prior3[1].all() == blob_to_array(pos3, np.float64, (3, 1)).all()

    assert pose_prior1[2] == coord_sys1
    assert pose_prior2[2] == coord_sys2
    assert pose_prior3[2] == coord_sys3

    assert pose_prior1[3].all() == blob_to_array(cov1, np.float64, (3, 3)).all()
    assert pose_prior2[3].all() == blob_to_array(cov2, np.float64, (3, 3)).all()
    assert pose_prior3[3].all() == blob_to_array(cov3, np.float64, (3, 3)).all()

    # Clean up.

    db.close()

    # if os.path.exists(args.database_path):
    #     os.remove(args.database_path)


if __name__ == "__main__":
    example_usage()