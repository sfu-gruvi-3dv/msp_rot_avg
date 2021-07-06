from data.util.colmap_database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids
import os
import numpy as np

db = COLMAPDatabase('/mnt/Exp_5/Westerminster_colmap/db.db')
rows = db.execute("SELECT * FROM two_view_geometries")

out = next(rows)
pair_id, n_matches, _, matches, config, F, E, H = out
pair_id = pair_id_to_image_ids(pair_id)
matches = blob_to_array(matches, np.uint32).reshape(n_matches, 2)
F = blob_to_array(F, np.float64).reshape(3, 3)
E = blob_to_array(E, np.float64).reshape(3, 3)
H = blob_to_array(H, np.float64).reshape(3, 3)
print(E)

keypoints = dict(
    (image_id, blob_to_array(data, np.float32, (-1, 2)))
    for image_id, data in db.execute(
        "SELECT image_id, data FROM keypoints WHERE image_id=1"))

keypoint_exp = len(keypoints)