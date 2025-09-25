# preprocess_to_hdf5.py
import numpy as np
import h5py
from pathlib import Path
import os
import argparse

_LINK_NAMES = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
_MESH_NAMES = ["link_1", "link_2_orange", "link_2_grey", "link_3",
               "link_4_orange", "link_4_grey", "link_5",
               "link_6_orange", "link_6_grey", "link_7"]

def _as_fixed_ascii(arr_obj):
    arr = np.asarray(arr_obj)
    max_len = max(int(len(x)) for x in arr)
    return arr.astype(f"S{max_len}")

def contactN2linkid(contact_num, max_contacts=10):
    linkids = []
    for i in range(len(contact_num)):
        c_n = contact_num[i]
        linkid = np.ones(max_contacts, dtype=int) * -1
        contact_idx = 0
        for j in range(len(c_n)):
            n_j = c_n[j]
            for n in range(n_j):
                if contact_idx < max_contacts:
                    linkid[contact_idx] = j
                    contact_idx += 1
                else:
                    break
            if contact_idx >= max_contacts:
                break
        linkids.append(linkid)
    return np.array(linkids)

def npz_to_hdf5(file_name: str,
                dir_name: str,
                robot_name: str = "kuka_iiwa_14",
                max_num_contacts: int = 10):
    root = Path(__file__).resolve().parent
    data_dir = root / "dataset" / dir_name
    data_dir.mkdir(parents=True, exist_ok=True)

    npz_path = data_dir / f"{file_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset npz: {npz_path}")

    # robot config
    if robot_name == "kuka_iiwa_14":
        num_joints = 7
        max_contact_id = 53642
        padding_id = max_contact_id + 1
    else:
        raise ValueError(f"Unknown robot_name: {robot_name}")

    # mesh dict (NO JAX)
    mesh_dict_path = (root / ".." / robot_name / "mesh_data" / "mesh_data_dict_no_jax.npy")
    if not mesh_dict_path.exists():
        raise FileNotFoundError(f"Missing mesh_data_dict_no_jax.npy at {mesh_dict_path}")
    m = np.load(mesh_dict_path, allow_pickle=True).item()

    # distance tables
    geod_dir = (root / ".." / robot_name / "mesh_geodesic")
    for mn in _MESH_NAMES:
        if not (geod_dir / f"distance_{mn}.npz").exists():
            raise FileNotFoundError(f"Missing distance table for {mn} in {geod_dir}")

    # ---- load raw episode data ----
    raw = np.load(npz_path)
    contacts = raw["contacts"]                     # [eps, ep_len, M, >=14]
    eps, ep_len, M = contacts.shape[:3]
    T = eps * ep_len

    # split contact fields
    ids = contacts[:, :, :, 13]                    # float with NaNs
    pos = contacts[:, :, :, :3]                    # xyz
    frc = contacts[:, :, :, 3:6]                   # force vector
    nrm = contacts[:, :, :, 6:9]                   # normal vector    

    # fill NaNs (IDs get padding_id; vectors get 0)
    ids = np.nan_to_num(ids, nan=padding_id).astype(np.int64)
    pos = np.nan_to_num(pos, nan=0.0).astype(np.float32)
    frc = np.nan_to_num(frc, nan=0.0).astype(np.float32)
    nrm = np.nan_to_num(nrm, nan=0.0).astype(np.float32)

    # pad K dimension up to max_num_contacts
    def pad_k(a, pad_val):
        need = max_num_contacts - a.shape[2]
        if need <= 0: return a
        pad_shape = list(a.shape); pad_shape[2] = need
        return np.concatenate([a, np.full(pad_shape, pad_val, dtype=a.dtype)], axis=2)

    ids = pad_k(ids, padding_id)                  # [eps, ep_len, K]
    pos = pad_k(pos, 0.0)                         # [eps, ep_len, K, 3]
    frc = pad_k(frc, 0.0)                           
    nrm = pad_k(nrm, 0.0)

    # flatten time to [T, ...]
    global_contact_ids = ids.reshape(T, max_num_contacts)            # (T,K)
    contact_positions   = pos.reshape(T, max_num_contacts * 3)       # (T,3K)  <-- match your code
    contact_forces      = frc.reshape(T, max_num_contacts * 3)       # (T,3K)
    surface_normals     = nrm.reshape(T, max_num_contacts * 3)       # (T,3K)

    # ---- feasible_contact_num (per-link counts per time step) ----
    # compute from *unflattened* contacts using link names from mesh dict
    gid2link = np.asarray(m["globalid2linkname"])
    link_index = {name: i for i, name in enumerate(_LINK_NAMES)}
    feasible_counts = np.zeros((T, len(_LINK_NAMES)), dtype=np.int32)
    t = 0
    for e in range(eps):
        for t_step in range(ep_len):
            c_num = np.zeros(len(_LINK_NAMES), dtype=np.int32)
            for j in range(M):
                if np.isnan(contacts[e, t_step, j, 0]):  # original NaN row → ignore
                    continue
                gid = int(contacts[e, t_step, j, 13])
                link_name = gid2link[gid]
                # handle np.str_/bytes
                if isinstance(link_name, bytes):
                    link_name = link_name.decode("utf-8")
                idx = link_index.get(str(link_name), None)
                if idx is not None:
                    c_num[idx] += 1
            feasible_counts[t] = c_num
            t += 1

    # ---- precompute link_ids ----
    link_ids = contactN2linkid(feasible_counts, max_contacts=max_num_contacts)  # (T, K)


    # ---- joints & stats ----
    def flat7(x): return raw[x].reshape(T, num_joints).astype(np.float32)
    jp = flat7("joint_pos")
    jv = flat7("joint_vel")
    jc = flat7("joint_tau_cmd")
    je = flat7("joint_tau_ext_gt")

    def stats(x):
        m = x.mean(axis=0).astype(np.float32)
        s = x.std(axis=0).astype(np.float32)
        s[s == 0] = 1.0
        return m, s

    jp_mean, jp_std = stats(jp)
    jv_mean, jv_std = stats(jv)
    jc_mean, jc_std = stats(jc)
    je_mean, je_std = stats(je)

    # ---- write HDF5 ----
    out_h5 = data_dir / f"{file_name}.h5"
    if out_h5.exists():
        os.remove(out_h5)

    with h5py.File(out_h5, "w") as f:
        # attrs
        f.attrs["robot_name"] = robot_name
        f.attrs["num_joints"] = num_joints
        f.attrs["max_contact_id"] = max_contact_id
        f.attrs["padding_id"] = padding_id
        f.attrs["eps"] = eps
        f.attrs["ep_len"] = ep_len
        f.attrs["total_steps"] = T
        f.attrs["max_num_contacts"] = max_num_contacts

        # core datasets (chunked + gzip)
        def put(name, arr):
            f.create_dataset(name, data=arr, compression="gzip",
                             compression_opts=4, shuffle=True, chunks=True)

        put("global_contact_ids", global_contact_ids)      # (T,K)
        put("contact_positions", contact_positions)        # (T,3K)
        put("contact_forces", contact_forces)              # (T,3K)
        put("surface_normals", surface_normals)            # (T,3K)
        put("feasible_contact_num", feasible_counts)       # (T,7)
        put("link_ids", link_ids)                          # (T,K)

        put("joint_pos", jp)
        put("joint_vel", jv)
        put("joint_tau_cmd", jc)
        put("joint_tau_ext_gt", je)

        stats_grp = f.create_group("stats")
        stats_grp.create_dataset("joint_pos_mean", data=jp_mean)
        stats_grp.create_dataset("joint_pos_std",  data=jp_std)
        stats_grp.create_dataset("joint_vel_mean", data=jv_mean)
        stats_grp.create_dataset("joint_vel_std",  data=jv_std)
        stats_grp.create_dataset("joint_tau_cmd_mean", data=jc_mean)
        stats_grp.create_dataset("joint_tau_cmd_std",  data=jc_std)
        stats_grp.create_dataset("joint_tau_ext_gt_mean", data=je_mean)
        stats_grp.create_dataset("joint_tau_ext_gt_std",  data=je_std)

        # mappings (store what your helpers need)
        mgrp = f.create_group("mappings")
        put("mappings/globalid2localid", np.asarray(m["globalid2localid"], dtype=np.int64))
        put("mappings/global_mesh_ids",  np.asarray(m["global_mesh_ids"], dtype=np.int64))
        put("mappings/global_start_end_indices", np.asarray(m["global_start_end_indices"], dtype=np.int64))
        put("mappings/global_geom_ids",  np.asarray(m["global_geom_ids"], dtype=np.int64))
        mgrp.create_dataset("global_mesh_names", data=_as_fixed_ascii(m["global_mesh_names"]))
        mgrp.create_dataset("globalid2linkname", data=_as_fixed_ascii(m["globalid2linkname"]))
        put("mappings/global_face_center_list", np.asarray(m["global_face_center_list"], dtype=np.float32))
        put("mappings/global_normal_list",      np.asarray(m["global_normal_list"], dtype=np.float32))
        put("mappings/global_rot_mat_list",     np.asarray(m["global_rot_mat_list"], dtype=np.float32))
        put("mappings/global_face_vertices_list", np.asarray(m["global_face_vertices_list"], dtype=np.float32))

        # distance tables (one dataset per mesh)
        dgrp = f.create_group("distance_tables")
        for mn in _MESH_NAMES:
            tbl = np.load(geod_dir / f"distance_{mn}.npz")["distances_faces_sliced"]
            dgrp.create_dataset(mn, data=tbl, compression="gzip",
                                compression_opts=4, shuffle=True, chunks=True)

    print(f"[OK] wrote {out_h5}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_name", default="dataset_batch_1_200eps")
    ap.add_argument("--dir_name",  default="data-link6-link7-3contact_100_v5")
    ap.add_argument("--robot_name", default="kuka_iiwa_14")
    ap.add_argument("--max_num_contacts", type=int, default=10)
    args = ap.parse_args()
    npz_to_hdf5(**vars(args))
