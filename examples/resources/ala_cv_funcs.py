#!/usr/bin/env python3
"""
State und descriptor functions for capped alanine dipetide.
"""
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals


def alpha_R(traj, skip=1):
    """
    Calculate alpha_R state function.

    The alpha_R state is defined in the space of the two dihedral angles psi
    and phi, for a configuration to belong to the state:
        phi: -pi < phi < 0
        psi: -50 degree < psi < 30 degree

    Parameters
    ----------
    traj : asyncmd.Trajectory
        The trajectory for which the state function is calculated.
    skip : int, optional
        stride for trajectory iteration, by default 1

    Returns
    -------
    numpy.ndarray, shape=(n_frames,)
        Array with boolean values for every configuration on the trajectory
        indicating if a configuration falls into the state or not.
    """
    u = mda.Universe(traj.structure_file, *traj.trajectory_files)
    psi_ag = u.select_atoms("resname ALA and name N")  # idx 6
    psi_ag += u.select_atoms("resname ALA and name CA")  # idx 8
    psi_ag += u.select_atoms("resname ALA and name C")  # idx 14
    psi_ag += u.select_atoms("resname NME and name N")  # idx 16
    phi_ag = u.select_atoms("resname ACE and name C")  # idx 4
    phi_ag += u.select_atoms("resname ALA and name N")  # idx 6
    phi_ag += u.select_atoms("resname ALA and name CA")  # idx 8
    phi_ag += u.select_atoms("resname ALA and name C")  # idx 14
    # empty arrays to fill
    state = np.full((len(u.trajectory[::skip]),), False, dtype=bool)
    phi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    # make sure MDAnalysis closes the underlying trajectory files directly
    u.trajectory.close()
    # phi: -pi -> 0
    # psi: > -50 but smaller 30 degree
    deg = 180/np.pi
    state[(phi <= 0) & (-50/deg <= psi) & (psi <= 30/deg)] = True
    return state


def C7_eq(traj, skip=1):
    """
    Calculate C7_eq state function.

    The C7_eq state is defined in the space of the two dihedral angles psi
    and phi, for a configuration to belong to the state:
        phi: -pi < phi < 0
        psi: 120 degree < psi < 200 degree

    Parameters
    ----------
    traj : asyncmd.Trajectory
        The trajectory for which the state function is calculated.
    skip : int, optional
        stride for trajectory iteration, by default 1

    Returns
    -------
    numpy.ndarray, shape=(n_frames,)
        Array with boolean values for every configuration on the trajectory
        indicating if a configuration falls into the state or not.
    """
    u = mda.Universe(traj.structure_file, *traj.trajectory_files)
    psi_ag = u.select_atoms("resname ALA and name N")  # idx 6
    psi_ag += u.select_atoms("resname ALA and name CA")  # idx 8
    psi_ag += u.select_atoms("resname ALA and name C")  # idx 14
    psi_ag += u.select_atoms("resname NME and name N")  # idx 16
    phi_ag = u.select_atoms("resname ACE and name C")  # idx 4
    phi_ag += u.select_atoms("resname ALA and name N")  # idx 6
    phi_ag += u.select_atoms("resname ALA and name CA")  # idx 8
    phi_ag += u.select_atoms("resname ALA and name C")  # idx 14
    # empty arrays to fill
    state = np.full((len(u.trajectory[::skip]),), False, dtype=bool)
    phi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    # make sure MDAnalysis closes the underlying trajectory files directly
    u.trajectory.close()
    # phi: -pi -> 0
    # psi: 120 -> 200 degree
    deg = 180/np.pi
    state[(phi <= 0) & ((120/deg <= psi) | (-160/deg >= psi))] = True
    return state


def descriptor_func_psi_phi(traj, skip=1):
    """
    Calculate psi and phi angle internal coordiantes.

    Parameters
    ----------
    traj : asyncmd.Trajectory
        Input trajectory.
    skip : int, optional
        stride for trajectory iteration, by default 1

    Returns
    -------
    np.ndarray
        psi, phi values for trajectory, shape=(n_frames, 2)
    """
    u = mda.Universe(traj.structure_file, *traj.trajectory_files)
    psi_ag = u.select_atoms("index 6 or index 8 or index 14 or index 16")
    phi_ag = u.select_atoms("index 4 or index 6 or index 8 or index 14")
    # empty arrays to fill
    phi = np.empty((len(u.trajectory[::skip]), 1), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]), 1), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f, 0] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f, 0] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    # make sure MDAnalysis closes the underlying trajectory files directly
    u.trajectory.close()
    return np.concatenate((psi, phi), axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Calculate CV values for alanine dipeptide",
                                     )
    parser.add_argument("structure_file", type=str)
    parser.add_argument("trajectory_files", type=str, nargs="+")
    parser.add_argument("output_file", type=str)
    parser.add_argument("-f", "--function", type=str,
                        default="descriptors",
                        choices=["alphaR", "C7eq", "descriptors_psi_phi"])
    parser.add_argument("-s", "--skip", type=int, default=1)
    args = parser.parse_args()
    # NOTE: since args is a namespace args.trajectory_file will be the path to
    #       the trajectory file, i.e. we can pass args instead of an
    #       aimmd.Trajectory to the functions above
    if args.function == "descriptors_psi_phi":
        vals = descriptor_func_psi_phi(args, skip=args.skip)
    elif args.function == "alphaR":
        vals = alpha_R(args, skip=args.skip)
    elif args.function == "C7eq":
        vals = C7_eq(args, skip=args.skip)

    np.save(args.output_file, vals)
