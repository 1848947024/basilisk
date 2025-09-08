# ISC License
#
# Copyright (c) 2025, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import inspect
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import os
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
splitPath = path.split('simulation')

from Basilisk.utilities import SimulationBaseClass, unitTestSupport, macros
from Basilisk.simulation import spacecraft
from Basilisk.simulation import gravityEffector
from Basilisk.simulation import prescribedMotionStateEffector
from Basilisk.simulation import prescribedLinearTranslation
from Basilisk.simulation import prescribedRotation1DOF
from Basilisk.simulation import linearSpringMassDamper
from Basilisk.simulation import linearTranslationOneDOFStateEffector
from Basilisk.simulation import spinningBodyOneDOFStateEffector
from Basilisk.simulation import spinningBodyTwoDOFStateEffector

from Basilisk.architecture import messaging
from Basilisk.utilities import RigidBodyKinematics as rbk

# Prescribed motion parameters
prescribed_pos_init = 0.0
prescribed_pos_ref = 0.1  # [m]
trans_axis_m = np.array([1.0, 0.0, 0.0])
r_pm_m_init = prescribed_pos_init * trans_axis_m
prescribed_trans_accel_max = 0.005  # [m/s^2]

prescribed_theta_init = 0.0 * macros.D2R
prescribed_theta_ref = 10.0 * macros.D2R  # [rad]
rot_axis_m = np.array([1.0, 0.0, 0.0])
prv_init_pm = prescribed_theta_init * rot_axis_m
sigma_pm_init = rbk.PRV2MRP(prv_init_pm)
prescribed_ang_accel_max = 0.5 * macros.D2R  # [rad/s^2]

sim_time = 20.0  # [s]

def test_prescribed_branching_spinning_body_one_dof(show_plots):

    task_name = "unitTask"
    process_name = "TestProcess"
    sim = SimulationBaseClass.SimBaseClass()
    process_rate = macros.sec2nano(0.001)
    process = sim.CreateNewProcess(process_name)
    process.addTask(sim.CreateNewTask(task_name, process_rate))

    # Create the spacecraft
    sc_object = create_spacecraft_hub()
    sim.AddModelToTask(task_name, sc_object)

    # Create prescribed motion object
    platform = create_prescribed_body(r_pm_m_init, sigma_pm_init)
    sim.AddModelToTask(task_name, platform)
    sc_object.addStateEffector(platform)

    # Create rotational motion profiler
    one_dof_rotation_profiler = prescribedRotation1DOF.PrescribedRotation1DOF()
    one_dof_rotation_profiler.ModelTag = "prescribedRotation1DOF"
    one_dof_rotation_profiler.setRotHat_M(rot_axis_m)
    one_dof_rotation_profiler.setThetaDDotMax(prescribed_ang_accel_max)
    one_dof_rotation_profiler.setThetaInit(prescribed_theta_init)
    one_dof_rotation_profiler.setCoastOptionBangDuration(1.0)
    one_dof_rotation_profiler.setSmoothingDuration(1.0)
    sim.AddModelToTask(task_name, one_dof_rotation_profiler)

    # Create the rotational motion reference message
    prescribed_rotation_msg_data = messaging.HingedRigidBodyMsgPayload()
    prescribed_rotation_msg_data.theta = prescribed_theta_ref
    prescribed_rotation_msg_data.thetaDot = 0.0  # [rad/s]
    prescribed_rotation_msg = messaging.HingedRigidBodyMsg().write(prescribed_rotation_msg_data)
    one_dof_rotation_profiler.spinningBodyInMsg.subscribeTo(prescribed_rotation_msg)
    platform.prescribedRotationInMsg.subscribeTo(one_dof_rotation_profiler.prescribedRotationOutMsg)

    # Create translational motion profiler
    one_dof_translation_profiler = prescribedLinearTranslation.PrescribedLinearTranslation()
    one_dof_translation_profiler.ModelTag = "prescribedLinearTranslation"
    one_dof_translation_profiler.setTransHat_M(trans_axis_m)
    one_dof_translation_profiler.setTransAccelMax(prescribed_trans_accel_max)
    one_dof_translation_profiler.setTransPosInit(prescribed_pos_init)
    one_dof_translation_profiler.setCoastOptionBangDuration(1.0)
    one_dof_translation_profiler.setSmoothingDuration(1.0)
    sim.AddModelToTask(task_name, one_dof_translation_profiler)

    # Create the translational motion reference message
    prescribed_translation_msg_data = messaging.LinearTranslationRigidBodyMsgPayload()
    prescribed_translation_msg_data.rho = prescribed_pos_ref
    prescribed_translation_msg_data.rhoDot = 0.0
    prescribed_translation_msg = messaging.LinearTranslationRigidBodyMsg().write(prescribed_translation_msg_data)
    one_dof_translation_profiler.linearTranslationRigidBodyInMsg.subscribeTo(prescribed_translation_msg)
    platform.prescribedTranslationInMsg.subscribeTo(one_dof_translation_profiler.prescribedTranslationOutMsg)

    # Create the spinning body
    spinning_body_one_dof = spinningBodyOneDOFStateEffector.SpinningBodyOneDOFStateEffector()
    spinning_body_one_dof.ModelTag = "spinningBodyOneDOF"
    spinning_body_one_dof.mass = 50.0
    spinning_body_one_dof.IPntSc_S = [[50.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 40.0]]
    spinning_body_one_dof.dcm_S0B = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    spinning_body_one_dof.r_ScS_S = [[0.0], [0.0], [0.0]]
    spinning_body_one_dof.r_SB_B = [[1.5], [0.0], [0.0]]
    spinning_body_one_dof.sHat_S = [[1], [0], [0]]
    spinning_body_one_dof.thetaInit = 1.0 * macros.D2R
    spinning_body_one_dof.thetaDotInit = 0.0 * macros.D2R
    spinning_body_one_dof.k = 200.0
    spinning_body_one_dof.c = 100.0
    sim.AddModelToTask(task_name, spinning_body_one_dof)

    # Create the torque message
    cmd_torque = 0
    cmd_array = messaging.ArrayMotorTorqueMsgPayload()
    cmd_array.motorTorque = [cmd_torque]  # [Nm]
    cmd_msg = messaging.ArrayMotorTorqueMsg().write(cmd_array)
    spinning_body_one_dof.motorTorqueInMsg.subscribeTo(cmd_msg)

    # Create the locking message
    lock_array = messaging.ArrayEffectorLockMsgPayload()
    lock_flag = [0]
    lock_array.effectorLockFlag = lock_flag
    lock_msg = messaging.ArrayEffectorLockMsg().write(lock_array)
    spinning_body_one_dof.motorLockInMsg.subscribeTo(lock_msg)

    # Create the spinning body reference message
    spinning_body_theta_ref = 0.0
    spinning_body_theta_ref_msg_data = messaging.HingedRigidBodyMsgPayload()
    spinning_body_theta_ref_msg_data.theta = spinning_body_theta_ref
    spinning_body_theta_ref_msg_data.thetaDot = 0.0
    spinning_body_theta_ref_msg = messaging.HingedRigidBodyMsg().write(spinning_body_theta_ref_msg_data)
    spinning_body_one_dof.spinningBodyRefInMsg.subscribeTo(spinning_body_theta_ref_msg)

    # Attach the spinning body to the prescribed motion platform
    # sc_object.addStateEffector(spinning_body_one_dof)
    platform.addStateEffector(spinning_body_one_dof)

    # Add Earth gravity to the simulation
    earth_gravity = gravityEffector.GravBodyData()
    earth_gravity.planetName = "earth_planet_data"
    earth_gravity.mu = 0.3986004415E+15  # meters!
    earth_gravity.isCentralBody = True
    sc_object.gravField.gravBodies = spacecraft.GravBodyVector([earth_gravity])

    # Set up data logging
    conservation_data_log = sc_object.logger(["totOrbAngMomPntN_N", "totRotAngMomPntC_N", "totOrbEnergy", "totRotEnergy"])
    prescribed_rotation_data_log = platform.prescribedRotationOutMsg.recorder()
    prescribed_translation_data_log = platform.prescribedTranslationOutMsg.recorder()
    one_dof_rotation_profiler_data_log = one_dof_rotation_profiler.spinningBodyOutMsg.recorder()
    spinning_body_theta_data_log = spinning_body_one_dof.spinningBodyOutMsg.recorder()
    sim.AddModelToTask(task_name, conservation_data_log)
    sim.AddModelToTask(task_name, prescribed_rotation_data_log)
    sim.AddModelToTask(task_name, prescribed_translation_data_log)
    sim.AddModelToTask(task_name, one_dof_rotation_profiler_data_log)
    sim.AddModelToTask(task_name, spinning_body_theta_data_log)

    # Run the simulation
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(sim_time))
    sim.ExecuteSimulation()

    # Extract the logged variables
    orb_energy = conservation_data_log.totOrbEnergy
    orb_ang_mom_n = conservation_data_log.totOrbAngMomPntN_N
    rot_ang_mom_n = conservation_data_log.totRotAngMomPntC_N
    rot_energy = conservation_data_log.totRotEnergy
    timespan = prescribed_rotation_data_log.times() * macros.NANO2SEC  # [s]
    prescribed_theta = macros.R2D * one_dof_rotation_profiler_data_log.theta  # [deg]
    omega_pm_p = prescribed_rotation_data_log.omega_PM_P * macros.R2D  # [deg]
    omega_prime_pm_p = prescribed_rotation_data_log.omegaPrime_PM_P * macros.R2D  # [deg]
    r_pm_m = prescribed_translation_data_log.r_PM_M
    r_prime_pm_m = prescribed_translation_data_log.rPrime_PM_M
    r_prime_prime_pm_m = prescribed_translation_data_log.rPrimePrime_PM_M
    spinning_body_theta = spinning_body_theta_data_log.theta
    spinning_body_theta_dot = spinning_body_theta_data_log.thetaDot

    # Plot results
    plot_conservation(timespan,
                      orb_ang_mom_n,
                      orb_energy,
                      rot_ang_mom_n,
                      rot_energy)
    plot_prescribed_motion(timespan,
                           prescribed_theta,
                           omega_pm_p,
                           omega_prime_pm_p,
                           r_pm_m,
                           r_prime_pm_m,
                           r_prime_prime_pm_m)

    # Plot spinning body angle
    plt.figure()
    plt.clf()
    plt.plot(timespan, spinning_body_theta, label=r"$\theta$", color="teal")
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Angle (deg)", fontsize=16)
    plt.legend(loc="center right", prop={"size": 16})
    plt.grid(True)

    # Plot spinning body angle rate
    plt.figure()
    plt.clf()
    plt.plot(timespan, spinning_body_theta_dot, label=r"$\dot{\theta}$", color="teal")
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Angle Rate (deg/s)", fontsize=16)
    plt.legend(loc="center right", prop={"size": 16})
    plt.grid(True)

    if show_plots:
        plt.show()
    plt.close("all")

    # Test check
    test_check_verification(orb_ang_mom_n, orb_energy, rot_ang_mom_n)


def test_prescribed_branching_spinning_body_two_dof(show_plots):

    task_name = "unitTask"
    process_name = "TestProcess"
    sim = SimulationBaseClass.SimBaseClass()
    process_rate = macros.sec2nano(0.001)
    process = sim.CreateNewProcess(process_name)
    process.addTask(sim.CreateNewTask(task_name, process_rate))

    # Create the spacecraft
    sc_object = create_spacecraft_hub()
    sim.AddModelToTask(task_name, sc_object)

    # Create prescribed motion object
    platform = create_prescribed_body(r_pm_m_init, sigma_pm_init)
    sim.AddModelToTask(task_name, platform)
    sc_object.addStateEffector(platform)

    # Create rotational motion profiler
    one_dof_rotation_profiler = prescribedRotation1DOF.PrescribedRotation1DOF()
    one_dof_rotation_profiler.ModelTag = "prescribedRotation1DOF"
    one_dof_rotation_profiler.setRotHat_M(rot_axis_m)
    one_dof_rotation_profiler.setThetaDDotMax(prescribed_ang_accel_max)
    one_dof_rotation_profiler.setThetaInit(prescribed_theta_init)
    one_dof_rotation_profiler.setCoastOptionBangDuration(1.0)
    one_dof_rotation_profiler.setSmoothingDuration(1.0)
    sim.AddModelToTask(task_name, one_dof_rotation_profiler)

    # Create the rotational motion reference message
    prescribed_rotation_msg_data = messaging.HingedRigidBodyMsgPayload()
    prescribed_rotation_msg_data.theta = prescribed_theta_ref
    prescribed_rotation_msg_data.thetaDot = 0.0  # [rad/s]
    prescribed_rotation_msg = messaging.HingedRigidBodyMsg().write(prescribed_rotation_msg_data)
    one_dof_rotation_profiler.spinningBodyInMsg.subscribeTo(prescribed_rotation_msg)
    platform.prescribedRotationInMsg.subscribeTo(one_dof_rotation_profiler.prescribedRotationOutMsg)

    # Create translational motion profiler
    one_dof_translation_profiler = prescribedLinearTranslation.PrescribedLinearTranslation()
    one_dof_translation_profiler.ModelTag = "prescribedLinearTranslation"
    one_dof_translation_profiler.setTransHat_M(trans_axis_m)
    one_dof_translation_profiler.setTransAccelMax(prescribed_trans_accel_max)
    one_dof_translation_profiler.setTransPosInit(prescribed_pos_init)
    one_dof_translation_profiler.setCoastOptionBangDuration(1.0)
    one_dof_translation_profiler.setSmoothingDuration(1.0)
    sim.AddModelToTask(task_name, one_dof_translation_profiler)

    # Create the translational motion reference message
    prescribed_translation_msg_data = messaging.LinearTranslationRigidBodyMsgPayload()
    prescribed_translation_msg_data.rho = prescribed_pos_ref
    prescribed_translation_msg_data.rhoDot = 0.0
    prescribed_translation_msg = messaging.LinearTranslationRigidBodyMsg().write(prescribed_translation_msg_data)
    one_dof_translation_profiler.linearTranslationRigidBodyInMsg.subscribeTo(prescribed_translation_msg)
    platform.prescribedTranslationInMsg.subscribeTo(one_dof_translation_profiler.prescribedTranslationOutMsg)

    # Create the spinning body
    spinning_body_two_dof = spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector()
    spinning_body_two_dof.ModelTag = "spinningBodyTwoDOF"
    spinning_body_two_dof.mass1 = 100.0
    spinning_body_two_dof.mass2 = 50.0
    spinning_body_two_dof.IS1PntSc1_S1 = [[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]
    spinning_body_two_dof.IS2PntSc2_S2 = [[50.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 40.0]]
    spinning_body_two_dof.dcm_S10B = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    spinning_body_two_dof.dcm_S20S1 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    spinning_body_two_dof.r_Sc1S1_S1 = [[0.0], [0.0], [0.0]]
    spinning_body_two_dof.r_Sc2S2_S2 = [[0.0], [0.0], [0.0]]
    spinning_body_two_dof.r_S1B_B = [[1.5], [0.0], [0.0]]
    spinning_body_two_dof.r_S2S1_S1 = [[1.0], [0.0], [0.0]]
    spinning_body_two_dof.s1Hat_S1 = [[1], [0], [0]]
    spinning_body_two_dof.s2Hat_S2 = [[1], [0], [0]]
    spinning_body_two_dof.theta1Init = 0 * macros.D2R
    spinning_body_two_dof.theta2Init = 5 * macros.D2R
    spinning_body_two_dof.theta1DotInit = 2.0 * macros.D2R
    spinning_body_two_dof.theta2DotInit = -1.0 * macros.D2R
    spinning_body_two_dof.k1 = 1000.0
    spinning_body_two_dof.k2 = 500.0
    sim.AddModelToTask(task_name, spinning_body_two_dof)

    # Create the torque message
    cmd_torque_1 = 0
    cmd_torque_2 = 0
    cmd_array = messaging.ArrayMotorTorqueMsgPayload()
    cmd_array.motorTorque = [cmd_torque_1, cmd_torque_2]  # [Nm]
    cmd_msg = messaging.ArrayMotorTorqueMsg().write(cmd_array)
    spinning_body_two_dof.motorTorqueInMsg.subscribeTo(cmd_msg)

    # Create the locking message
    lock_array = messaging.ArrayEffectorLockMsgPayload()
    lock_flag = [0, 0]
    lock_array.effectorLockFlag = lock_flag
    lock_msg = messaging.ArrayEffectorLockMsg().write(lock_array)
    spinning_body_two_dof.motorLockInMsg.subscribeTo(lock_msg)

    # Create the reference messages
    spinning_body_theta_1_ref = 0.0
    spinning_body_theta_1_ref_msg_data = messaging.HingedRigidBodyMsgPayload()
    spinning_body_theta_1_ref_msg_data.theta = spinning_body_theta_1_ref
    spinning_body_theta_1_ref_msg_data.thetaDot = 0.0
    spinning_body_theta_1_ref_msg = messaging.HingedRigidBodyMsg().write(spinning_body_theta_1_ref_msg_data)
    spinning_body_two_dof.spinningBodyRefInMsgs[0].subscribeTo(spinning_body_theta_1_ref_msg)

    spinning_body_theta_2_ref = 0.0
    spinning_body_theta_2_ref_msg_data = messaging.HingedRigidBodyMsgPayload()
    spinning_body_theta_2_ref_msg_data.theta = spinning_body_theta_2_ref
    spinning_body_theta_2_ref_msg_data.thetaDot = 0.0
    spinning_body_theta_2_ref_msg = messaging.HingedRigidBodyMsg().write(spinning_body_theta_2_ref_msg_data)
    spinning_body_two_dof.spinningBodyRefInMsgs[1].subscribeTo(spinning_body_theta_2_ref_msg)

    # Attach the spinning body to the prescribed motion platform
    # sc_object.addStateEffector(spinning_body_two_dof)
    platform.addStateEffector(spinning_body_two_dof)

    # Add Earth gravity to the simulation
    earth_gravity = gravityEffector.GravBodyData()
    earth_gravity.planetName = "earth_planet_data"
    earth_gravity.mu = 0.3986004415E+15  # meters!
    earth_gravity.isCentralBody = True
    sc_object.gravField.gravBodies = spacecraft.GravBodyVector([earth_gravity])

    # Set up data logging
    conservation_data_log = sc_object.logger(["totOrbAngMomPntN_N", "totRotAngMomPntC_N", "totOrbEnergy", "totRotEnergy"])
    prescribed_rotation_data_log = platform.prescribedRotationOutMsg.recorder()
    prescribed_translation_data_log = platform.prescribedTranslationOutMsg.recorder()
    one_dof_rotation_profiler_data_log = one_dof_rotation_profiler.spinningBodyOutMsg.recorder()
    spinning_body_theta_1_data_log = spinning_body_two_dof.spinningBodyOutMsgs[0].recorder()
    spinning_body_theta_2_data_log = spinning_body_two_dof.spinningBodyOutMsgs[1].recorder()
    sim.AddModelToTask(task_name, conservation_data_log)
    sim.AddModelToTask(task_name, prescribed_rotation_data_log)
    sim.AddModelToTask(task_name, prescribed_translation_data_log)
    sim.AddModelToTask(task_name, one_dof_rotation_profiler_data_log)
    sim.AddModelToTask(task_name, spinning_body_theta_1_data_log)
    sim.AddModelToTask(task_name, spinning_body_theta_2_data_log)

    # Run the simulation
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(sim_time))
    sim.ExecuteSimulation()

    # Extract the logged variables
    orb_energy = conservation_data_log.totOrbEnergy
    orb_ang_mom_n = conservation_data_log.totOrbAngMomPntN_N
    rot_ang_mom_n = conservation_data_log.totRotAngMomPntC_N
    rot_energy = conservation_data_log.totRotEnergy
    timespan = prescribed_rotation_data_log.times() * macros.NANO2SEC  # [s]
    prescribed_theta = macros.R2D * one_dof_rotation_profiler_data_log.theta  # [deg]
    omega_pm_p = prescribed_rotation_data_log.omega_PM_P * macros.R2D  # [deg]
    omega_prime_pm_p = prescribed_rotation_data_log.omegaPrime_PM_P * macros.R2D  # [deg]
    r_pm_m = prescribed_translation_data_log.r_PM_M
    r_prime_pm_m = prescribed_translation_data_log.rPrime_PM_M
    r_prime_prime_pm_m = prescribed_translation_data_log.rPrimePrime_PM_M
    spinning_body_theta_1 = spinning_body_theta_1_data_log.theta
    spinning_body_theta_1_dot = spinning_body_theta_1_data_log.thetaDot
    spinning_body_theta_2 = spinning_body_theta_2_data_log.theta
    spinning_body_theta_2_dot = spinning_body_theta_2_data_log.thetaDot

    # Plot results
    plot_conservation(timespan,
                      orb_ang_mom_n,
                      orb_energy,
                      rot_ang_mom_n,
                      rot_energy)
    plot_prescribed_motion(timespan,
                           prescribed_theta,
                           omega_pm_p,
                           omega_prime_pm_p,
                           r_pm_m,
                           r_prime_pm_m,
                           r_prime_prime_pm_m)

    # Plot spinning body theta 1
    plt.figure()
    plt.clf()
    plt.plot(timespan, spinning_body_theta_1, label=r"$\theta_1$", color="teal")
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Angle (deg)", fontsize=16)
    plt.legend(loc="center right", prop={"size": 16})
    plt.grid(True)

    # Plot spinning body theta dot 1
    plt.figure()
    plt.clf()
    plt.plot(timespan, spinning_body_theta_1_dot, label=r"$\dot{\theta}_1$", color="teal")
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Angle Rate (deg/s)", fontsize=16)
    plt.legend(loc="center right", prop={"size": 16})
    plt.grid(True)

    # Plot spinning body theta 2
    plt.figure()
    plt.clf()
    plt.plot(timespan, spinning_body_theta_2, label=r"$\theta_2$", color="teal")
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Angle (deg)", fontsize=16)
    plt.legend(loc="center right", prop={"size": 16})
    plt.grid(True)

    # Plot spinning body theta dot 2
    plt.figure()
    plt.clf()
    plt.plot(timespan, spinning_body_theta_2_dot, label=r"$\dot{\theta}_2$", color="teal")
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Angle Rate (deg/s)", fontsize=16)
    plt.legend(loc="center right", prop={"size": 16})
    plt.grid(True)

    if show_plots:
        plt.show()
    plt.close("all")

    # Test check
    test_check_verification(orb_ang_mom_n, orb_energy, rot_ang_mom_n)


def test_prescribed_branching_linear_translation_one_dof(show_plots):

    task_name = "unitTask"
    process_name = "TestProcess"
    sim = SimulationBaseClass.SimBaseClass()
    process_rate = macros.sec2nano(0.001)
    process = sim.CreateNewProcess(process_name)
    process.addTask(sim.CreateNewTask(task_name, process_rate))

    # Create the spacecraft
    sc_object = create_spacecraft_hub()
    sim.AddModelToTask(task_name, sc_object)

    # Create prescribed motion object
    platform = create_prescribed_body(r_pm_m_init, sigma_pm_init)
    sim.AddModelToTask(task_name, platform)
    sc_object.addStateEffector(platform)

    # Create rotational motion profiler
    one_dof_rotation_profiler = prescribedRotation1DOF.PrescribedRotation1DOF()
    one_dof_rotation_profiler.ModelTag = "prescribedRotation1DOF"
    one_dof_rotation_profiler.setRotHat_M(rot_axis_m)
    one_dof_rotation_profiler.setThetaDDotMax(prescribed_ang_accel_max)
    one_dof_rotation_profiler.setThetaInit(prescribed_theta_init)
    one_dof_rotation_profiler.setCoastOptionBangDuration(1.0)
    one_dof_rotation_profiler.setSmoothingDuration(1.0)
    sim.AddModelToTask(task_name, one_dof_rotation_profiler)

    # Create the rotational motion reference message
    prescribed_rotation_msg_data = messaging.HingedRigidBodyMsgPayload()
    prescribed_rotation_msg_data.theta = prescribed_theta_ref
    prescribed_rotation_msg_data.thetaDot = 0.0  # [rad/s]
    prescribed_rotation_msg = messaging.HingedRigidBodyMsg().write(prescribed_rotation_msg_data)
    one_dof_rotation_profiler.spinningBodyInMsg.subscribeTo(prescribed_rotation_msg)
    platform.prescribedRotationInMsg.subscribeTo(one_dof_rotation_profiler.prescribedRotationOutMsg)

    # Create translational motion profiler
    one_dof_translation_profiler = prescribedLinearTranslation.PrescribedLinearTranslation()
    one_dof_translation_profiler.ModelTag = "prescribedLinearTranslation"
    one_dof_translation_profiler.setTransHat_M(trans_axis_m)
    one_dof_translation_profiler.setTransAccelMax(prescribed_trans_accel_max)
    one_dof_translation_profiler.setTransPosInit(prescribed_pos_init)
    one_dof_translation_profiler.setCoastOptionBangDuration(1.0)
    one_dof_translation_profiler.setSmoothingDuration(1.0)
    sim.AddModelToTask(task_name, one_dof_translation_profiler)

    # Create the translational motion reference message
    prescribed_translation_msg_data = messaging.LinearTranslationRigidBodyMsgPayload()
    prescribed_translation_msg_data.rho = prescribed_pos_ref
    prescribed_translation_msg_data.rhoDot = 0.0
    prescribed_translation_msg = messaging.LinearTranslationRigidBodyMsg().write(prescribed_translation_msg_data)
    one_dof_translation_profiler.linearTranslationRigidBodyInMsg.subscribeTo(prescribed_translation_msg)
    platform.prescribedTranslationInMsg.subscribeTo(one_dof_translation_profiler.prescribedTranslationOutMsg)

    # Create the translating body
    translating_body_one_dof = linearTranslationOneDOFStateEffector.linearTranslationOneDOFStateEffector()
    translating_body_one_dof.ModelTag = "translatingBody"
    translating_body_rho_init = 0.0
    translating_body_rho_dot_init = 0.0
    fHat_B = [[1.0], [0.0], [0.0]]
    r_FcF_F = [[0.0], [0.0], [0.0]]
    r_F0B_B = [[1.0], [0.0], [0.0]]
    IPntFc_F = [[50.0, 0.0, 0.0],
                [0.0, 80.0, 0.0],
                [0.0, 0.0, 60.0]]
    dcm_FB = [[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]]
    translating_body_one_dof.setMass(20.0)
    translating_body_one_dof.setK(100.0)
    translating_body_one_dof.setC(25.0)
    translating_body_one_dof.setRhoInit(translating_body_rho_init)
    translating_body_one_dof.setRhoDotInit(translating_body_rho_dot_init)
    translating_body_one_dof.setFHat_B(fHat_B)
    translating_body_one_dof.setR_FcF_F(r_FcF_F)
    translating_body_one_dof.setR_F0B_B(r_F0B_B)
    translating_body_one_dof.setIPntFc_F(IPntFc_F)
    translating_body_one_dof.setDCM_FB(dcm_FB)
    sim.AddModelToTask(task_name, translating_body_one_dof)

    # Attach the translating body to the prescribed motion platform
    # sc_object.addStateEffector(translating_body_one_dof)
    platform.addStateEffector(translating_body_one_dof)

    # Add Earth gravity to the simulation
    earth_gravity = gravityEffector.GravBodyData()
    earth_gravity.planetName = "earth_planet_data"
    earth_gravity.mu = 0.3986004415E+15  # meters!
    earth_gravity.isCentralBody = True
    sc_object.gravField.gravBodies = spacecraft.GravBodyVector([earth_gravity])

    # Set up data logging
    conservation_data_log = sc_object.logger(["totOrbAngMomPntN_N", "totRotAngMomPntC_N", "totOrbEnergy", "totRotEnergy"])
    prescribed_rotation_data_log = platform.prescribedRotationOutMsg.recorder()
    prescribed_translation_data_log = platform.prescribedTranslationOutMsg.recorder()
    one_dof_rotation_profiler_data_log = one_dof_rotation_profiler.spinningBodyOutMsg.recorder()
    translating_body_data_log = translating_body_one_dof.translatingBodyOutMsg.recorder()
    sim.AddModelToTask(task_name, conservation_data_log)
    sim.AddModelToTask(task_name, prescribed_rotation_data_log)
    sim.AddModelToTask(task_name, prescribed_translation_data_log)
    sim.AddModelToTask(task_name, one_dof_rotation_profiler_data_log)
    sim.AddModelToTask(task_name, translating_body_data_log)

    # Run the simulation
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(sim_time))
    sim.ExecuteSimulation()

    # Extract the logged variables
    orb_energy = conservation_data_log.totOrbEnergy
    orb_ang_mom_n = conservation_data_log.totOrbAngMomPntN_N
    rot_ang_mom_n = conservation_data_log.totRotAngMomPntC_N
    rot_energy = conservation_data_log.totRotEnergy
    timespan = prescribed_rotation_data_log.times() * macros.NANO2SEC  # [s]
    prescribed_theta = macros.R2D * one_dof_rotation_profiler_data_log.theta  # [deg]
    omega_pm_p = prescribed_rotation_data_log.omega_PM_P * macros.R2D  # [deg]
    omega_prime_pm_p = prescribed_rotation_data_log.omegaPrime_PM_P * macros.R2D  # [deg]
    r_pm_m = prescribed_translation_data_log.r_PM_M
    r_prime_pm_m = prescribed_translation_data_log.rPrime_PM_M
    r_prime_prime_pm_m = prescribed_translation_data_log.rPrimePrime_PM_M
    translating_body_rho = translating_body_data_log.rho  # [m]
    translating_body_rho_dot = translating_body_data_log.rhoDot  # [m/s]

    # Plot results
    plot_conservation(timespan,
                      orb_ang_mom_n,
                      orb_energy,
                      rot_ang_mom_n,
                      rot_energy)
    plot_prescribed_motion(timespan,
                           prescribed_theta,
                           omega_pm_p,
                           omega_prime_pm_p,
                           r_pm_m,
                           r_prime_pm_m,
                           r_prime_prime_pm_m)

    # Plot translating body displacement
    plt.figure()
    plt.clf()
    plt.plot(timespan, translating_body_rho, label=r"$\rho$", color="teal")
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Displacement (m)", fontsize=16)
    plt.legend(loc="upper right", prop={"size": 16})
    plt.grid(True)

    # Plot translating body displacement rate
    plt.figure()
    plt.clf()
    plt.plot(timespan, translating_body_rho_dot, label=r"$\dot{\rho}$", color="teal")
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Displacement Rate (m/s)", fontsize=16)
    plt.legend(loc="upper right", prop={"size": 16})
    plt.grid(True)

    if show_plots:
        plt.show()
    plt.close("all")

    # Test check
    test_check_verification(orb_ang_mom_n, orb_energy, rot_ang_mom_n)


def test_prescribed_branching_linear_mass_particle(show_plots):

    task_name = "unitTask"
    process_name = "TestProcess"
    sim = SimulationBaseClass.SimBaseClass()
    process_rate = macros.sec2nano(0.001)
    process = sim.CreateNewProcess(process_name)
    process.addTask(sim.CreateNewTask(task_name, process_rate))

    # Create the spacecraft
    sc_object = create_spacecraft_hub()
    sim.AddModelToTask(task_name, sc_object)

    # Create prescribed motion object
    platform = create_prescribed_body(r_pm_m_init, sigma_pm_init)
    sim.AddModelToTask(task_name, platform)
    sc_object.addStateEffector(platform)

    # Create rotational motion profiler
    one_dof_rotation_profiler = prescribedRotation1DOF.PrescribedRotation1DOF()
    one_dof_rotation_profiler.ModelTag = "prescribedRotation1DOF"
    one_dof_rotation_profiler.setRotHat_M(rot_axis_m)
    one_dof_rotation_profiler.setThetaDDotMax(prescribed_ang_accel_max)
    one_dof_rotation_profiler.setThetaInit(prescribed_theta_init)
    one_dof_rotation_profiler.setCoastOptionBangDuration(1.0)
    one_dof_rotation_profiler.setSmoothingDuration(1.0)
    sim.AddModelToTask(task_name, one_dof_rotation_profiler)

    # Create the rotational motion reference message
    prescribed_rotation_msg_data = messaging.HingedRigidBodyMsgPayload()
    prescribed_rotation_msg_data.theta = prescribed_theta_ref
    prescribed_rotation_msg_data.thetaDot = 0.0  # [rad/s]
    prescribed_rotation_msg = messaging.HingedRigidBodyMsg().write(prescribed_rotation_msg_data)
    one_dof_rotation_profiler.spinningBodyInMsg.subscribeTo(prescribed_rotation_msg)
    platform.prescribedRotationInMsg.subscribeTo(one_dof_rotation_profiler.prescribedRotationOutMsg)

    # Create translational motion profiler
    one_dof_translation_profiler = prescribedLinearTranslation.PrescribedLinearTranslation()
    one_dof_translation_profiler.ModelTag = "prescribedLinearTranslation"
    one_dof_translation_profiler.setTransHat_M(trans_axis_m)
    one_dof_translation_profiler.setTransAccelMax(prescribed_trans_accel_max)
    one_dof_translation_profiler.setTransPosInit(prescribed_pos_init)
    one_dof_translation_profiler.setCoastOptionBangDuration(1.0)
    one_dof_translation_profiler.setSmoothingDuration(1.0)
    sim.AddModelToTask(task_name, one_dof_translation_profiler)

    # Create the translational motion reference message
    prescribed_translation_msg_data = messaging.LinearTranslationRigidBodyMsgPayload()
    prescribed_translation_msg_data.rho = prescribed_pos_ref
    prescribed_translation_msg_data.rhoDot = 0.0
    prescribed_translation_msg = messaging.LinearTranslationRigidBodyMsg().write(prescribed_translation_msg_data)
    one_dof_translation_profiler.linearTranslationRigidBodyInMsg.subscribeTo(prescribed_translation_msg)
    platform.prescribedTranslationInMsg.subscribeTo(one_dof_translation_profiler.prescribedTranslationOutMsg)

    # Create liner mass particle
    linear_mass_particle = linearSpringMassDamper.LinearSpringMassDamper()
    linear_mass_particle.k = 100.0
    linear_mass_particle.c = 0.0
    linear_mass_particle.r_PB_B = [[0.1], [0], [-0.1]]
    linear_mass_particle.pHat_B = [[np.sqrt(3)/3], [np.sqrt(3)/3], [np.sqrt(3)/3]]
    linear_mass_particle.rhoInit = 0.05
    linear_mass_particle.rhoDotInit = 0.0
    linear_mass_particle.massInit = 10.0
    sim.AddModelToTask(task_name, linear_mass_particle)

    # Attach the particle to the prescribed motion platform
    # sc_object.addStateEffector(linear_mass_particle)
    platform.addStateEffector(linear_mass_particle)

    # Add Earth gravity to the simulation
    earth_gravity = gravityEffector.GravBodyData()
    earth_gravity.planetName = "earth_planet_data"
    earth_gravity.mu = 0.3986004415E+15  # meters!
    earth_gravity.isCentralBody = True
    sc_object.gravField.gravBodies = spacecraft.GravBodyVector([earth_gravity])

    # Set up data logging
    conservation_data_log = sc_object.logger(["totOrbAngMomPntN_N", "totRotAngMomPntC_N", "totOrbEnergy", "totRotEnergy"])
    prescribed_rotation_data_log = platform.prescribedRotationOutMsg.recorder()
    prescribed_translation_data_log = platform.prescribedTranslationOutMsg.recorder()
    one_dof_rotation_profiler_data_log = one_dof_rotation_profiler.spinningBodyOutMsg.recorder()
    sim.AddModelToTask(task_name, conservation_data_log)
    sim.AddModelToTask(task_name, prescribed_rotation_data_log)
    sim.AddModelToTask(task_name, prescribed_translation_data_log)
    sim.AddModelToTask(task_name, one_dof_rotation_profiler_data_log)

    # Run the simulation
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(sim_time))
    sim.ExecuteSimulation()

    # Extract the logged variables
    orb_energy = conservation_data_log.totOrbEnergy
    orb_ang_mom_n = conservation_data_log.totOrbAngMomPntN_N
    rot_ang_mom_n = conservation_data_log.totRotAngMomPntC_N
    rot_energy = conservation_data_log.totRotEnergy
    timespan = prescribed_rotation_data_log.times() * macros.NANO2SEC  # [s]
    prescribed_theta = macros.R2D * one_dof_rotation_profiler_data_log.theta  # [deg]
    omega_pm_p = prescribed_rotation_data_log.omega_PM_P * macros.R2D  # [deg]
    omega_prime_pm_p = prescribed_rotation_data_log.omegaPrime_PM_P * macros.R2D  # [deg]
    r_pm_m = prescribed_translation_data_log.r_PM_M
    r_prime_pm_m = prescribed_translation_data_log.rPrime_PM_M
    r_prime_prime_pm_m = prescribed_translation_data_log.rPrimePrime_PM_M

    # Plot results
    plot_conservation(timespan,
                      orb_ang_mom_n,
                      orb_energy,
                      rot_ang_mom_n,
                      rot_energy)
    plot_prescribed_motion(timespan,
                           prescribed_theta,
                           omega_pm_p,
                           omega_prime_pm_p,
                           r_pm_m,
                           r_prime_pm_m,
                           r_prime_prime_pm_m)

    if show_plots:
        plt.show()
    plt.close("all")

    # Test check
    test_check_verification(orb_ang_mom_n, orb_energy, rot_ang_mom_n)


def create_spacecraft_hub():

    mass_hub = 800  # [kg]
    length_hub = 1.0  # [m]
    width_hub = 1.0  # [m]
    depth_hub = 1.0  # [m]
    I_hub_11 = (1 / 12) * mass_hub * (length_hub * length_hub + depth_hub * depth_hub)  # [kg m^2]
    I_hub_22 = (1 / 12) * mass_hub * (length_hub * length_hub + width_hub * width_hub)  # [kg m^2]
    I_hub_33 = (1 / 12) * mass_hub * (width_hub * width_hub + depth_hub * depth_hub)  # [kg m^2]

    sc_object = spacecraft.Spacecraft()
    sc_object.ModelTag = "sc_object"
    sc_object.hub.mHub = mass_hub  # kg
    sc_object.hub.r_BcB_B = [0.0, 0.0, 0.0]  # [m]
    sc_object.hub.IHubPntBc_B = [[I_hub_11, 0.0, 0.0], [0.0, I_hub_22, 0.0], [0.0, 0.0, I_hub_33]]  # [kg m^2] (Hub approximated as a cube)
    sc_object.hub.r_CN_NInit = [[-4020338.690396649], [7490566.741852513], [5248299.211589362]]
    sc_object.hub.v_CN_NInit = [[-5199.77710904224], [-3436.681645356935], [1041.576797498721]]
    sc_object.hub.omega_BN_BInit = [[0.01], [-0.01], [0.01]]
    sc_object.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]

    return sc_object


def create_prescribed_body(r_PM_M, sigma_PM):
    mass_prescribed = 100  # [kg]
    length_prescribed = 1.0  # [m]
    width_prescribed = 1.0  # [m]
    depth_prescribed = 1.0  # [m]
    I_prescribed_11 = (1 / 12) * mass_prescribed * (length_prescribed * length_prescribed + depth_prescribed * depth_prescribed)  # [kg m^2]
    I_prescribed_22 = (1 / 12) * mass_prescribed * (length_prescribed * length_prescribed + width_prescribed * width_prescribed)  # [kg m^2]
    I_prescribed_33 = (1 / 12) * mass_prescribed * (width_prescribed * width_prescribed + depth_prescribed * depth_prescribed)  # [kg m^2]
    I_platform_pc_p = [[I_prescribed_11, 0.0, 0.0], [0.0, I_prescribed_22, 0.0], [0.0, 0.0,I_prescribed_33]]  # [kg m^2] (approximated as a cube)

    platform = prescribedMotionStateEffector.PrescribedMotionStateEffector()
    platform.ModelTag = "platform"
    platform.mass = mass_prescribed
    platform.IPntPc_P = I_platform_pc_p
    platform.r_MB_B = [0.5, 0.0, 0.0]
    platform.r_PcP_P = [0.5, 0.0, 0.0]
    platform.r_PM_M = r_PM_M
    platform.rPrime_PM_M = np.array([0.0, 0.0, 0.0])
    platform.rPrimePrime_PM_M = np.array([0.0, 0.0, 0.0])
    platform.omega_PM_P = np.array([0.0, 0.0, 0.0])
    platform.omegaPrime_PM_P = np.array([0.0, 0.0, 0.0])
    platform.sigma_PM = sigma_PM
    platform.omega_MB_B = [0.0, 0.0, 0.0]
    platform.omegaPrime_MB_B = [0.0, 0.0, 0.0]
    platform.sigma_MB = [0.0, 0.0, 0.0]

    return platform


def plot_prescribed_motion(timespan,
                           prescribed_theta,
                           omega_pm_p,
                           omega_prime_pm_p,
                           r_pm_m,
                           r_prime_pm_m,
                           r_prime_prime_pm_m):

    # Plot prescribed position and angle
    prescribed_theta_ref_plotting = np.ones(len(timespan)) * prescribed_theta_ref * macros.R2D  # [deg]
    prescribed_rho_ref_plotting = np.ones(len(timespan)) * prescribed_pos_ref  # [m]
    prescribed_rho = np.dot(r_pm_m, trans_axis_m)  # [m]

    fig1, ax1 = plt.subplots()
    ax1.plot(timespan, prescribed_theta, label=r"$\theta$", color="teal")
    ax1.plot(timespan, prescribed_theta_ref_plotting, "--", label=r"$\theta_{\text{ref}}$", color="teal")
    ax1.tick_params(axis="y", labelcolor="teal")
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("Angle (deg)", color="teal", fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(timespan, 100.0 * prescribed_rho, label=r"$\rho$", color="darkviolet")
    ax2.plot(timespan, 100.0 * prescribed_rho_ref_plotting, "--", label=r"$\rho_{\text{ref}}$", color="darkviolet")
    ax2.set_ylabel("Displacement (cm)", color="darkviolet", fontsize=16)
    ax2.tick_params(axis="y", labelcolor="darkviolet")
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    handles = handles_ax1 + handles_ax2
    labels = labels_ax1 + labels_ax2
    plt.legend(handles=handles, labels=labels, loc="center left", prop={"size": 16})
    plt.grid(True)

    # Plot prescribed velocities
    prescribed_rho_dot = np.dot(r_prime_pm_m, trans_axis_m)  # [m/s]
    prescribed_theta_dot = np.dot(omega_pm_p, rot_axis_m)  # [deg/s]

    fig2, ax1 = plt.subplots()
    ax1.plot(timespan, prescribed_theta_dot, label=r"$\dot{\theta}$", color="teal")
    ax1.tick_params(axis="y", labelcolor="teal")
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("Angle Rate (deg/s)", color="teal", fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(timespan, 100.0 * prescribed_rho_dot, label=r"$\dot{\rho}$", color="darkviolet")
    ax2.set_ylabel("Displacement Rate (cm/s)", color="darkviolet", fontsize=16)
    ax2.tick_params(axis="y", labelcolor="darkviolet")
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    handles = handles_ax1 + handles_ax2
    labels = labels_ax1 + labels_ax2
    plt.legend(handles=handles, labels=labels, loc="center", prop={"size": 16})
    plt.grid(True)

    # Plot prescribed accelerations
    prescribed_rho_ddot = np.dot(r_prime_prime_pm_m, trans_axis_m)
    prescribed_theta_ddot = np.dot(omega_prime_pm_p, rot_axis_m)  # [deg/s^2]

    fig3, ax1 = plt.subplots()
    ax1.plot(timespan, prescribed_theta_ddot, label=r"$\ddot{\theta}$", color="teal")
    ax1.tick_params(axis="y", labelcolor="teal")
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("Angular Acceleration (deg/$s^2$)", color="teal", fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(timespan, 100.0 * prescribed_rho_ddot, label=r"$\ddot{\rho}$", color="darkviolet")
    ax2.set_ylabel("Linear Acceleration (cm/$s^2$)", color="darkviolet", fontsize=16)
    ax2.tick_params(axis="y", labelcolor="darkviolet")
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    handles = handles_ax1 + handles_ax2
    labels = labels_ax1 + labels_ax2
    plt.legend(handles=handles, labels=labels, loc="upper right", prop={"size": 16})
    plt.grid(True)


def plot_conservation(timespan, orb_ang_mom_n, orb_energy, rot_ang_mom_n, rot_energy):

    # Plot orbital angular momentum relative difference
    plt.figure()
    plt.clf()
    plt.plot(timespan, (orb_ang_mom_n[:, 0] - orb_ang_mom_n[0, 0]) / orb_ang_mom_n[0, 0], color="teal", label=r'$\hat{n}_1$')
    plt.plot(timespan, (orb_ang_mom_n[:, 1] - orb_ang_mom_n[0, 1]) / orb_ang_mom_n[0, 1], color="darkviolet", label=r'$\hat{n}_2$')
    plt.plot(timespan, (orb_ang_mom_n[:, 2] - orb_ang_mom_n[0, 2]) / orb_ang_mom_n[0, 2], color="blue", label=r'$\hat{n}_3$')
    plt.title('Orbital Angular Momentum Relative Difference', fontsize=16)
    plt.ylabel('Relative Difference (Nms)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='lower right', prop={'size': 16})
    plt.grid(True)

    # Plot orbital energy relative difference
    plt.figure()
    plt.clf()
    plt.plot(timespan, (orb_energy - orb_energy[0]) / orb_energy[0], color="teal")
    plt.title('Orbital Energy Relative Difference', fontsize=16)
    plt.ylabel('Relative Difference (J)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.grid(True)

    # Plot sc angular momentum relative difference
    plt.figure()
    plt.clf()
    plt.plot(timespan, (rot_ang_mom_n[:, 0] - rot_ang_mom_n[0, 0]) / rot_ang_mom_n[0, 0], color="teal", label=r'$\hat{n}_1$')
    plt.plot(timespan, (rot_ang_mom_n[:, 1] - rot_ang_mom_n[0, 1]) / rot_ang_mom_n[0, 1], color="darkviolet", label=r'$\hat{n}_2$')
    plt.plot(timespan, (rot_ang_mom_n[:, 2] - rot_ang_mom_n[0, 2]) / rot_ang_mom_n[0, 2], color="blue", label=r'$\hat{n}_3$')
    plt.title('Rotational Angular Momentum Relative Difference', fontsize=16)
    plt.ylabel('Relative Difference (Nms)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='upper right', prop={'size': 16})
    plt.grid(True)

    # Plot sc energy difference
    plt.figure()
    plt.clf()
    plt.plot(timespan, (rot_energy - rot_energy[0]), color="teal")
    plt.title('Rotational Energy Difference', fontsize=16)
    plt.ylabel('Difference (J)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.grid(True)


def test_check_verification(orb_ang_mom_n, orb_energy, rot_ang_mom_n):

    # Check conservation of orbital angular momentum, energy, and sc rotational angular momentum
    np.testing.assert_allclose(orb_ang_mom_n[0], orb_ang_mom_n[-1], rtol=1e-6, verbose=True)
    np.testing.assert_allclose(orb_energy[0], orb_energy[-1], rtol=1e-6, verbose=True)
    np.testing.assert_allclose(rot_ang_mom_n[0], rot_ang_mom_n[-1], rtol=1e-6, verbose=True)


if __name__ == "__main__":

    test_prescribed_branching_spinning_body_one_dof(True)
    test_prescribed_branching_spinning_body_two_dof(True)
    test_prescribed_branching_linear_translation_one_dof(True)
    test_prescribed_branching_linear_mass_particle(True)
