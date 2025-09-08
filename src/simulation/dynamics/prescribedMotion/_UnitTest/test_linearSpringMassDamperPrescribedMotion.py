
# ISC License
#
# Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
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
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
from Basilisk.simulation import spacecraft
from Basilisk.simulation import linearSpringMassDamper
from Basilisk.simulation import prescribedMotionStateEffector
from Basilisk.simulation import prescribedLinearTranslation
from Basilisk.simulation import prescribedRotation1DOF
from Basilisk.simulation import gravityEffector
from Basilisk.utilities import macros
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.architecture import messaging

@pytest.mark.parametrize("useFlag, testCase", [
    (False,'NoGravity'),
    (False,'Gravity'),
    (False,'Damping'),
    (False,'MassDepletion')
])

def test_fuelSlosh(show_plots, useFlag, testCase):
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"
    unitTestSim = SimulationBaseClass.SimBaseClass()
    testProcessRate = macros.sec2nano(0.001)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Create the spacecraft module
    massHub = 800  # [kg]
    lengthHub = 1.0  # [m]
    widthHub = 1.0  # [m]
    depthHub = 1.0  # [m]
    IHub_11 = (1 / 12) * massHub * (lengthHub * lengthHub + depthHub * depthHub)  # [kg m^2]
    IHub_22 = (1 / 12) * massHub * (lengthHub * lengthHub + widthHub * widthHub)  # [kg m^2]
    IHub_33 = (1 / 12) * massHub * (widthHub * widthHub + depthHub * depthHub)  # [kg m^2]

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "scObject"
    scObject.hub.mHub = massHub  # kg
    scObject.hub.r_BcB_B = [0.0, 0.0, 0.0]  # [m]
    scObject.hub.IHubPntBc_B = [[IHub_11, 0.0, 0.0], [0.0, IHub_22, 0.0], [0.0, 0.0, IHub_33]]  # [kg m^2] (Hub approximated as a cube)
    scObject.hub.r_CN_NInit = [[-4020338.690396649], [7490566.741852513], [5248299.211589362]]
    scObject.hub.v_CN_NInit = [[-5199.77710904224], [-3436.681645356935], [1041.576797498721]]
    scObject.hub.omega_BN_BInit = [[0.1], [-0.1], [0.1]]
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    unitTestSim.AddModelToTask(unitTaskName, scObject)

    # Define variables for particle 1
    particle1 = linearSpringMassDamper.LinearSpringMassDamper()
    particle1.k = 100.0
    particle1.c = 0.0
    particle1.r_PB_B = [[0.1], [0], [-0.1]]
    particle1.pHat_B = [[np.sqrt(3)/3], [np.sqrt(3)/3], [np.sqrt(3)/3]]
    particle1.rhoInit = 0.05
    particle1.rhoDotInit = 0.0
    particle1.massInit = 10.0
    unitTestSim.AddModelToTask(unitTaskName, particle1)

    # Define variables for particle 2
    particle2 = linearSpringMassDamper.LinearSpringMassDamper()
    particle2.k = 100.0
    particle2.c = 17.0
    particle2.r_PB_B = [[0], [0], [0.1]]
    particle2.pHat_B = [[np.sqrt(3)/3], [-np.sqrt(3)/3], [-np.sqrt(3)/3]]
    particle2.rhoInit = -0.025
    particle2.rhoDotInit = 0.0
    particle2.massInit = 20.0
    # unitTestSim.AddModelToTask(unitTaskName, particle2)

    # Define variables for particle 3
    particle3 = linearSpringMassDamper.LinearSpringMassDamper()
    particle3.k = 100.0
    particle3.c = 11.0
    particle3.r_PB_B = [[-0.1], [0], [0.1]]
    particle3.pHat_B = [[-np.sqrt(3)/3], [-np.sqrt(3)/3], [np.sqrt(3)/3]]
    particle3.rhoInit = -0.015
    particle3.rhoDotInit = 0.0
    particle3.massInit = 15.0
    # unitTestSim.AddModelToTask(unitTaskName, particle3)

    # Prescribed motion parameters
    posInit = 0.0
    thetaInit = 0.0
    transAxis_M = np.array([1.0, 0.0, 0.0])
    rotAxis_M = np.array([1.0, 0.0, 0.0])
    prvInit_PM = thetaInit * rotAxis_M
    sigma_PM = rbk.PRV2MRP(prvInit_PM)

    massPlatform = 100  # [kg]
    lengthPlatform = 1.0  # [m]
    widthPlatform = 1.0  # [m]
    depthPlatform = 1.0  # [m]
    IPlatform_11 = (1 / 12) * massPlatform * (lengthPlatform * lengthPlatform + depthPlatform * depthPlatform)  # [kg m^2]
    IPlatform_22 = (1 / 12) * massPlatform * (lengthPlatform * lengthPlatform + widthPlatform * widthPlatform)  # [kg m^2]
    IPlatform_33 = (1 / 12) * massPlatform * (widthPlatform * widthPlatform + depthPlatform * depthPlatform)  # [kg m^2]
    IPlatform_Pc_P = [[IPlatform_11, 0.0, 0.0], [0.0, IPlatform_22, 0.0], [0.0, 0.0,IPlatform_33]]  # [kg m^2] (approximated as a cube)

    # Create prescribed motion object
    platform = prescribedMotionStateEffector.PrescribedMotionStateEffector()
    platform.ModelTag = "platform"
    platform.mass = massPlatform
    platform.IPntPc_P = IPlatform_Pc_P
    platform.r_MB_B = [0.5, 0.0, 0.0]
    platform.r_PcP_P = [0.5, 0.0, 0.0]
    platform.r_PM_M = [0.0, 0.0, 0.0]
    platform.rPrime_PM_M = np.array([0.0, 0.0, 0.0])
    platform.rPrimePrime_PM_M = np.array([0.0, 0.0, 0.0])
    platform.omega_PM_P = np.array([0.0, 0.0, 0.0])
    platform.omegaPrime_PM_P = np.array([0.0, 0.0, 0.0])
    platform.sigma_PM = sigma_PM
    platform.omega_MB_B = [0.0, 0.0, 0.0]
    platform.omegaPrime_MB_B = [0.0, 0.0, 0.0]
    platform.sigma_MB = [0.0, 0.0, 0.0]
    unitTestSim.AddModelToTask(unitTaskName, platform)
    scObject.addStateEffector(platform)

    # Create rotational motion profiler
    angAccelMax = 0.5 * macros.D2R  # [rad/s^2]
    prescribedRotation = prescribedRotation1DOF.PrescribedRotation1DOF()
    prescribedRotation.ModelTag = "prescribedRotation1DOF"
    prescribedRotation.setRotHat_M(rotAxis_M)
    prescribedRotation.setThetaDDotMax(angAccelMax)
    prescribedRotation.setThetaInit(thetaInit)
    prescribedRotation.setCoastOptionBangDuration(1.0)
    prescribedRotation.setSmoothingDuration(1.0)
    unitTestSim.AddModelToTask(unitTaskName, prescribedRotation)

    # Create the rotational motion reference message
    prescribedThetaRef = 10.0 * macros.D2R  # [rad]
    prescribedRotationMessageData = messaging.HingedRigidBodyMsgPayload()
    prescribedRotationMessageData.theta = prescribedThetaRef
    prescribedRotationMessageData.thetaDot = 0.0  # [rad/s]
    prescribedRotationMessage = messaging.HingedRigidBodyMsg().write(prescribedRotationMessageData)
    prescribedRotation.spinningBodyInMsg.subscribeTo(prescribedRotationMessage)
    platform.prescribedRotationInMsg.subscribeTo(prescribedRotation.prescribedRotationOutMsg)

    # Create translational motion profiler
    transAccelMax = 0.005  # [m/s^2]
    prescribedTranslation = prescribedLinearTranslation.PrescribedLinearTranslation()
    prescribedTranslation.ModelTag = "prescribedLinearTranslation"
    prescribedTranslation.setTransHat_M(transAxis_M)
    prescribedTranslation.setTransAccelMax(transAccelMax)
    prescribedTranslation.setTransPosInit(posInit)
    prescribedTranslation.setCoastOptionBangDuration(1.0)
    prescribedTranslation.setSmoothingDuration(1.0)
    unitTestSim.AddModelToTask(unitTaskName, prescribedTranslation)

    # Create the translational motion reference message
    posRef = 0.1  # [m]
    prescribedTranslationMessageData = messaging.LinearTranslationRigidBodyMsgPayload()
    prescribedTranslationMessageData.rho = posRef
    prescribedTranslationMessageData.rhoDot = 0.0
    prescribedTranslationMessage = messaging.LinearTranslationRigidBodyMsg().write(prescribedTranslationMessageData)
    prescribedTranslation.linearTranslationRigidBodyInMsg.subscribeTo(prescribedTranslationMessage)
    platform.prescribedTranslationInMsg.subscribeTo(prescribedTranslation.prescribedTranslationOutMsg)

    scObject.addStateEffector(particle1)
    # platform.addStateEffector(particle1)

    # scObject.addStateEffector(particle2)
    # platform.addStateEffector(particle2)

    # scObject.addStateEffector(particle3)
    # platform.addStateEffector(particle3)

    # Add Earth as a gravitational body to the simulation
    earthGravBody = gravityEffector.GravBodyData()
    earthGravBody.planetName = "earth_planet_data"
    earthGravBody.mu = 0.3986004415E+15 # meters!
    earthGravBody.isCentralBody = True
    scObject.gravField.gravBodies = spacecraft.GravBodyVector([earthGravBody])
    scObject.hub.r_CN_NInit = [[-4020338.690396649],	[7490566.741852513],	[5248299.211589362]]
    scObject.hub.v_CN_NInit = [[-5199.77710904224],	[-3436.681645356935],	[1041.576797498721]]

    # Set up data logging
    dataLog = scObject.scStateOutMsg.recorder()
    scObjectLog = scObject.logger(["totOrbEnergy", "totOrbAngMomPntN_N", "totRotAngMomPntC_N", "totRotEnergy"])
    unitTestSim.AddModelToTask(unitTaskName, dataLog)
    unitTestSim.AddModelToTask(unitTaskName, scObjectLog)

    # Run the simulation
    simTime = 15.0  # [s]
    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(simTime))
    unitTestSim.ExecuteSimulation()

    orbEnergy = unitTestSupport.addTimeColumn(scObjectLog.times(), scObjectLog.totOrbEnergy)
    orbAngMom_N = unitTestSupport.addTimeColumn(scObjectLog.times(), scObjectLog.totOrbAngMomPntN_N)
    rotAngMom_N = unitTestSupport.addTimeColumn(scObjectLog.times(), scObjectLog.totRotAngMomPntC_N)
    rotEnergy = unitTestSupport.addTimeColumn(scObjectLog.times(), scObjectLog.totRotEnergy)

    plt.close('all')

    plt.figure()
    plt.clf()
    plt.plot(orbAngMom_N[:,0]*1e-9, (orbAngMom_N[:,1] - orbAngMom_N[0,1])/orbAngMom_N[0,1], orbAngMom_N[:,0]*1e-9, (orbAngMom_N[:,2] - orbAngMom_N[0,2])/orbAngMom_N[0,2], orbAngMom_N[:,0]*1e-9, (orbAngMom_N[:,3] - orbAngMom_N[0,3])/orbAngMom_N[0,3])
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Difference")
    unitTestSupport.writeFigureLaTeX("ChangeInOrbitalAngularMomentum" + testCase, "Change in Orbital Angular Momentum " + testCase, plt, r"width=0.8\textwidth", path)

    plt.figure()
    plt.clf()
    plt.plot(orbEnergy[:,0]*1e-9, (orbEnergy[:,1] - orbEnergy[0,1])/orbEnergy[0,1])
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Difference")
    unitTestSupport.writeFigureLaTeX("ChangeInOrbitalEnergy" + testCase, "Change in Orbital Energy " + testCase, plt, r"width=0.8\textwidth", path)

    plt.figure()
    plt.clf()
    plt.plot(rotAngMom_N[:,0]*1e-9, (rotAngMom_N[:,1] - rotAngMom_N[0,1])/rotAngMom_N[0,1], rotAngMom_N[:,0]*1e-9, (rotAngMom_N[:,2] - rotAngMom_N[0,2])/rotAngMom_N[0,2], rotAngMom_N[:,0]*1e-9, (rotAngMom_N[:,3] - rotAngMom_N[0,3])/rotAngMom_N[0,3])
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Difference")
    unitTestSupport.writeFigureLaTeX("ChangeInRotationalAngularMomentum" + testCase, "Change in Rotational Angular Momentum " + testCase, plt, r"width=0.8\textwidth", path)

    plt.figure()
    plt.clf()
    plt.plot(rotEnergy[:,0]*1e-9, (rotEnergy[:,1] - rotEnergy[0,1])/rotEnergy[0,1])
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Difference")
    unitTestSupport.writeFigureLaTeX("ChangeInRotationalEnergy" + testCase, "Change in Rotational Energy " + testCase, plt, r"width=0.8\textwidth", path)

    if show_plots:
        plt.show()
        plt.close('all')

if __name__ == "__main__":
    test_fuelSlosh(True,False,'Gravity')
