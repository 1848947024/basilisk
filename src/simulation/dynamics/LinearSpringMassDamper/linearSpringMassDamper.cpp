/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

 */

#include "linearSpringMassDamper.h"
#include "architecture/utilities/avsEigenSupport.h"

/*! This is the constructor, setting variables to default values */
LinearSpringMassDamper::LinearSpringMassDamper()
{
	// - zero the contributions for mass props and mass rates
	this->effProps.mEff = 0.0;
	this->effProps.IEffPntB_B.setZero();
	this->effProps.rEff_CB_B.setZero();
	this->effProps.rEffPrime_CB_B.setZero();
	this->effProps.IEffPrimePntB_B.setZero();

	// - Initialize the variables to working values
	this->massSMD = 1.0;
	this->r_PB_B.setZero();
	this->pHat_B.setIdentity();
	this->k = 1.0;
	this->c = 0.0;
    this->rhoInit = 0.0;
    this->rhoDotInit = 0.0;
    this->massInit = 0.0;
	this->nameOfRhoState = "linearSpringMassDamperRho" + std::to_string(this->effectorID);
	this->nameOfRhoDotState = "linearSpringMassDamperRhoDot" + std::to_string(this->effectorID);
	this->nameOfMassState = "linearSpringMassDamperMass" + std::to_string(this->effectorID);
    this->effectorID++;

	return;
}

uint64_t LinearSpringMassDamper::effectorID = 1;

/*! This is the destructor, nothing to report here */
LinearSpringMassDamper::~LinearSpringMassDamper()
{
    this->effectorID = 1;    /* reset the panel ID*/
    return;
}

void LinearSpringMassDamper::Reset(uint64_t CurrentClock) {
}

/*! Method for spring mass damper particle to access the states that it needs. It needs gravity and the hub states */
void LinearSpringMassDamper::linkInStates(DynParamManager& statesIn)
{
    // Get access to the hub's states needed for dynamic coupling
    this->hubOmega = statesIn.getStateObject("hubOmega");

    // - Grab access to gravity
    this->g_N = statesIn.getPropertyReference("g_N");

    // - Grab access to c_B and cPrime_B
    this->c_B = statesIn.getPropertyReference(this->propName_centerOfMassSC);
    this->cPrime_B = statesIn.getPropertyReference(this->propName_centerOfMassPrimeSC);

    return;
}

/*! This method is used to link properties
 @return void
 @param properties The parameter manager to collect from
 */
void LinearSpringMassDamper::linkInPrescribedMotionProperties(DynParamManager& properties)
{
    this->prescribedPositionProperty = properties.getPropertyReference(this->propName_prescribedPosition);
    this->prescribedVelocityProperty = properties.getPropertyReference(this->propName_prescribedVelocity);
    this->prescribedAccelerationProperty = properties.getPropertyReference(this->propName_prescribedAcceleration);
    this->prescribedAttitudeProperty = properties.getPropertyReference(this->propName_prescribedAttitude);
    this->prescribedAngVelocityProperty = properties.getPropertyReference(this->propName_prescribedAngVelocity);
    this->prescribedAngAccelerationProperty = properties.getPropertyReference(this->propName_prescribedAngAcceleration);
}

/*! This is the method for the spring mass damper particle to register its states: rho and rhoDot */
void LinearSpringMassDamper::registerStates(DynParamManager& states)
{
    // - Register rho and rhoDot
	this->rhoState = states.registerState(1, 1, nameOfRhoState);
    Eigen::MatrixXd rhoInitMatrix(1,1);
    rhoInitMatrix(0,0) = this->rhoInit;
    this->rhoState->setState(rhoInitMatrix);
	this->rhoDotState = states.registerState(1, 1, nameOfRhoDotState);
    Eigen::MatrixXd rhoDotInitMatrix(1,1);
    rhoDotInitMatrix(0,0) = this->rhoDotInit;
    this->rhoDotState->setState(rhoDotInitMatrix);

	// - Register mass
	this->massState = states.registerState(1, 1, nameOfMassState);
    Eigen::MatrixXd massInitMatrix(1,1);
    massInitMatrix(0,0) = this->massInit;
    this->massState->setState(massInitMatrix);

	return;
}

/*! This is the method for the SMD to add its contributions to the mass props and mass prop rates of the vehicle */
void LinearSpringMassDamper::updateEffectorMassProps(double integTime)
{
	// - Grab rho from state manager and define r_PcB_B
	this->rho = this->rhoState->getState()(0,0);
	this->r_PcB_B = this->rho * this->pHat_B + this->r_PB_B;
	this->massSMD = this->massState->getState()(0, 0);

	// - Update the effectors mass
	this->effProps.mEff = this->massSMD;
	// - Update the position of CoM
	this->effProps.rEff_CB_B = this->r_PcB_B;
	// - Update the inertia about B
	this->rTilde_PcB_B = eigenTilde(this->r_PcB_B);
	this->effProps.IEffPntB_B = this->massSMD * this->rTilde_PcB_B * this->rTilde_PcB_B.transpose();

	// - Grab rhoDot from the stateManager and define rPrime_PcB_B
	this->rhoDot = this->rhoDotState->getState()(0, 0);
	this->rPrime_PcB_B = this->rhoDot * this->pHat_B;
	this->effProps.rEffPrime_CB_B = this->rPrime_PcB_B;

	// - Update the body time derivative of inertia about B
	this->rPrimeTilde_PcB_B = eigenTilde(this->rPrime_PcB_B);
	this->effProps.IEffPrimePntB_B = -this->massSMD*(this->rPrimeTilde_PcB_B*this->rTilde_PcB_B
                                                                          + this->rTilde_PcB_B*this->rPrimeTilde_PcB_B);

    return;
}

/*! This is method is used to pass mass properties information to the fuelTank */
void LinearSpringMassDamper::retrieveMassValue(double integTime)
{
    // Save mass value into the fuelSlosh class variable
    this->fuelMass = this->massSMD;

    return;
}

/*! This method is for the SMD to add its contributions to the back-sub method */
void LinearSpringMassDamper::updateContributions(double integTime, BackSubMatrices & backSubContr, Eigen::Vector3d sigma_BN, Eigen::Vector3d omega_BN_B, Eigen::Vector3d g_N)
{
    // - Find dcm_BN
    Eigen::MRPd sigmaLocal_BN;
    Eigen::Matrix3d dcm_BN;
    Eigen::Matrix3d dcm_NB;
    sigmaLocal_BN = (Eigen::Vector3d ) sigma_BN;
    dcm_NB = sigmaLocal_BN.toRotationMatrix();
    dcm_BN = dcm_NB.transpose();

    // - Map gravity to body frame
    Eigen::Vector3d gLocal_N;
    Eigen::Vector3d g_B;
    gLocal_N = *this->g_N;
    g_B = dcm_BN*gLocal_N;

	// - Define aRho
    this->aRho = -this->pHat_B;

    // - Define bRho
    this->bRho = -this->rTilde_PcB_B*this->pHat_B;

    // - Define cRho
    Eigen::Vector3d omega_BN_B_local = omega_BN_B;
    Eigen::Matrix3d omegaTilde_BN_B_local;
    omegaTilde_BN_B_local = eigenTilde(omega_BN_B_local);
	cRho = 1.0/(this->massSMD)*(this->pHat_B.dot(this->massSMD * g_B) - this->k*this->rho - this->c*this->rhoDot
		         - 2 * this->massSMD*this->pHat_B.dot(omegaTilde_BN_B_local * this->rPrime_PcB_B)
		                   - this->massSMD*this->pHat_B.dot(omegaTilde_BN_B_local*omegaTilde_BN_B_local*this->r_PcB_B));

	// - Compute matrix/vector contributions
	backSubContr.matrixA = this->massSMD*this->pHat_B*this->aRho.transpose();
    backSubContr.matrixB = this->massSMD*this->pHat_B*this->bRho.transpose();
    backSubContr.matrixC = this->massSMD*this->rTilde_PcB_B*this->pHat_B*this->aRho.transpose();
	backSubContr.matrixD = this->massSMD*this->rTilde_PcB_B*this->pHat_B*this->bRho.transpose();
	backSubContr.vecTrans = -this->massSMD*this->cRho*this->pHat_B;
	backSubContr.vecRot = -this->massSMD*omegaTilde_BN_B_local * this->rTilde_PcB_B *this->rPrime_PcB_B -
                                                             this->massSMD*this->cRho*this->rTilde_PcB_B * this->pHat_B;
    return;
}

void LinearSpringMassDamper::addPrescribedMotionCouplingContributions(BackSubMatrices & backSubContr) {

    // Access prescribed motion properties
    Eigen::Vector3d r_PB_B = (Eigen::Vector3d)*this->prescribedPositionProperty;
    Eigen::Vector3d rPrime_PB_B = (Eigen::Vector3d)*this->prescribedVelocityProperty;
    Eigen::Vector3d rPrimePrime_PB_B = (Eigen::Vector3d)*this->prescribedAccelerationProperty;
    Eigen::MRPd sigma_PB;
    sigma_PB = (Eigen::Vector3d)*this->prescribedAttitudeProperty;
    Eigen::Vector3d omega_PB_P = (Eigen::Vector3d)*this->prescribedAngVelocityProperty;
    Eigen::Vector3d omegaPrime_PB_P = (Eigen::Vector3d)*this->prescribedAngAccelerationProperty;
    Eigen::Matrix3d dcm_PB = sigma_PB.toRotationMatrix().transpose();

    // Collect hub states
    Eigen::Vector3d omega_bN_b = this->hubOmega->getState();
    Eigen::Vector3d omega_BN_P = dcm_PB * omega_bN_b;

    // Prescribed motion translation coupling contributions
    Eigen::Vector3d fHat_P = this->pHat_B;
    Eigen::Vector3d r_PB_P = dcm_PB * r_PB_B;
    Eigen::Matrix3d rTilde_PB_P = eigenTilde(r_PB_P);
    backSubContr.matrixB += - this->massSMD * fHat_P * this->aRho.transpose() * rTilde_PB_P;

    Eigen::Matrix3d omegaTilde_PB_P = eigenTilde(omega_PB_P);
    Eigen::Vector3d rPPrime_FcP_P = this->rPrime_PcB_B;
    Eigen::Matrix3d omegaPrimeTilde_PB_P = eigenTilde(omegaPrime_PB_P);
    Eigen::Vector3d r_FcP_P = this->r_PcB_B;
    Eigen::Vector3d rPrimePrime_PB_P = dcm_PB * rPrimePrime_PB_B;
    Eigen::Matrix3d omegaTilde_BN_P = eigenTilde(omega_BN_P);
    Eigen::Vector3d rPrime_PB_P = dcm_PB * rPrime_PB_B;
    Eigen::Vector3d term1 = 2.0 * omegaTilde_PB_P * rPPrime_FcP_P
                            + omegaPrimeTilde_PB_P * r_FcP_P
                            + omegaTilde_PB_P * omegaTilde_PB_P * r_FcP_P
                            + rPrimePrime_PB_P;
    Eigen::Vector3d term2 = rPrimePrime_PB_P + 2.0 * omegaTilde_BN_P * rPrime_PB_P
                            + omegaTilde_BN_P * omegaTilde_BN_P * r_PB_P;
    Eigen::Vector3d term3 = omegaPrime_PB_P + omegaTilde_BN_P * omega_PB_P;
    backSubContr.vecTrans += - this->massSMD * term1
                             - this->massSMD * this->aRho.transpose() * term2 * fHat_P
                             - this->massSMD * this->bRho.transpose() * term3 * fHat_P;

    // Prescribed motion rotation coupling contributions
    backSubContr.matrixC += this->massSMD * rTilde_PB_P * fHat_P * this->aRho.transpose();

    Eigen::Vector3d r_FcB_P = r_FcP_P + r_PB_P;
    Eigen::Matrix3d rTilde_FcB_P = eigenTilde(r_FcB_P);
    backSubContr.matrixD += this->massSMD * rTilde_PB_P * fHat_P * this->bRho.transpose()
                            - this->massSMD * rTilde_FcB_P * fHat_P * this->aRho.transpose() * rTilde_PB_P;

    Eigen::Vector3d omega_PN_P = omega_PB_P + omega_BN_P;
    Eigen::Matrix3d omegaTilde_PN_P = eigenTilde(omega_PN_P);
    Eigen::Matrix3d rTilde_FcP_P = eigenTilde(r_FcP_P);

    Eigen::Vector3d vecRotTerm1 = - this->massSMD * rTilde_FcB_P * term1;
    Eigen::Vector3d vecRotTerm2 = - this->massSMD * (omegaTilde_BN_P * rTilde_PB_P - omegaTilde_PB_P * rTilde_FcP_P) * rPPrime_FcP_P;
                                  - this->massSMD * omegaTilde_BN_P * rTilde_FcB_P * (omegaTilde_PB_P * r_FcP_P + rPrime_PB_P);
    Eigen::Vector3d vecRotTerm3 = - this->massSMD * this->cRho * rTilde_PB_P * fHat_P;
    Eigen::Vector3d vecRotTerm4 = - this->massSMD * rTilde_FcB_P * fHat_P * (this->aRho.transpose() * term2)
                                  - this->massSMD * rTilde_FcB_P * fHat_P * (this->bRho.transpose() * term3);
    backSubContr.vecRot += vecRotTerm1
                           + vecRotTerm2
                           + vecRotTerm3
                           + vecRotTerm4;
}

/*! This method is used to define the derivatives of the SMD. One is the trivial kinematic derivative and the other is
 derived using the back-sub method */
void LinearSpringMassDamper::computeDerivatives(double integTime, Eigen::Vector3d rDDot_BN_N, Eigen::Vector3d omegaDot_BN_B, Eigen::Vector3d sigma_BN)
{

	// - Find DCM
	Eigen::MRPd sigmaLocal_BN;
	Eigen::Matrix3d dcm_BN;
	sigmaLocal_BN = (Eigen::Vector3d) sigma_BN;
	dcm_BN = (sigmaLocal_BN.toRotationMatrix()).transpose();

	// - Set the derivative of rho to rhoDot
	this->rhoState->setDerivative(this->rhoDotState->getState());

	// - Compute rhoDDot
	Eigen::MatrixXd conv(1,1);
    Eigen::Vector3d omegaDot_BN_B_local = omegaDot_BN_B;
    Eigen::Vector3d rDDot_BN_N_local = rDDot_BN_N;
	Eigen::Vector3d rDDot_BN_B_local = dcm_BN*rDDot_BN_N_local;
    conv(0, 0) = this->aRho.dot(rDDot_BN_B_local) + this->bRho.dot(omegaDot_BN_B_local) + this->cRho;
	this->rhoDotState->setDerivative(conv);

    // - Set the massDot already computed from fuelTank to the stateDerivative of mass
    conv(0,0) = this->fuelMassDot;
    this->massState->setDerivative(conv);

    return;
}

/*! This method is for the SMD to add its contributions to energy and momentum */
void LinearSpringMassDamper::updateEnergyMomContributions(double integTime, Eigen::Vector3d & rotAngMomPntCContr_B,
                                                          double & rotEnergyContr, Eigen::Vector3d omega_BN_B)
{
    //  - Get variables needed for energy momentum calcs
    Eigen::Vector3d omegaLocal_BN_B;
    omegaLocal_BN_B = omega_BN_B;
    Eigen::Vector3d rDotPcB_B;

    // - Find rotational angular momentum contribution from hub
    rDotPcB_B = this->rPrime_PcB_B + omegaLocal_BN_B.cross(this->r_PcB_B);
    rotAngMomPntCContr_B = this->massSMD*this->r_PcB_B.cross(rDotPcB_B);

    // - Find rotational energy contribution from the hub
    rotEnergyContr = 1.0/2.0*this->massSMD*rDotPcB_B.dot(rDotPcB_B) + 1.0/2.0*this->k*this->rho*this->rho;

    return;
}

void LinearSpringMassDamper::calcForceTorqueOnBody(double integTime, Eigen::Vector3d omega_BN_B)
{
    // - Get the current omega state
    Eigen::Vector3d omegaLocal_BN_B;
    omegaLocal_BN_B = omega_BN_B;
    Eigen::Matrix3d omegaLocalTilde_BN_B;
    omegaLocalTilde_BN_B = eigenTilde(omegaLocal_BN_B);

    // - Get rhoDDot from last integrator call
    double rhoDDotLocal;
    rhoDDotLocal = rhoDotState->getStateDeriv()(0, 0);

    // - Calculate force that the FSP is applying to the spacecraft
    this->forceOnBody_B = -(this->massSMD*this->pHat_B*rhoDDotLocal + 2*omegaLocalTilde_BN_B*this->massSMD
                            *this->rhoDot*this->pHat_B);

    // - Calculate torque that the FSP is applying about point B
    this->torqueOnBodyPntB_B = -(this->massSMD*this->rTilde_PcB_B*this->pHat_B*rhoDDotLocal + this->massSMD*omegaLocalTilde_BN_B*this->rTilde_PcB_B*this->rPrime_PcB_B - this->massSMD*(this->rPrimeTilde_PcB_B*this->rTilde_PcB_B + this->rTilde_PcB_B*this->rPrimeTilde_PcB_B)*omegaLocal_BN_B);

    // - Define values needed to get the torque about point C
    Eigen::Vector3d cLocal_B = *this->c_B;
    Eigen::Vector3d cPrimeLocal_B = *this->cPrime_B;
    Eigen::Vector3d r_PcC_B = this->r_PcB_B - cLocal_B;
    Eigen::Vector3d rPrime_PcC_B = this->rPrime_PcB_B - cPrimeLocal_B;
    Eigen::Matrix3d rTilde_PcC_B = eigenTilde(r_PcC_B);
    Eigen::Matrix3d rPrimeTilde_PcC_B = eigenTilde(rPrime_PcC_B);

    // - Calculate the torque about point C
    this->torqueOnBodyPntC_B = -(this->massSMD*rTilde_PcC_B*this->pHat_B*rhoDDotLocal + this->massSMD*omegaLocalTilde_BN_B*rTilde_PcC_B*rPrime_PcC_B - this->massSMD*(rPrimeTilde_PcC_B*rTilde_PcC_B + rTilde_PcC_B*rPrimeTilde_PcC_B)*omegaLocal_BN_B);

    return;
}
