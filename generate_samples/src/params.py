import numpy as np 


# gather system parameters
class Params():
	def __init__(self, 
					Nx=2, Ny=2, 
					omega0=0., gamma0=0., 
					v=1., w=0.5, phi=0.,
					kappa=0., 
					gamma_c=1., delta=0., Ns=30.,
					offset=1., delta_gaussian=1.0
					):
		"""
		Init the parameters of the system
		"""
		self.Nx, self.Ny = Nx, Ny 	# system size
		self.omega0 = omega0 		# resonant freq resonator
		self.gamma0 = gamma0 			# internal loss
		self.v = v					# coupling coeff
		self.w = w					# coupling coeff
		self.phi = phi				# Haldane flux
		self.kappa = kappa  		# Kerr coeff
		self.gamma_c = gamma_c 		# in/out couplings
		self.delta = delta 			# detuning source
		self.offset = offset 			# offset
		self.delta_gaussian = delta_gaussian 			# detuning source
		self.Ns = Ns				# number points per cycle

		self.T = 2*np.pi / (omega0+delta) if (omega0+delta != 0) else 2*np.pi / 1.
		self.dt = self.T / Ns
		

	def __str__(self):
		str_params = "#####################\n### Params system ###\n#####################\n"
		str_params +=  f"Size: (Nx, Ny) = ({self.Nx}, {self.Ny}) \n"
		str_params += f"Resonant freq: {self.omega0} \n"
		str_params += f"Internal loss: {self.gamma0} \n"
		str_params += f"Coupling ceoff: {self.v} \n"
		str_params += f"Coupling ceoff: {self.w} \n"
		str_params += f"Haldane flux: {self.phi/np.pi}*pi \n"
		str_params += f"Kerr coeff: {self.kappa} \n"
		str_params += f"In/Out coupling: {self.gamma_c} \n"
		str_params += f"Detuning source: {self.delta} \n"
		str_params += f"Width of gaussian: {self.delta_gaussian} \n"
		str_params += f"Number points per cylce: {self.Ns} \n"
		str_params += f"Characteristic time: {self.T} \n"
		str_params += f"dt: {self.dt} \n"
		str_params += f"offset: {self.offset} \n"
		return str_params
