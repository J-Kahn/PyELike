# Code to implement empircal likelihood estimation

import numpy as np
import scipy.optimize as opt

def cueobj(x):
	return x**2

def cueobjg(x):
	return 2*x

def cueobjgg(x):
	return 2 + (x * 0)

def logm(x2, e = 0.001):
	if x2 < e:
		return np.log(e) - 1.5 + 2*x2/e - x2**2/(2*e**2)
	else:
		return np.log(x2)

logmv = np.vectorize(logm)

def elobj(x):
	return logmv(1 + x)

def elobjg(x):
	return 1 / (x + 1)

def elobjgg(x):
	return -1 / np.power(x + 1, 2)

def etobj(x):
	return -np.exp(x)

def wtfever(x):
	return x

class elspec:

	def __init__(self, moment, type = "EM", grad = ""):

		self.moments = moment 
		self.grad = grad

		self.data = np.nan
		self.lagrange = np.nan
		self.prob = np.nan
		self.res = np.nan

		self.theta = np.nan
		self.ltol = 50
		self.np = 0
		self.nm = 0

		self.obj = elobj
		self.gobj = elobjg
		self.ggobj = elobjgg
		self.estim = ""
		self.W = ""

		if type == "ET":
			self.obj = etobj
			self.gobj = etobj
			self.ggobj = etobj
		if type == "CUE":
			self.obj = cueobj
			self.objg = cueobjg
			self.objgg = cueobjgg

	def add_data(self, x):
		self.data = x

	def gel_estimate(self, initpar):
		self.mv(initpar)
		self.np = len(initpar)
		self.nm = self.res.shape[1]

		self.lagrange = np.zeros(self.nm)
		self.estim = opt.minimize(self.gel_obj, initpar, method = 'Nelder-Mead')
		self.theta = self.estim.x

	def gmm_estimate(self, initpar):
		self.mv(initpar)
		self.np = len(initpar)
		self.nm = self.res.shape[1]
		self.mdiff = 50
		self.W = np.identity(self.nm) 

		while self.mdiff > 1e-6:
			self.estim = opt.minimize(self.gmm_obj, initpar, method = 'Nelder-Mead')
			par = self.estim.x
			self.update_weight(par)
			self.mdiff = np.max((self.estim.x - initpar)**2)
			print(self.mdiff)
			initpar = self.estim.x

		self.theta = par

	def gmm_obj(self, par):
		self.mv(par)
		moe = np.mean(self.res, axis = 0)
		return np.dot(np.dot(moe.T, self.W), moe)

	def update_weight(self, par):
		self.mv(par)
		self.W = np.linalg.inv(np.dot(self.res.T,self.res))/float(self.data.shape[0])

	def gel_obj(self, par):

		self.mv(par)

		self.ltol = 50
		self.lagrange = np.zeros((self.nm,1))

		while self.ltol > 1e-6:
			self.lagrange_step(par)

		ovec = self.om(par)

		return np.mean(ovec)


	def lagrange_step(self, par):
		xx = np.dot(self.res.T, self.ggom(par) * self.res)
		xy = np.dot(self.res.T,self.gom(par))

		lnew = self.lagrange - np.linalg.solve(xx, xy)/float(self.data.shape[0])

		self.ltol = np.sum((lnew - self.lagrange)**2)

		self.lagrange = lnew

	def mv(self, par):

		self.res = self.moments(par, self.data)

	def om(self, par):
		return self.obj(np.dot(self.res, self.lagrange))

	def gom(self, par):
		return self.gobj(np.dot(self.res, self.lagrange))

	def ggom(self, par):
		return self.ggobj(np.dot(self.res, self.lagrange))

	def pv(self, par = ''):

		if par == '':
			par = self.theta

		return self.gom(par)/float(self.data.shape[0])

	def ecdfi(self, x, par = ''):

		if par == '':
			par = self.theta

		xer = self.gom(par)/float(self.data.shape[0]) * (self.data<=x)
		return np.sum(xer)

	def ecdf(self, x, par = ''):

		pvec = np.zeros(x.shape)
		
		for i in range(0,x.shape[0]):

			pvec[i] = self.ecdfi(x[i], par)

		return pvec