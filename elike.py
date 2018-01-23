# Code to implement empircal likelihood estimation

import numpy as np
import scipy.optimize as opt

from numpy import *
from math import *

#	 Numerical Jacobian using Richardson extrapolation
#	 Jay Kahn - University of Rochester, November 19, 2012
#	 f - function to take derivative over
#	 x - point (vector) at which to evaluate derivatives
#	 o - order of error term desired relative to f
#	 h1 - starting factor
#	 *control - any extra arguments to be passed to f
def richardson(f, x0, o, h1, v, *control):
	x=np.array(x0)
	d=x.shape[0]
	i=1
	r=o/2
	while i <= d:
		j=1
		while j <= r:
			if j==1:
				h=h1
			else:
				h=h/v
			
			idd=np.eye(d)*h
			xup=x+idd[:,i-1]
			xdown=x-idd[:,i-1]
			fat=f(x,*control)
			fup=f(xup,*control)
			fdown=f(xdown,*control)
			ddu=fup-fat
			ddd=fdown-fat
			
			hp=h
			
			if j==1:
				dds=np.array([ddu, ddd])
				hhs=np.array([[hp, -hp]])
				
			else:
				dds=concatenate((dds, np.array([ddu, ddd])),0)
				hhs=concatenate((hhs, np.array([[hp, -hp]])),1)
				
			j=j+1
			
		mat=hhs
		
		j=2
		
		while j<=o:
			mat=np.concatenate((mat, power(hhs,j)/factorial(j)),0)
			j=j+1
		
		mat
		der= np.dot(np.transpose(np.linalg.inv(mat)),dds)
		
		
		if i==1:
			g=der
		else:
			g=np.concatenate((g,der),1)
		
		i=i+1
	return g

#	 Jacobian running as shell of Richardson. Ends up with matrix
#	 whose rows are derivatives with respect to different elements
#	 of x and columns are derivatives of different elements of f(x).
#	 For scalar valued f(x) simplifies to column gradient.
#	 Jay Kahn - University of Rochester, November 19, 2012
#	 f - function to take derivative over
#	 x - point (vector) at which to evaluate derivatives
#	 o - order of error term desired relative to f
#	 h1 - starting factor
#	 *control - any extra arguments to be passed to f
def jacobian(f, x0, o=4, h1=0.5, v=2, *control):
	fn=f(x0,*control).shape[0]
	x=np.array(x0)
	xn=x.shape[0]
	J=np.zeros((xn,fn))
	g=richardson(f, x, o, h1, v, *control)
	j=0
	while j<=xn-1:
		i=0
		while i<=fn-1:
			J[j,i]=g[0,i+j*fn]
			i=i+1
		j=j+1
	return J.T


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

	def gmm_iteration(self, initpar, method = 'Nelder-Mead', minimizer_kwargs = {}):
		if method == 'basinhopping':
			self.estim = opt.basinhopping(self.gmm_obj, initpar, minimizer_kwargs = {})
		else:
			self.estim = opt.minimize(self.gmm_obj, initpar, method = method)
		par = self.estim.x
		        
		return par
    
	def gmm_initiate(self, initpar):
		self.mv(initpar)
		self.np = len(initpar)
		self.nm = self.res.shape[1]
		self.mdiff = 50
		self.W = np.identity(self.nm)         
        
	def gmm_estimate(self, initpar, method = 'Nelder-Mead', maxit = 5000, minimizer_kwargs = {}):
		self.gmm_initiate(initpar)
		it = 0
		while self.mdiff > 1e-6:
			
			par = self.gmm_iteration(initpar, method = method, minimizer_kwargs = minimizer_kwargs)
			self.update_weight(par)   
			self.mdiff = np.max((par - initpar)**2)
			print(self.mdiff)
			if it > maxit:
				self.mdiff = 0
			initpar = par

		self.theta = par

	def gmm_obj(self, par):
		self.mv(par)
		moe = np.mean(self.res, axis = 0)
		return np.dot(np.dot(moe.T, self.W), moe)
    
	def gmm_var(self, par, efficient = False):
		G = jacobian(self.mvm,self.theta,4,0.5,2)
		gwginv = np.linalg.inv(np.dot(np.dot(G.T, self.W), G))
		if efficient:
			self.var = gwginv / float(self.data.shape[0])
		else:
			omega = self.get_var(par)
			gwowg = np.dot(np.dot(G.T, self.W), omega)
			gwowg = np.dot(gwowg, np.dot(self.W.T, G))
			self.var = np.dot(np.dot(gwginv, gwowg), gwginv)/float(self.data.shape[0])

	def gmm_jstat(self, par, efficient = False):
		if efficient:
			psi = np.linalg.inv(self.W)
			invpsi = self.W
		else:
			omega = self.get_var(par)
			G = jacobian(self.mvm,self.theta,4,0.5,2)
			peye = np.eye(G.shape[0])
			gwginv = np.linalg.inv(np.dot(np.dot(G.T, self.W), G))
			gwc = np.dot(np.dot(np.dot(G, gwginv), G.T), self.W)
			psi = np.dot(np.dot(peye - gwc, omega), (peye - gwc).T)
			invpsi = np.linalg.pinv(psi)
		self.psi = psi / float(self.data.shape[0])
		m = self.mvm(par)
		self.J = np.dot(np.dot(m.transpose(), invpsi), m) * float(self.data.shape[0])

	def update_weight(self, par):
		self.mv(par)
		self.W = np.linalg.inv(np.dot(self.res.T,self.res)/float(self.data.shape[0]))

	def get_var(self, par):
		self.mv(par)
		omega= np.dot(self.res.T,self.res)/float(self.data.shape[0])
		return omega

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

	def mvm(self, par):

		return np.mean(self.moments(par, self.data),0)

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