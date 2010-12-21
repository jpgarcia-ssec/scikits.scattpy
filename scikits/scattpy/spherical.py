from numpy import *
from scipy.special.orthogonal import ps_roots
from scipy import special
import f_utils

# (r, theta, phi)
class spherical_utilities(object):
	def __init__(self,ng,n,lab):
		self.n=n
		self.ng=ng
		self.set_ngauss()
		self.set_funcs_ang()
		self.set_all_layers(lab)

	def set_ngauss(self):
		k,w = ps_roots(self.ng)
		self.knots,self.weights = k*pi,w*pi
		self.thetas = self.knots
		self.sint=sin(self.thetas)
		self.cost=cos(self.thetas)
		self.tgt = tan(self.thetas)
		self.ctgt=1/self.tgt
	
	def set_funcs_ang(self):
		P = array([special.lpmn(self.n,self.n,ct) for ct in self.cost])
		self.data_Ang,self.data_Angd = P[:,0,:,:],P[:,1,:,:]
	
	def set_all_layers(self,lab):
		self.data_layers=[0]
		for bnd in lab.boundaries():
			r,rd,rdd = bnd.shape.R(self.knots)
			rdr=rd/r
			r2rd2=r**2+rd**2
			Rad =[0,{},{}]
			Radd=[0,{},{}]
			kis=[0,bnd.k1,bnd.k2]
			for i in [1,2]:
				krs = kis[i]*r
				JY = array([special.sph_jnyn(self.n,kr) for kr in krs])[:,:,:]
				Rad[i] ={'j':JY[:,0,:], 'h':JY[:,0,:]+1j*JY[:,2,:]}
				Radd[i]={'j':JY[:,1,:], 'h':JY[:,1,:]+1j*JY[:,3,:]}

			self.data_layers.append({	'ki':kis,\
							'r':r,\
							'rd':rd,\
							'rdd':rdd,\
							'rdr':rdr,\
							'r2rd2':r2rd2,\
							'Rad':Rad,\
							'Radd':Radd })
	def set_layer_no(self,lay):
		self._lay=lay

	def _get_const(self,name):
		return self.data_layers[self._lay][name]

	def _get_r(self):return self._get_const("r")
	r   = property(fget=_get_r)

	def _get_rd(self):return self._get_const("rd")
	rd  = property(_get_rd)

	def _get_rdd(self):return self._get_const("rdd")
	rdd  = property(_get_rdd)

	def _get_rdr(self):return self._get_const("rdr")
	rdr  = property(_get_rdr)

	def _get_r2rd2(self):return self._get_const("r2rd2")
	r2rd2  = property(_get_r2rd2)

	def _get_ki(self):return self._get_const("ki")
	ki  = property(_get_ki)

	def Rad(self,m,ij,i):
		return self.data_layers[self._lay]['Rad'][i][ij][:,m:]

	def Radd(self,m,ij,i):
		return self.data_layers[self._lay]['Radd'][i][ij][:,m:]

	def Ang(self,m):
		return self.data_Ang[:,m,m:]

	def Angd(self,m):
		return self.data_Angd[:,m,m:]


def matA0(C,m,jh,i,coef):
	Rad = C.Rad(m,jh,i)
	Angm = C.Ang(m)
	#func = lambda k:  outer( Rad[k]*Angm[k], coef[k]*Angm[k])
	#return mat_integrate(func)
	return f_utils.mat_a0(Rad,Angm,coef,C.weights)

def matA(C,m,jh,i):
	#return matA0(C,m,jh,i,coef=C.sint)
	Rad = C.Rad(m,jh,i)
	Angm = C.Ang(m)
	return f_utils.mat__a(Rad,Angm,C.sint,C.weights)

def matB(C,m,jh,i):
	ki = C.ki[i]
	Rad = C.Rad(m,jh,i)
	Radd= C.Radd(m,jh,i)
	Angm = C.Ang(m)
	Angmd= C.Angd(m)
	#func = lambda k: \
	#   outer(ki*r[k]*Radd[k]*Angm[k]+rdr[k]*sint[k]*Rad[k]*Angmd[k],\
	#         sint[k]*Angm[k] )
	#return mat_integrate(func)
	#return f_utils.mat_b(ki,Rad,Radd,Angm,Angmd,C.r,C.rdr,C.sint,C.weights)
	return f_utils.mat__b(ki,Rad,Radd,Angm,Angmd,C.r,C.rd,C.sint,C.weights)

def matC(C,m,jh,i,e12, B=None):
	#ki = C.ki[i]
	#if B is None: B = matB(n,m,fRad,fAng,ki)
	#ff = (1.-C.ctgt*C.rdr)*C.sint
	#A = matA0(C,m,jh,i,coef=ff)
	#return e12*B + (e12-1.)*A
	ki = C.ki[i]
	Rad = C.Rad(m,jh,i)
	Radd= C.Radd(m,jh,i)
	Angm = C.Ang(m)
	Angmd= C.Angd(m)
	return f_utils.mat__c(ki,Rad,Radd,Angm,Angmd,C.r,C.rd,C.sint,C.ctgt,e12,C.weights)

def matD0(C,m,jh,i,coef):
	ki = C.ki[i]
	Rad = C.Rad(m,jh,i)
	Radd= C.Radd(m,jh,i)
	Angm = C.Ang(m)
	Angmd= C.Angd(m)
	#func = lambda k: \
	#   outer(ki*r[k]*cost[k]*Radd[k]*Angm[k]+sint[k]**2*Rad[k]*Angmd[k],\
	#         Angm[k]*coef[k] )
	#return mat_integrate(func)
	return f_utils.mat_d0\
			(ki,Rad,Radd,Angm,Angmd,C.r,C.sint,C.cost,coef,C.weights)

def matE0(C,m,jh,i,coef):
	ki = C.ki[i]
	Rad = C.Rad(m,jh,i)
	Radd= C.Radd(m,jh,i)
	Angm = C.Ang(m)
	Angmd= C.Angd(m)
	#func = lambda k: \
	#   outer((ki*r[k]*Radd[k]+Rad[k])*Angm[k], Angm[k]*coef[k] )
	#return mat_integrate(func)
	return f_utils.mat_e0(ki,Rad,Radd,Angm,C.r,coef,C.weights)

def matG0(C,m,jh,i,coef):
	ki = C.ki[i]
	Rad = C.Rad(m,jh,i)
	Radd= C.Radd(m,jh,i)
	Angm = C.Ang(m)
	Angmd= C.Angd(m)
	#func = lambda k: \
	#  outer(ki*rd[k]*Radd[k]*Angm[k] - sint[k]*Rad[k]*Angmd[k],\
	#         Angm[k]*coef[k] )
	#return mat_integrate(func)
	return f_utils.mat_g0(ki,Rad,Radd,Angm,Angmd,C.r,C.rd,C.sint,coef,C.weights)

def matD(C,m,jh,i,e12,B=None):
	if B is None: B = matB(C,m,jh,i)
	fd = C.rdr
	D0 = matD0(C,m,jh,i,coef=fd)
	return B + (e12-1.)*D0

def matE(C,m,jh,i,e12):
	fe = C.rd
	E0 = matE0(C,m,jh,i,coef=fe)
	return (e12-1.)*E0

def matF(C,m,jh,i,e12):
	fd = (C.rd*C.cost-C.r*C.sint)/C.r**2
	D0 = matD0(C,m,jh,i,coef=fd)
	return -(e12-1.)*D0

def matG(C,m,jh,i,e12,B=None):
	if B is None: B = matB(C,m,jh,i)
	fe = (C.rd*C.cost-C.r*C.sint)/C.r
	E0 = matE0(C,m,jh,i,coef=fe)
	return B - (e12-1.)*E0

def matA12(C,m,jh,i,e21,A=None):
	if A is None: A = matA(C,m,jh,i)
	fa = C.r*(C.rd*C.cost-C.r*C.sint)/C.r2rd2
	A0 = matA0(C,m,jh,i,coef=fa)
	return A - (e21-1)*A0

def matA14(C,m,jh,i,e21):
	fa = (C.r**2*C.rd)/C.r2rd2
	A0 = matA0(C,m,jh,i,coef=fa)
	return -(e21-1)*A0

def matA32(C,m,jh,i,e21):
	fa = (C.rd*C.sint+C.r*C.cost)*(C.rd*C.cost-C.r*C.sint)/(C.r*C.r2rd2)
	A0 = matA0(C,m,jh,i,coef=fa)
	return (e21-1)*A0

def matA34(C,m,jh,i,e21,A=None):
	if A is None: A = matA(C,m,jh,i)
	fa = C.rd*(C.rd*C.sint+C.r*C.cost)/C.r2rd2
	A0 = matA0(C,m,jh,i,coef=fa)
	return A + (e21-1)*A0

def matA22(C,m,jh,i,e21,B=None):
	if B is None: B=matB(C,m,jh,i)
	fg = C.rd*(C.rd*C.cost-C.r*C.sint)/C.r2rd2
	fa = (C.r**2 - C.r*C.rdd + 2*C.rd**2) \
	    *( C.r*(C.rd*C.cost-C.r*C.sint) + C.rd*(C.rd*C.sint+C.r*C.cost) ) \
	    /(C.r2rd2)**2
	#fa = f_utils.f1(r,rd,rdd,sint,cost)
	A0 = matA0(C,m,jh,i,coef=fa)
	G0 = matG0(C,m,jh,i,coef=fg)
	return B - (e21-1)*(G0-A0)

def matA24(C,m,jh,i,e21):
	fg = C.r*C.rd**2/C.r2rd2
	fa = C.rd*(C.r**4 - 2*C.r**3*C.rdd + 2*C.r**2*C.rd**2 - C.rd**4) \
	    /(C.r2rd2)**2
	#fa = f_utils.f2(r,rd,rdd) * rd
	A0 = matA0(C,m,jh,i,coef=fa)
	G0 = matG0(C,m,jh,i,coef=fg)
	return -(e21-1)*(G0-A0)

def matA42(C,m,jh,i,e21):
	fg = (C.rd*C.cost-C.r*C.sint)**2/(C.r*C.r2rd2)
	fa = 2*(C.r**2 - C.r*C.rdd + 2*C.rd**2) \
	    *(C.rd*C.cost-C.r*C.sint)*(C.rd*C.sint+C.r*C.cost) \
	    /(C.r*C.r2rd2**2)
	#fa = f_utils.f3(r,rd,rdd,sint,cost) / r
	A0 = matA0(C,m,jh,i,coef=fa)
	G0 = matG0(C,m,jh,i,coef=fg)
	return (e21-1)*(G0-A0)

def matA44(C,m,jh,i,e21,B=None):
	if B is None: B = matB(C,m,jh,i)
	fg = C.rd*(C.rd*C.cost-C.r*C.sint)/C.r2rd2
	fa =((C.r**3*C.rdd + C.r**2*C.rd**2 - C.r*C.rd**2*C.rdd +3*C.rd**4)*C.sint\
	    +C.rdr*C.cost*(C.r**4 - 2*C.r**3*C.rdd + 2*C.r**2*C.rd**2 - C.rd**4)) \
	    /(C.r2rd2)**2
	#fa = f_utils.f4(r,rd,rdd,sint,cost)
	A0 = matA0(C,m,jh,i,coef=fa)
	G0 = matG0(C,m,jh,i,coef=fg)
	return B + (e21-1)*(G0-A0)
