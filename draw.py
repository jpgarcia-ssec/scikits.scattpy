from numpy import *
from scikits import scattpy
from scikits.scattpy import f_coords
import pylab
from scipy.special import sph_jn,sph_jnyn,lpmn

def get_Pmn(m,n,x):
	return array([lpmn(m,n,xl) for xl in x])

def norm(m,l):
	return sqrt((2*l+1)/2. *factorial(l-m)/factorial(l+m))

def get_Pmn_normed(m,n,x):
	P = get_Pmn(m,n,x)
	for M in xrange(m+1):
		for L in xrange(n+1):
			P[:,:,M,L] = P[:,:,M,L]*norm(M,L)
	return P

def get_JnHn(n,x):
	JnYn = array([sph_jnyn(n,xl) for xl in x])
	return JnYn[:,:2,:],JnYn[:,:2,:]+1j*JnYn[:,2:,:]

def get_Ui_Vr(xs,ys,zs,coefs,ki,jh):
	Rx = zeros([size(xs),size(ys),size(zs)],complex)
	Ry = zeros([size(xs),size(ys),size(zs)],complex)
	Rz = zeros([size(xs),size(ys),size(zs)],complex)
	
	for i,x in enumerate(xs):
	  print x
	  for j,y in enumerate(ys):
	    for k,z in enumerate(zs):
		U=V=0.
		cPoint = [x,y,z]
		sPoint = f_coords.point_c2s(cPoint)
		r,t,p = sPoint
		cost = cos(t)
		n  = shape(coefs)[2]
		for m,coefs_m in enumerate(coefs):
			if m==0:continue # skip axisymmetric part
		        P,Pd = get_Pmn(m,n,[cost])[0]
        		Pml  = P[m,m:n+1]
		        Pdml = Pd[m,m:n+1]
        		Bess,Hank = get_JnHn(n,[ki*r])
			if jh=='j':
				Rad = Bess[0,0,m:]
				Radd= Bess[0,1,m:]
			else:
	        		Rad = Hank[0,0,m:]
        			Radd= Hank[0,1,m:]
			Nsize = n-m+1
		        a = coefs_m[0,-Nsize:]
        		b = coefs_m[1,-Nsize:]
	        	l = arange(m,n+1)

			U += sum(a*Rad*Pml)*cos(m*p)
			V += sum(b*Rad*Pml)*cos(m*p)

		sVec = (V,0.,0.)
		cVec = f_coords.vector_s2c(sVec,sPoint)
		cVec[2] += U
		Rx[i,j,k],Ry[i,j,k],Rz[i,j,k] = cVec
	return array([Rx,Ry,Rz])

def get_I(particle,xs,ys,zs):
	I = zeros([len(xs),len(ys),len(zs)],int)

	for i,x in enumerate(xs):
	  print x
	  for j,y in enumerate(ys):
	    for k,z in enumerate(zs):
			cPoint = [x,y,z]
			sPoint = f_coords.point_c2s(cPoint)
			r,t,p = sPoint
			for lay_no in xrange(particle.Nlayers):
				R,Rd,Rdd = particle.layers[0].shape.R(t)
				if r<R:
					I[i,j,k]=lay_no+1
				
	return I

def get_If(particle,xs,ys,zs):
	def fR(t):
		return particle.layers[0].shape.R(t)[0]
	return array(f_coords.get_i(fR,xs,ys,zs))

def get_all(particle,xs,ys,zs,res=None,wavelen=None):
	global Hmax,Hmin,Emax,Emin,Pmax,Pmin
	LAB = scattpy.Lab(particle,0.)
	if res:
		RESULTS = res
	else:
		print "solve"
		RESULTS = scattpy.svm(LAB,allfields=True)

	print "indicator function"
	dx=xs[1]-xs[0]
	dy=ys[1]-ys[0]
	dz=zs[1]-zs[0]
	I = get_I(particle,xs,ys,zs)
	III = array([I,I,I])

	R = zeros(III.shape,dtype=complex)

	for lay_no,c_lay_tm in enumerate(RESULTS._c_all_tm):
		print "Layer No.%s" % lay_no
		if lay_no==0:
			ki = LAB.k1
		else:
			ki = LAB.boundary(lay_no-1).k2
		Ri = get_Ui_Vr(xs,ys,zs,c_lay_tm[0],ki,'j')
		if lay_no+1<len(RESULTS._c_all_tm):
			Ri += get_Ui_Vr(xs,ys,zs,c_lay_tm[1],ki,'h')
		R = where(III==lay_no,Ri,R)

	H = curl(R,dx,dy,dz)
	E = curl(H,dx,dy,dz)*(-1./1j)

	H = real(H)
	E = real(E)
	return H,E

#	print "scattered vector field"
#	Rsca = get_Ui_Vr(xs,ys,zs,RESULTS.c_sca_tm,LAB.k1,'h')
#	Hsca = curl(Rsca,dx,dy,dz)
#	Esca = array(curl(Hsca,dx,dy,dz))*(-1./1j)
#	Hsca = [real(Hsca[0])*I,real(Hsca[1])*I,real(Hsca[2])*I]
#	Esca = [real(Esca[0])*I,real(Esca[1])*I,real(Esca[2])*I]
#	Psca = vector_cross(Esca,Hsca)
#
#	print "incident vector field"
#	Rinc = get_Ui_Vr(xs,ys,zs,RESULTS._c_all_tm[0,0],LAB.k1,'j')
#	Hinc = curl(Rinc,dx,dy,dz)
#	Einc = array(curl(Hinc,dx,dy,dz))*(-1./1j)
#	Hinc = [real(Hinc[0])*I,real(Hinc[1])*I,real(Hinc[2])*I]
#	Einc = [real(Einc[0])*I,real(Einc[1])*I,real(Einc[2])*I]
#	Pinc = vector_cross(Einc,Hinc)
#
#	print "internal vector field"
#	Rint = get_Ui_Vr(xs,ys,zs,RESULTS._c_all_tm[1,0],LAB.boundary(0).k2,'j')
#	Hint = curl(Rint,dx,dy,dz)
#	Eint = array(curl(Hint,dx,dy,dz))*(-1./1j)
#	Hint = [real(Hint[0])*I1,real(Hint[1])*I1,real(Hint[2])*I1]
#	Eint = [real(Eint[0])*I1,real(Eint[1])*I1,real(Eint[2])*I1]
#	Pint = vector_cross(Eint,Hint)
#
#	Emax = max(sqrt(Einc[0]**2+Einc[1]**2+Einc[2]**2).reshape(len(xs)*len(ys)*len(zs)))*2
#	Hmax = max(sqrt(Hinc[0]**2+Hinc[1]**2+Hinc[2]**2).reshape(len(xs)*len(ys)*len(zs)))*2
#	Pmax = max(sqrt(Pinc[0]**2+Pinc[1]**2+Pinc[2]**2).reshape(len(xs)*len(ys)*len(zs)))*2
#	Hmin=Emin=Pmin = 0
#
#	return Hsca,Esca,Psca,Hinc,Einc,Pinc,Hint,Eint,Pint

def plot_vf(vf,xs,ys,zs,str,quiver_step=1,Fmin=None,Fmax=None,filename=None,title=None,interp=False,nsize=100,wavelen=None,lab=None):
	from scipy import interpolate
	Fx,Fy,Fz = vf
	qs = quiver_step
	if str == 'xz':
		n = (len(ys)-1)/2
		Fabs = sqrt(Fx**2+Fy**2+Fz**2)[:,n,:]
		F1 = Fx[:,n,:]
		F2 = Fz[:,n,:]
		x1=xs
		x2=zs
	elif str=='yz':
		n = (len(xs)-1)/2
		Fabs = sqrt(Fx**2+Fy**2+Fz**2)[n,:,:]
		F1 = Fy[n,:,:]
		F2 = Fz[n,:,:]
		x1=ys
		x2=zs
	if str == 'xy':
		n = (len(zs)-1)/2
		Fabs = sqrt(Fx**2+Fy**2+Fz**2)[:,:,n]
		F1 = Fx[:,:,n]
		F2 = Fy[:,:,n]
		x1=xs
		x2=ys

	if interp:
		spl_Fabs = interpolate.RectBivariateSpline(xs,zs,log10(Fabs))
		nx1 = linspace(min(x1),max(x1),nsize)
		nx2 = linspace(min(x2),max(x2),nsize)
		nFabs = spl_Fabs(nx1,nx2)
		X1,X2 = meshgrid(nx1,nx2)
	else:
		nFabs = log10(Fabs)
		X1,X2 = meshgrid(x1,x2)
	pylab.pcolor(X1,X2,nFabs.T,vmin=Fmin,vmax=Fmax)
	pylab.colorbar()
	X1,X2 = meshgrid(x1,x2)
	pylab.quiver(X1[::qs,::qs],X2[::qs,::qs],\
			(F1/Fabs)[::qs,::qs],(F2/Fabs)[::qs,::qs],\
			color='w')
	if lab:
		plot_particle(lab.particle,str,wavelen=wavelen)
	if title:
		pylab.title(title)
	if filename:
		pylab.savefig(filename)
	else:
		pylab.show()

def plot_vf_lic(F,xs,ys,zs,str,nsize=401,cmap='hot',ftype=None,Fmin=None,Fmax=None):
	"""Plot vector field using colored linear integral convolution"""
	from scikits import vectorplot as vp
	from scipy import interpolate
	global Hmax,Hmin,Emax,Emin,Pmax,Pmin
	if ftype == "H":
		Fmax = Hmax
		Fmin = Hmin
	elif ftype == "E":
		Fmax = Emax
		Fmin = Emin
	elif ftype == "P":
		Fmax = Pmax
		Fmin = Pmin
	else:
		if not Fmax:
			Fmax = None
		if not Fmin:
			Fmin = None
	n = (len(xs)-1)/2
	if str == 'xz':
		x1=xs
		x2=zs
		F1=F[0][:,n,:]
		F2=F[2][:,n,:]
	elif str == 'yz':
		x1=ys
		x2=zs
		F1=F[1][n,:,:]
		F2=F[2][n,:,:]
	elif str == 'xy':
		x1=xs
		x2=ys
		F1=F[0][:,:,n]
		F2=F[1][:,:,n]
	spl_F1 = interpolate.RectBivariateSpline(x1,x2,F1)
	spl_F2 = interpolate.RectBivariateSpline(x1,x2,F2)
	nx1 = linspace(min(x1),max(x1),nsize)
	nx2 = linspace(min(x2),max(x2),nsize)
	nF1 = spl_F1(nx1,nx2).T
	nF2 = spl_F2(nx1,nx2).T

	kernellen=31
	kernel = sin(arange(kernellen)*pi/kernellen)
	kernel = kernel.astype(float32)
	texture = random.rand(nsize,nsize).astype(float32)
	image = vp.line_integral_convolution(nF1.astype(float32), nF2.astype(float32), texture, kernel)

	imax = max(image.reshape(nsize**2))
	pylab.figimage(log10(sqrt(nF1**2+nF2**2))*(image+imax)/imax/2,cmap=cmap,vmin=Fmin,vmax=Fmax)
	pylab.show()

def curl(F,dx,dy,dz):
	Fx,Fy,Fz = F
	Fxx,Fxy,Fxz = gradient(Fx,dx,dy,dz)
	Fyx,Fyy,Fyz = gradient(Fy,dx,dy,dz)
	Fzx,Fzy,Fzz = gradient(Fz,dx,dy,dz)
	return array([ Fzy-Fyz , Fxz-Fzx, Fyx-Fxy ])

def vector_cross(A,B):
	Ax,Ay,Az = A
	Bx,By,Bz = B
	return array([ Ay*Bz-Az*By , Az*Bx-Ax*Bz , Ax*By-Ay*Bx ])

def plot_particle(particle,str,color='k',wavelen=None):
	for lay in particle.layers:
	  if str=='xz' or str=='yz':
		thetas = linspace(0,pi,200)
		r = lay.shape.R(thetas)[0]
		if wavelen:
			r = r*wavelen/(2*pi)
		pylab.plot( r*sin(thetas),r*cos(thetas),color)
		pylab.plot(-r*sin(thetas),r*cos(thetas),color)
	  elif str=='xy':
		phis = linspace(0,2*pi,400)
		r = lay.shape.R(pi/2)[0]
		if wavelen:
			r = r*wavelen/(2*pi)
		pylab.plot( r*sin(phis),r*cos(phis),color)
