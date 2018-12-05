/* A demo for illustrating how to update the model
*/
/*
Important references:
[1] Clayton, Robert, and Bj?rn Engquist. "Absorbing boundary
conditions for acoustic and elastic wave equations." Bulletin
of the Seismological Society of America 67.6 (1977): 1529-1540.
[2] Tarantola, Albert. "Inversion of seismic reflection data in the
acoustic approximation." Geophysics 49.8 (1984): 1259-1266.
[3] Pica, A., J. P. Diet, and A. Tarantola. "Nonlinear inversion
of seismic reflection data in a laterally invariant medium."
Geophysics 55.3 (1990): 284-292.
[4] Dussaud, E., Symes, W. W., Williamson, P., Lemaistre, L.,
Singer, P., Denel, B., & Cherrett, A. (2008). Computational
strategies for reverse-time migration. In SEG Technical Program
Expanded Abstracts 2008 (pp. 2267-2271).
[5] Hager, William W., and Hongchao Zhang. "A survey of nonlinear
conjugate gradient methods." Pacific journal of Optimization
2.1 (2006): 35-58.
*/

#include <rsf.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int nb, nz, nx, nt, ng, ns, nzpad, nxpad;
static float dz, dx, dt, fm, amp, c0, c11, c12, c21, c22;
static float *bndr, *wlt;
static float **vv, **sp0, **sp1, **gp0, **gp1, **ptr = NULL, *trans, *dobs;
static int *sxz, *gxz;


void expand2d(float** b, float** a)
/*< expand domain of 'a' to 'b': source(a)-->destination(b) >*/
{
	int iz, ix;

#ifdef _OPENMP
#pragma omp parallel for default(none)		\
    private(ix,iz)				\
    shared(b,a,nb,nz,nx)
#endif
	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			b[nb + ix][nb + iz] = a[ix][iz];
		}
	}

	for (ix = 0; ix<nxpad; ix++) {
		for (iz = 0; iz<nb; iz++) {
			b[ix][iz] = b[ix][nb];
			b[ix][nzpad - iz - 1] = b[ix][nzpad - nb - 1];
		}
	}

	for (ix = 0; ix<nb; ix++) {
		for (iz = 0; iz<nzpad; iz++) {
			b[ix][iz] = b[nb][iz];
			b[nxpad - ix - 1][iz] = b[nxpad - nb - 1][iz];
		}
	}
}

void random2d(float** b)
/*< expand domain of 'a' to 'b': source(a)-->destination(b) >*/
{
	int iz, ix;
	int a;
	//top and bottom
	for (ix = 0; ix<nxpad; ix++) {
		for (iz = 0; iz<nb; iz++) {
			a = (int)b[ix][iz];
			b[ix][iz] -= (float)(rand() % a) / nb*(nb - iz);
			a = (int)b[ix][nzpad - iz - 1];
			b[ix][nzpad - iz - 1] -= (float)(rand() % a) / nb*(nb - iz);
		}
	}

	for (ix = 0; ix<nb; ix++) {
		for (iz = 0; iz<nzpad; iz++) {
			a = (int)b[ix][iz];
			b[ix][iz] -= (float)(rand() % a) / nb*(nb - ix);
			a = (int)b[nxpad - ix - 1][iz];
			b[nxpad - ix - 1][iz] -= (float)(rand() % a) / nb*(nb - ix);

		}
	}
}


void window2d(float **a, float **b)
/*< window 'b' to 'a': source(b)-->destination(a) >*/
{
	int iz, ix;

#ifdef _OPENMP
#pragma omp parallel for default(none)		\
    private(ix,iz)				\
    shared(b,a,nb,nz,nx)
#endif
	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			a[ix][iz] = b[nb + ix][nb + iz];
		}
	}
}

void apply_sponge(float**p0, float **p1)
/*< apply absorbing boundary condition >*/
{
	int ix, iz;

#ifdef _OPENMP
#pragma omp parallel for			\
    private(ix,iz)				\
    shared(bndr,p0,p1)
#endif
	for (ix = 0; ix<nxpad; ix++)
	{
		for (iz = 0; iz<nb; iz++) {	// top ABC			
			p0[ix][iz] = bndr[iz] * p0[ix][iz];
			p1[ix][iz] = bndr[iz] * p1[ix][iz];
		}
		for (iz = nz + nb; iz<nzpad; iz++) {// bottom ABC			
			p0[ix][iz] = bndr[nzpad - iz - 1] * p0[ix][iz];
			p1[ix][iz] = bndr[nzpad - iz - 1] * p1[ix][iz];
		}
	}

#ifdef _OPENMP
#pragma omp parallel for			\
    private(ix,iz)				\
    shared(bndr,p0,p1)
#endif
	for (iz = 0; iz<nzpad; iz++)
	{
		for (ix = 0; ix<nb; ix++) {	// left ABC			
			p0[ix][iz] = bndr[ix] * p0[ix][iz];
			p1[ix][iz] = bndr[ix] * p1[ix][iz];
		}
		for (ix = nx + nb; ix<nxpad; ix++) {// right ABC			
			p0[ix][iz] = bndr[nxpad - ix - 1] * p0[ix][iz];
			p1[ix][iz] = bndr[nxpad - ix - 1] * p1[ix][iz];
		}
	}
}


void step_forward(float **p0, float **p1)
/*< forward modeling step >*/
{
	int ix, iz;
	float tmp;

#ifdef _OPENMP
#pragma omp parallel for				\
    private(ix,iz,tmp)					\
    shared(p0,p1,vv,c0,c11,c12,c21,c22,nzpad,nxpad)
#endif	
	for (ix = 2; ix < nxpad - 2; ix++)
		for (iz = 2; iz < nzpad - 2; iz++)
		{
			tmp = c0*p1[ix][iz] +
				c11*(p1[ix][iz - 1] + p1[ix][iz + 1]) +
				c12*(p1[ix][iz - 2] + p1[ix][iz + 2]) +
				c21*(p1[ix - 1][iz] + p1[ix + 1][iz]) +
				c22*(p1[ix - 2][iz] + p1[ix + 2][iz]);
			p0[ix][iz] = 2 * p1[ix][iz] - p0[ix][iz] + vv[ix][iz] * tmp;
		}
}

void matrix_transpose(float *matrix, float *trans, int n1, int n2)
/*< matrix transpose: matrix tansposed to be trans >*/
{
	int i1, i2;

	for (i2 = 0; i2<n2; i2++)
		for (i1 = 0; i1<n1; i1++)
			trans[i2 + n2*i1] = matrix[i1 + n1*i2];
}

void sg_init(int *sxz, int szbeg, int sxbeg, int jsz, int jsx, int ns, int nz)
/*< shot/geophone position initialize >*/
{
	int is, sz, sx;
	for (is = 0; is<ns; is++)
	{
		sz = szbeg + is*jsz;
		sx = sxbeg + is*jsx;
		sxz[is] = sz + nz*sx;
	}
}

void add_source(float **p, float *source, int *sxz, int ns, int nz, int nb, bool add)
/*< add/subtract seismic sources >*/
{
	int is, sx, sz;
	if (add) {
		for (is = 0; is<ns; is++) {
			sx = sxz[is] / nz;
			sz = sxz[is] % nz;
			p[sx + nb][sz + nb] += source[is];
		}
	}
	else {
		for (is = 0; is<ns; is++) {
			sx = sxz[is] / nz;
			sz = sxz[is] % nz;
			p[sx + nb][sz + nb] -= source[is];
		}
	}
}

void record_seis(float *seis_it, int *gxz, float **p, int ng, int nz, int nb)
/*< record seismogram at time it into a vector length of ng >*/
{
	int ig, gx, gz;
	for (ig = 0; ig<ng; ig++)
	{
		gx = gxz[ig] / nz;
		gz = gxz[ig] % nz;
		seis_it[ig] = p[gx + nb][gz + nb];
	}
}




void step_backward(float **illum, float **lap, float **p0, float **p1)
/*< step backward >*/
{
	int ix, iz;
	float tmp;
	float  diff1, diff2;
#ifdef _OPENMP
#pragma omp parallel for				\
    private(ix,iz,tmp)					\
    shared(p0,p1,vv,c0,c11,c12,c21,c22,nzpad,nxpad)
#endif	
	for (ix = 2; ix < nxpad - 2; ix++)
		for (iz = 2; iz < nzpad - 2; iz++)
		{
			tmp = c0*p1[ix][iz] +
				c11*(p1[ix][iz - 1] + p1[ix][iz + 1]) +
				c12*(p1[ix][iz - 2] + p1[ix][iz + 2]) +
				c21*(p1[ix - 1][iz] + p1[ix + 1][iz]) +
				c22*(p1[ix - 2][iz] + p1[ix + 2][iz]);
			diff1 = diff2 = -2.0*p1[ix][iz];
			diff1 += (iz - 1 >= 0) ? p1[ix][iz - 1] : 0.0;
			diff1 += (iz + 1<nzpad) ? p1[ix][iz + 1] : 0.0;
			diff2 += (ix - 1 >= 0) ? p1[ix - 1][iz] : 0.0;
			diff2 += (ix + 1<nxpad) ? p1[ix + 1][iz] : 0.0;
			lap[ix][iz] = diff1 + diff2;
			illum[ix][iz] += p1[ix][iz] * p1[ix][iz];
			p0[ix][iz] = 2 * p1[ix][iz] - p0[ix][iz] + vv[ix][iz] * tmp;
		}
}



void rw_bndr(float *bndr, float **p, int nz, int nx, int nb, int order, bool write)
/*< if write==true, write/save boundaries out of variables;
else  read boundaries into variables (for 2nd order FD) >*/
{
	int i,ix,iz;
	if (write) {
		// top and bottom
		for (ix = 0; ix<nx; ix++)
		{
			for (iz = 0; iz<order; iz++) {
				bndr[iz + order * 2 * ix] = p[ix + nb][iz + nb];  // top 		
				bndr[iz + order + order * 2 * ix] = p[ix + nb][nz + nb - order + iz];  // bottom 	
			}
		}
		// left and right
		for (ix = 0; iz<order; ix++)
		{
			for (iz = 0; iz<nz; iz++) {
				bndr[nx * 2 * order + iz + ix*nz] = p[ix + nb][iz + nb];  // left
				bndr[nx * 2 * order + iz + (ix + order)*nz] = p[nb + nx - order + ix][iz + nb];  // right	
			}

		}
	}
	else {
		for (ix = 0; ix<nx; ix++)
		{
			for (iz = 0; iz<order; iz++) {
				p[ix + nb][iz + nb] = bndr[iz + order * 2 * ix];  // top 		
				p[ix + nb][nz + nb - order + iz] = bndr[iz + order + order * 2 * ix];  // bottom 	
			}
		}
		// left and right
		for (ix = 0; iz<order; ix++)
		{
			for (iz = 0; iz<nz; iz++) {
				p[ix + nb][iz + nb] = bndr[nx * 2 * order + iz + ix*nz];   // left
				p[nb + nx - order + ix][iz + nb] = bndr[nx * 2 * order + iz + (ix + order)*nz];  // right	
			}

		}

	}
}

void cal_residuals(float *dcal, float *dobs, float *dres, int ng)
/*< calculate residual >*/
{
	int ig;
	for (ig = 0; ig<ng; ig++) {
		dres[ig] = dcal[ig] - dobs[ig];
	}
}

void cal_gradient(float **grad, float **lap, float **gp)
/*< calculate gradient >*/
{
	int ix, iz;
	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			grad[ix][iz] += lap[ix + nb][iz + nb] * gp[ix + nb][iz + nb];
		}
	}
}

void scale_gradient(float **grad, float **vv, float **illum, int nz, int nx, bool precon)
/*< scale gradient >*/
{
	int ix, iz;
	float a;
	for (ix = 1; ix<nx - 1; ix++) {
		for (iz = 1; iz<nz - 1; iz++) {
			a = vv[ix + nb][iz + nb];
			if (precon) a *= sqrtf(illum[ix + nb][iz + nb] + SF_EPS);/*precondition with residual wavefield illumination*/
			grad[ix][iz] *= 2.0 / a;
		}
	}

	for (ix = 0; ix<nx; ix++) {
		grad[ix][0] = grad[ix][1];
		grad[ix][nz - 1] = grad[ix][nz - 2];
	}

	for (iz = 0; iz<nz; iz++) {
		grad[0][iz] = grad[1][iz];
		grad[nx - 1][iz] = grad[nx - 2][iz];
	}
}

float cal_objective(float *dres, int ng)
/*< calculate the value of objective function >*/
{
	int i;
	float a, obj = 0;

	for (i = 0; i<ng; i++) {
		a = dres[i];
		obj += a*a;
	}
	return obj;
}

float cal_beta(float **g0, float **g1, float **cg, int nz, int nx)
/*< calculate beta >*/
{
	int ix, iz;
	float a, b, c;

	a = b = c = 0;
	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			a += g1[ix][iz] * (g1[ix][iz] - g0[ix][iz]);// numerator of HS
			b += cg[ix][iz] * (g1[ix][iz] - g0[ix][iz]);// denominator of HS,DY
			c += g1[ix][iz] * g1[ix][iz];		// numerator of DY
		}
	}

	float	beta_HS = (fabsf(b)>0) ? (a / b) : 0.0;
	float beta_DY = (fabsf(b)>0) ? (c / b) : 0.0;
	return SF_MAX(0.0, SF_MIN(beta_HS, beta_DY));
}


void cal_conjgrad(float **g1, float **cg, float beta, int nz, int nx)
/*< calculate conjugate gradient >*/
{
	int ix, iz;

	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			cg[ix][iz] = -g1[ix][iz] + beta*cg[ix][iz];
		}
	}
}

float cal_epsilon(float **vv, float **cg, int nz, int nx)
/*< calculate epsilcon >*/
{
	int ix, iz;
	float vvmax, cgmax;
	vvmax = cgmax = 0.0;

	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			vvmax = SF_MAX(vvmax, fabsf(vv[ix][iz]));
			cgmax = SF_MAX(cgmax, fabsf(cg[ix][iz]));
		}
	}

	return 0.01*vvmax / (cgmax + SF_EPS);
}

void cal_vtmp(float **vtmp, float **v0, float **cg, float epsil, int nz, int nx)
/*< calculate temporary velcity >*/
{
	int ix, iz;

	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			vtmp[ix][iz] = v0[ix ][iz ] + epsil*cg[ix][iz];
		}
	}
}

void sum_alpha12(float *alpha1, float *alpha2, float *dcaltmp, float *dobs, float *derr, int ng)
/*< calculate numerator and denominator of alpha >*/
{
	int ig;
	float a, b, c;
	for (ig = 0; ig<ng; ig++) {
		c = derr[ig];
		a = dobs[ig] + c;/* since f(mk)-dobs[id]=derr[id], thus f(mk)=b+c; */
		b = dcaltmp[ig] - a;/* f(mk+epsil*cg)-f(mk) */
		alpha1[ig] -= b*c;
		alpha2[ig] += b*b;
	}
}


float cal_alpha(float *alpha1, float *alpha2, float epsil, int ng)
/*< calculate alpha >*/
{
	int ig;
	float a, b;

	a = b = 0;
	for (ig = 0; ig<ng; ig++) {
		a += alpha1[ig];
		b += alpha2[ig];
	}

	return (a*epsil / (b + SF_EPS));
}

void update_vel(float **vv, float **cg, float alpha, int nz, int nx)
/*< update velcity >*/
{
	int ix, iz;

	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			vv[ix][iz] += alpha*cg[ix][iz];
		}
	}
}

void bell_smoothz(float **g, float **smg, int rbell, int nz, int nx)
/*< gaussian bell smoothing for z-axis >*/
{
	int ix, iz, i;
	float s;

	for (ix = 0; ix<nx; ix++)
		for (iz = 0; iz<nz; iz++)
		{
			s = 0.0;
			for (i = -rbell; i <= rbell; i++) if (iz + i >= 0 && iz + i<nz) s += expf(-(2.0*i*i) / rbell)*g[ix][iz + i];
			smg[ix][iz] = s;
		}
}

void bell_smoothx(float **g, float **smg, int rbell, int nz, int nx)
/*< gaussian bell smoothing for x-axis >*/
{
	int ix, iz, i;
	float s;

	for (ix = 0; ix<nx; ix++)
		for (iz = 0; iz<nz; iz++)
		{
			s = 0.0;
			for (i = -rbell; i <= rbell; i++) if (ix + i >= 0 && ix + i<nx) s += expf(-(2.0*i*i) / rbell)*g[ix + i][iz];
			smg[ix][iz] = s;
		}
}


int main(int argc, char *argv[])
{
	/* variables on host */
	bool verb, precon, csdgather;
	int is, it, iter, niter, distx, distz, csd, rbell,ix,iz;
	int sxbeg, szbeg, gxbeg, gzbeg, jsx, jsz, jgx, jgz;/*  parameters of acquisition geometery */
	float  tmp,  obj1, obj, beta, epsil, alpha;
	float  *dcal, *derr, *brw, *objval;
	float  **illum, **lap, **vtmp, **g0, **g1, **cg, **cg0,*alpha1, *alpha2;
	float **v0;
	int order,ib;
	clock_t start, stop;/* timer */
	sf_file vinit, shots, vupdates, fg0, fg1, fcg0,fcg1, objs,  fobjs;/* I/O files */

	/* initialize Madagascar */
	sf_init(argc, argv);

	/* set up I/O files */
	vinit = sf_input("in");   /* initial velocity model, unit=m/s */
	shots = sf_input("shots"); /* recorded shots from exact velocity model */
	vupdates = sf_output("out"); /* updated velocity in iterations */
	fg0 = sf_input("g0");  /* gradient in iterations */
	fg1 = sf_input("g1");  /* gradient in iterations */
	fcg0 = sf_input("cg0");  /* gradient in iterations */
	fcg1 = sf_output("cg1");  /* gradient in iterations */



	fobjs = sf_input("obj");  /* gradient in iterations */
	//illums = sf_output("illums");/* source illumination in iterations */
	//objs = sf_output("objs");/* values of objective function in iterations */

	/* get parameters from velocity model and recorded shots */
	if (!sf_getbool("verb", &verb)) verb = true;/* vebosity */
	if (!sf_histint(vinit, "n1", &nz)) sf_error("no n1");/* nz */
	if (!sf_histint(vinit, "n2", &nx)) sf_error("no n2");/* nx */
	if (!sf_histfloat(vinit, "d1", &dz)) sf_error("no d1");/* dz */
	if (!sf_histfloat(vinit, "d2", &dx)) sf_error("no d2");/* dx */
	if (!sf_getbool("precon", &precon)) precon = false;/* precondition or not */
	if (!sf_getint("niter", &niter))   niter = 1;	/* number of iterations */
	if (!sf_getint("rbell", &rbell))	  rbell = 2;	/* radius of bell smooth */
	
	if (!sf_getint("iter", &iter))	  iter = 0;	/* radius of bell smooth */
	if (!sf_histint(shots, "n1", &nt)) sf_error("no nt");
	/* total modeling time steps */
	if (!sf_histint(shots, "n2", &ng)) sf_error("no ng");
	/* total receivers in each shot */
	if (!sf_histint(shots, "nb", &nb)) sf_error("no nb");
	if (!sf_histint(shots, "n3", &ns)) sf_error("no ns");
	/* number of shots */
	if (!sf_histfloat(shots, "d1", &dt)) sf_error("no dt");
	/* time sampling interval */
	if (!sf_histfloat(shots, "amp", &amp)) sf_error("no amp");
	/* maximum amplitude of ricker */
	if (!sf_histfloat(shots, "fm", &fm)) sf_error("no fm");
	/* dominant freq of ricker */
	if (!sf_histint(shots, "sxbeg", &sxbeg)) sf_error("no sxbeg");
	/* x-begining index of sources, starting from 0 */
	if (!sf_histint(shots, "szbeg", &szbeg)) sf_error("no szbeg");
	/* x-begining index of sources, starting from 0 */
	if (!sf_histint(shots, "gxbeg", &gxbeg)) sf_error("no gxbeg");
	/* x-begining index of receivers, starting from 0 */
	if (!sf_histint(shots, "gzbeg", &gzbeg)) sf_error("no gzbeg");
	/* x-begining index of receivers, starting from 0 */
	if (!sf_histint(shots, "jsx", &jsx)) sf_error("no jsx");
	/* source x-axis  jump interval  */
	if (!sf_histint(shots, "jsz", &jsz)) sf_error("no jsz");
	/* source z-axis jump interval  */
	if (!sf_histint(shots, "jgx", &jgx)) sf_error("no jgx");
	/* receiver x-axis jump interval  */
	if (!sf_histint(shots, "jgz", &jgz)) sf_error("no jgz");
	/* receiver z-axis jump interval  */
	if (!sf_histint(shots, "csdgather", &csd)) sf_error("csdgather or not required");
	/* default, common shot-gather; if n, record at every point*/
	nzpad = nz + 2 * nb;
	nxpad = nx + 2 * nb;

	sf_putint(vupdates, "n1", nz);
	sf_putint(vupdates, "n2", nx);
	sf_putfloat(vupdates, "d1", dz);
	sf_putfloat(vupdates, "d2", dx);
	sf_putstring(vupdates, "label1", "Depth");
	sf_putstring(vupdates, "label2", "Distance");
	sf_putstring(vupdates, "label3", "Iteration");
	sf_putint(vupdates, "n3", niter);
	sf_putint(vupdates, "d3", 1);
	sf_putint(vupdates, "o3", 1);
	sf_putint(fcg1, "n1", nz);
	sf_putint(fcg1, "n2", nx);
	sf_putint(fcg1, "n3", niter);
	sf_putfloat(fcg1, "d1", dz);
	sf_putfloat(fcg1, "d2", dx);
	sf_putint(fcg1, "d3", 1);
	sf_putint(fcg1, "o3", 1);
	sf_putstring(fcg1, "label1", "Depth");
	sf_putstring(fcg1, "label2", "Distance");
	sf_putstring(fcg1, "label3", "Iteration");
	sf_putint(fcg1, "n1", nz);
	sf_putint(fcg1, "n2", nx);
	sf_putint(fcg1, "n3", niter);
	sf_putfloat(fcg1, "d1", dz);
	sf_putfloat(fcg1, "d2", dx);
	sf_putint(fcg1, "d3", 1);
	sf_putint(fcg1, "o3", 1);
	sf_putstring(fcg1, "label1", "Depth");
	sf_putstring(fcg1, "label2", "Distance");
	sf_putstring(fcg1, "label3", "Iteration");



	csdgather = (csd>0) ? true : false;
	tmp = 1.0 / (dz*dz);
	c11 = 4.0*tmp / 3.0;
	c12 = -tmp / 12.0;
	tmp = 1.0 / (dx*dx);
	c21 = 4.0*tmp / 3.0;
	c22 = -tmp / 12.0;
	c0 = -2.0*(c11 + c12 + c21 + c22);

	vv = sf_floatalloc2(nzpad, nxpad);/* updated velocity */
	v0 = sf_floatalloc2(nz, nx);/* temporary velocity computed with epsil */
	vtmp = sf_floatalloc2(nz, nx);/* temporary velocity computed with epsil */
	sp0 = sf_floatalloc2(nzpad, nxpad);/* source wavefield p0 */
	sp1 = sf_floatalloc2(nzpad, nxpad);/* source wavefield p1 */
	gp0 = sf_floatalloc2(nzpad, nxpad);/* geophone/receiver wavefield p0 */
	gp1 = sf_floatalloc2(nzpad, nxpad);/* geophone/receiver wavefield p1 */
	bndr = sf_floatalloc(nb);
	g0 = sf_floatalloc2(nz, nx);/* gradient at previous step */
	g1 = sf_floatalloc2(nz, nx);/* gradient at curret step */
	cg = sf_floatalloc2(nz, nx);/* conjugate gradient */
	cg0 = sf_floatalloc2(nz, nx);/* conjugate gradient */
	lap = sf_floatalloc2(nzpad, nxpad);/* laplace of the source wavefield */
	illum = sf_floatalloc2(nzpad, nxpad);/* illumination of the source wavefield */
	objval = (float*)malloc(niter*sizeof(float));/* objective/misfit function */
	wlt = (float*)malloc(nt*sizeof(float));/* ricker wavelet */
	sxz = (int*)malloc(ns*sizeof(int)); /* source positions */
	gxz = (int*)malloc(ng*sizeof(int)); /* geophone positions */
	//brw = (float*)malloc(nt * 2 * order*(nz + nx) * sizeof(float));/* boundaries for wavefield reconstruction */
	trans = (float*)malloc(ng*nt*sizeof(float));/* transposed one shot */
	dobs = (float*)malloc(ng*nt*sizeof(float));/* observed seismic data */
	dcal = (float*)malloc(ng*sizeof(float));	/* calculated/synthetic seismic data */
	derr = (float*)malloc(ns*ng*nt*sizeof(float));/* residual/error between synthetic and observation */
	alpha1 = (float*)malloc(ng*sizeof(float));/* numerator of alpha, length=ng */
	alpha2 = (float*)malloc(ng*sizeof(float));/* denominator of alpha, length=ng */

	/* initialize varibles */
	for (it = 0; it<nt; it++) {
		tmp = SF_PI*fm*(it*dt - 1.0 / fm); tmp *= tmp;
		wlt[it] = (1.0 - 2.0*tmp)*expf(-tmp);
	}
	for (ib = 0; ib<nb; ib++) {
		tmp = 0.015*(nb - ib);
		bndr[ib] = expf(-tmp*tmp);
	}
	/* initialize varibles */
	sf_floatread(v0[0], nz*nx, vinit);

	memset(sp0[0], 0, nzpad*nxpad * sizeof(float));
	memset(sp1[0], 0, nzpad*nxpad * sizeof(float));
	
	memset(gp0[0], 0, nzpad*nxpad * sizeof(float));
	memset(gp1[0], 0, nzpad*nxpad * sizeof(float));
	
	memset(g0[0], 0, nz*nx * sizeof(float));
	memset(g1[0], 0, nz*nx * sizeof(float));
	memset(cg[0], 0, nz*nx * sizeof(float));
	memset(lap[0], 0, nzpad*nxpad * sizeof(float));
	memset(vtmp[0], 0, nz*nx * sizeof(float));
	memset(illum[0], 0, nzpad*nxpad * sizeof(float));


	if (!(sxbeg >= 0 && szbeg >= 0 && sxbeg + (ns - 1)*jsx<nx && szbeg + (ns - 1)*jsz<nz))
	{
		sf_warning("sources exceeds the computing zone!\n"); exit(1);
	}
	sg_init(sxz, szbeg, sxbeg, jsz, jsx, ns, nz);
	distx = sxbeg - gxbeg;
	distz = szbeg - gzbeg;
	if (csdgather)	{
		if (!(gxbeg >= 0 && gzbeg >= 0 && gxbeg + (ng - 1)*jgx<nx && gzbeg + (ng - 1)*jgz<nz &&
			(sxbeg + (ns - 1)*jsx) + (ng - 1)*jgx - distx <nx && (szbeg + (ns - 1)*jsz) + (ng - 1)*jgz - distz <nz))
		{
			sf_warning("geophones exceeds the computing zone!\n"); exit(1);
		}
	}
	else{
		if (!(gxbeg >= 0 && gzbeg >= 0 && gxbeg + (ng - 1)*jgx<nx && gzbeg + (ng - 1)*jgz<nz))
		{
			sf_warning("geophones exceeds the computing zone!\n"); exit(1);
		}
	}
	sg_init(gxz, gzbeg, gxbeg, jgz, jgx, ng, nz);
	//memset(brw, 0, nt * 2 * order*(nz + nx) * sizeof(float));
	memset(dobs, 0, ng*nt*sizeof(float));
	memset(dcal, 0, ng*sizeof(float));
	memset(derr, 0, ns*ng*nt*sizeof(float));
	memset(alpha1, 0, ng*sizeof(float));
	memset(alpha2, 0, ng*sizeof(float));
	memset(dobs, 0, ng*nt*sizeof(float));
	memset(objval, 0, niter*sizeof(float));

	int i, j;
	FILE *fp_err;
	fp_err = fopen("errors.bin", "rb");
	for (i = 0; i < ns; i++)
	{
		//sf_floatread(&derr[i*ng*nt], ng*nt, errors);
		fread(&derr[i*ng*nt],sizeof(float), ng*nt, fp_err);
		//matrix_transpose(trans, dobs, nt, ng);
		//memcpy(&derr[i*ng*nt], trans, nx*nz*sizeof(float));
	}

	sf_floatread(cg[0], nx*nz, fcg0);
	sf_floatread(g0[0], nx*nz, fg0);
	sf_floatread(g1[0], nx*nz, fg1);
	sf_floatread(&objval[0], 1, fobjs);

	//for (iter = 0; iter<niter; iter++)
	{

		if (verb) {
			start = clock();// record starting time
			//sf_warning("iter=%d", iter);
		}
		sf_seek(shots, 0L, SEEK_SET);

		if (iter>0) beta = cal_beta(g0, g1, cg, nz, nx); else beta = 0.0;
		cal_conjgrad(g1, cg, beta, nz, nx);
		epsil = cal_epsilon(v0, cg, nz, nx);

		sf_seek(shots, 0L, SEEK_SET);
		memset(alpha1, 0, ng*sizeof(float));
		memset(alpha2, 0, ng*sizeof(float));

		cal_vtmp(vtmp, v0, cg, epsil, nz, nx);
		expand2d(vv, vtmp);

		for (ix = 0; ix<nxpad; ix++) {
			for (iz = 0; iz<nzpad; iz++) {
				tmp = vv[ix][iz] * dt;
				vv[ix][iz] = tmp*tmp;
			}
		}

		for (is = 0; is<ns; is++)
		{
			sf_floatread(trans, ng*nt, shots);
			matrix_transpose(trans, dobs, nt, ng);
			if (csdgather)	{
				gxbeg = sxbeg + is*jsx - distx;
				sg_init(gxz, gzbeg, gxbeg, jgz, jgx, ng, nz);
			}
			memset(sp0[0], 0, nzpad*nxpad*sizeof(float));
			memset(sp1[0], 0, nzpad*nxpad*sizeof(float));
			for (it = 0; it<nt; it++)
			{
				add_source(sp1, &wlt[it], &sxz[is], 1, nz,nb, true);
				step_forward(sp0, sp1);
				apply_sponge(sp0, sp1);
				ptr = sp0; sp0 = sp1; sp1 = ptr;

				record_seis(dcal, gxz, sp0, ng, nz,nb);
				sum_alpha12(alpha1, alpha2, dcal, &dobs[it*ng], &derr[is*ng*nt + it*ng], ng);
			}
		}

		alpha = cal_alpha(alpha1, alpha2, epsil, ng);
		update_vel(v0, cg, alpha, nz, nx);

		sf_floatwrite(v0[0], nz*nx, vupdates);
		sf_floatwrite(cg[0], nx*nz, fcg1);
		if (verb) {// output important information at each FWI iteration 
                        stop = clock();// record ending time 
			sf_warning("iter= %d objs= %f beta=%f  epsil=%f  alpha=%f alpha1= %f alpha2=%f finished: %f (s) \n \n \n \n",iter, objval[0],beta, epsil, alpha,*alpha1, *alpha2,((float)(stop - start)) / CLOCKS_PER_SEC);
			
		}
	}
	free(*vv); free(vv);
	free(*vtmp); free(vtmp);
	free(*sp0); free(sp0);
	free(*sp1); free(sp1);
	
	free(*gp0); free(gp0);
	free(*gp1); free(gp1);
	
	free(*g0); free(g0);
	free(*g1); free(g1);
	free(*cg); free(cg);
	free(*lap); free(lap);
	free(*illum); free(illum);
	free(objval);
	free(wlt);
	free(sxz);
	free(gxz);
	free(bndr);
	free(trans);
	free(dobs);
	free(dcal);
	free(derr);
	free(alpha1);
	free(alpha2);

	exit(0);
}
