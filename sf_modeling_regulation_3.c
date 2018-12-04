/*Acoustic wave equation,2D FD with 4-order space and 2-order time schemes.
surface-free boundary condition 
The first version come from Madagascar(Pengliang Yang) and then modified 
by Guiting Chen, Chinese Academy of Science, Institude of Geology and Geophyiscs.
*/
#include <rsf.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#include <time.h>
#endif
static int nb, nz, nx, nt, ng, ns, nzpad, nxpad;
static float dz, dx, dt, fm, amp, c0, c11, c12, c21, c22;
static float *bndr, *wlt;
static float **vv, **p0, **p1, **ptr = NULL, *trans, *dobs;
static int *sxz, *gxz;
void expand2d(float** b, float** a)
/*< expand domain of 'a' to 'b': source(a)-->destination(b) >*/
{
	int iz, ix;
	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			b[nb + ix][ iz] = a[ix][iz];
		}
	}

	for (ix = 0; ix<nxpad; ix++) {
		for (iz = 0; iz<nb; iz++) {
			//b[ix][iz] = b[ix][nb];
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


void window2d(float **a, float **b)
/*< window 'b' to 'a': source(b)-->destination(a) >*/
{
	int iz, ix;


	for (ix = 0; ix<nx; ix++) {
		for (iz = 0; iz<nz; iz++) {
			a[ix][iz] = b[nb + ix][ iz];
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
		//for (iz = 0; iz<nb; iz++) {	// top ABC			
		//	p0[ix][iz] = bndr[iz] * p0[ix][iz];
		//	p1[ix][iz] = bndr[iz] * p1[ix][iz];
		//}
		for (iz = nz ; iz<nzpad; iz++) {// bottom ABC			
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
			p[sx + nb][sz ] += source[is];
		}
	}
	else {
		for (is = 0; is<ns; is++) {
			sx = sxz[is] / nz;
			sz = sxz[is] % nz;
			p[sx + nb][sz ] -= source[is];
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
		seis_it[ig] = p[gx + nb][gz];
	}
}

int main(int argc, char* argv[])
{
	bool csdgather;
	int jt, ft, it, ib, ix, iz, sx, sz;
	float tmp;
	float **v0;
	int  is, kt, distx, distz, sxbeg, szbeg, gxbeg, gzbeg, jsx, jsz, jgx, jgz;
	sf_file Fv, shots;
	clock_t start, end;
	sf_init(argc, argv);
#ifdef _OPENMP
	omp_init();
#endif

	Fv = sf_input("in");/* veloctiy model */
	shots = sf_output("out");/* wavefield snaps */

	if (!sf_histint(Fv, "n1", &nz)) sf_error("No n1= in input");/* veloctiy model: nz */
	if (!sf_histint(Fv, "n2", &nx)) sf_error("No n2= in input");/* veloctiy model: nx */
	if (!sf_histfloat(Fv, "d1", &dz)) sf_error("No d1= in input");/* veloctiy model: dz */
	if (!sf_histfloat(Fv, "d2", &dx)) sf_error("No d2= in input");/* veloctiy model: dx */
	if (!sf_getint("nb", &nb)) nb = 30; /* thickness of sponge ABC */
	if (!sf_getint("nt", &nt)) sf_error("nt required");/* number of time steps */
	if (!sf_getfloat("dt", &dt)) sf_error("dt required");/* time sampling interval */
	if (!sf_getfloat("fm", &fm)) fm = 20.0; /*dominant freq of Ricker wavelet */
	if (!sf_getfloat("amp", &amp)) amp = 1000;
	if (!sf_getint("ns", &ns))   sf_error("no ns");
	if (!sf_getint("ng", &ng))   sf_error("no ng");
	if (!sf_getint("jsx", &jsx))   sf_error("no jsx");
	if (!sf_getint("jsz", &jsz))   jsz = 0;
	if (!sf_getint("jgx", &jgx))   jgx = 1;
	if (!sf_getint("jgz", &jgz))   jgz = 0;
	if (!sf_getint("sxbeg", &sxbeg))   sf_error("no sxbeg");
	if (!sf_getint("szbeg", &szbeg))   sf_error("no szbeg");
	if (!sf_getint("gxbeg", &gxbeg))   sf_error("no gxbeg");
	if (!sf_getint("gzbeg", &gzbeg))   sf_error("no gzbeg");
	if (!sf_getbool("csdgather", &csdgather)) csdgather = false;



	sf_putint(shots, "n1", nt);
	sf_putint(shots, "n2", ng);
	sf_putint(shots, "n3", ns);
	sf_putfloat(shots, "d1", dt);
	sf_putfloat(shots, "d2", jgx*dx);
	sf_putfloat(shots, "o1", 0);
	sf_putstring(shots, "label1", "Time");
	sf_putstring(shots, "label2", "Lateral");
	sf_putstring(shots, "label3", "Shot");
	sf_putstring(shots, "unit1", "sec");
	sf_putstring(shots, "unit2", "m");
	sf_putfloat(shots, "amp", amp);
	sf_putfloat(shots, "fm", fm);
	sf_putint(shots, "ng", ng);
	sf_putint(shots, "nb", nb);
	sf_putint(shots, "szbeg", szbeg);
	sf_putint(shots, "sxbeg", sxbeg);
	sf_putint(shots, "gzbeg", gzbeg);
	sf_putint(shots, "gxbeg", gxbeg);
	sf_putint(shots, "jsx", jsx);
	sf_putint(shots, "jsz", jsz);
	sf_putint(shots, "jgx", jgx);
	sf_putint(shots, "jgz", jgz);
	sf_putint(shots, "csdgather", csdgather ? 1 : 0);




	nzpad = nz + nb;
	nxpad = nx + 2 * nb;
	sx = nxpad / 2;
	sz = nzpad / 2;

	/*< initialize 4-th order fd coefficients >*/
	tmp = 1.0 / (dz*dz);
	c11 = 4.0*tmp / 3.0;
	c12 = -tmp / 12.0;
	tmp = 1.0 / (dx*dx);
	c21 = 4.0*tmp / 3.0;
	c22 = -tmp / 12.0;
	c0 = -2.0*(c11 + c12 + c21 + c22);

	wlt = sf_floatalloc(nt);
	bndr = sf_floatalloc(nb);
	v0 = sf_floatalloc2(nz, nx);
	vv = sf_floatalloc2(nzpad, nxpad);
	p0 = sf_floatalloc2(nzpad, nxpad);
	p1 = sf_floatalloc2(nzpad, nxpad);
	dobs = (float*)malloc(ng*nt * sizeof(float));
	trans = (float*)malloc(ng*nt * sizeof(float));
	sxz = (int*)malloc(ns * sizeof(int));
	gxz = (int*)malloc(ng * sizeof(int));
	//memset(bndr, 0, nt*(2 * nz + nx) * sizeof(float));
	memset(dobs, 0, ng*nt * sizeof(float));
	memset(trans, 0, ng*nt * sizeof(float));
	for (it = 0; it<nt; it++) {
		tmp = SF_PI*fm*(it*dt - 1.0 / fm); tmp *= tmp;
		wlt[it] = (1.0 - 2.0*tmp)*expf(-tmp);
	}
	for (ib = 0; ib<nb; ib++) {
		tmp = 0.015*(nb - ib);
		bndr[ib] = expf(-tmp*tmp);
	}
	sf_floatread(v0[0], nz*nx, Fv);
	expand2d(vv, v0);
	for (ix = 0; ix<nxpad; ix++) {
		for (iz = 0; iz<nzpad; iz++) {
			tmp = vv[ix][iz] * dt;
			vv[ix][iz] = tmp*tmp;
		}
	}
	memset(p0[0], 0, nzpad*nxpad * sizeof(float));
	memset(p1[0], 0, nzpad*nxpad * sizeof(float));
	if (!(sxbeg >= 0 && szbeg >= 0 && sxbeg + (ns - 1)*jsx<nx && szbeg + (ns - 1)*jsz<nz))
	{
		sf_error("sources exceeds the computing zone!\n"); exit(1);
	}
	sg_init(sxz, szbeg, sxbeg, jsz, jsx, ns, nz);
	distx = sxbeg - gxbeg;
	distz = szbeg - gzbeg;
	if (!(gxbeg >= 0 && gzbeg >= 0 && gxbeg + (ng - 1)*jgx<nx && gzbeg + (ng - 1)*jgz<nz))
	{
		printf("geophones exceeds the computing zone!\n"); exit(1);
	}
	if (csdgather) {
		if (!((sxbeg + (ns - 1)*jsx) + (ng - 1)*jgx - distx <nx && (szbeg + (ns - 1)*jsz) + (ng - 1)*jgz - distz <nz))
		{
			printf("geophones exceeds the computing zone!\n"); exit(1);
		}
	}
	sg_init(gxz, gzbeg, gxbeg, jgz, jgx, ng, nz);

	for (is = 0; is<ns; is++)
	{
		start = clock();

		if (csdgather) {/* reset position according to gather type */
			gxbeg = sxbeg + is*jsx - distx;
			sg_init(gxz, gzbeg, gxbeg, jgz, jgx, ng, nz);
		}
		memset(p0[0], 0, nzpad*nxpad * sizeof(float));
		memset(p1[0], 0, nzpad*nxpad * sizeof(float));
		/* forward modeling */
		for (it = 0; it<nt; it++)
		{
			add_source(p1, &wlt[it], &sxz[is], 1, nz, nb, true);
			step_forward(p0, p1);
			apply_sponge(p0, p1);
			ptr = p0; p0 = p1; p1 = ptr;
			record_seis(&dobs[it*ng], gxz, p0, ng, nz, nb);

		}
		/* save the modeled shot in trace-by-trace format */
		matrix_transpose(dobs, trans, ng, nt);
		sf_floatwrite(trans, ng*nt, shots);

		end = clock();
		sf_warning("shot %d finished: %f (s)", is + 1, ((float)(end - start)) / CLOCKS_PER_SEC);		
	}
	free(wlt);
	free(*v0); free(v0);
	free(*vv); free(vv);
	free(*p0); free(p0);
	free(*p1); free(p1);
	free(bndr);
	exit(0);
}
