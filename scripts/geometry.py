"""Closest points between a separated convex polytope and a solid cylinder.

Every closest polytope point belongs to a face, edge, or vertex. Face-interior
solutions use the cylinder support function; edge solutions minimize squared
point-to-cylinder distance on the closed segment, including its endpoints.
This enumerates the complete feature set, without sampling the curved surface.
The cylinder is centered at the origin with its axis along local z.
"""
from functools import lru_cache
import numpy as np


def project_cylinder(points, radius, half):
    projected = points.copy()
    rho = np.linalg.norm(points[:, :2], axis=1)
    scale = np.ones(len(points))
    np.divide(radius, rho, out=scale, where=rho > radius)
    projected[:, :2] *= scale[:, None]
    projected[:, 2] = np.clip(points[:, 2], -half, half)
    return projected


def separated_polytope_cylinder_distance(vertices, edges, faces, normals, radius, half):
    """Return distance and witnesses; the solid shapes must be separated."""
    vertices, normals = [np.ascontiguousarray(x, dtype=np.float64) for x in (vertices, normals)]
    edges, faces = [np.ascontiguousarray(x, dtype=np.int32) for x in (edges, faces)]
    result = np.empty(7, dtype=np.float64)
    ffi, native = _edge_kernel()
    native.polytope_cylinder(len(edges), len(faces), ffi.from_buffer('double[]', vertices),
        ffi.from_buffer('int[]', edges), ffi.from_buffer('int[]', faces),
        ffi.from_buffer('double[]', normals), radius, half, ffi.from_buffer('double[]', result))
    return result[0], result[1:4].copy(), result[4:7].copy()


def polytope_face_witnesses(u, direction, triangles, normals, closest):
    """Clip all edge/triangle-prism pairs in one float64 native batch.

    This is the same complete material-witness enumeration as the NumPy
    formulation. Triangulation, inclusion tolerances, and unique rows remain
    unchanged; compilation removes intermediate broadcast allocations only.
    """
    u, direction, triangles, normals = [np.ascontiguousarray(x, dtype=np.float64)
                                        for x in (u, direction, triangles, normals)]
    out = np.empty((2*len(u)*len(triangles), 7), dtype=np.float64)
    ffi, native = _edge_kernel()
    count = native.polytope_faces(len(u), len(triangles),
        *[ffi.from_buffer('double[]', x) for x in (u, direction, triangles, normals)],
        closest, ffi.from_buffer('double[]', out))
    return np.unique(out[:count], axis=0)


@lru_cache(maxsize=1)
def _edge_kernel():
    """Compile the scalar convex edge solve; no fast-math or lower precision."""
    import fcntl
    import os
    import tempfile
    from pathlib import Path
    from cffi import FFI
    ffi = FFI()
    ffi.cdef("""void cylinder_edges(int, const double*, const double*, double, double, double*);
        void polytope_cylinder(int, int, const double*, const int*, const int*, const double*, double, double, double*);
        int polytope_faces(int, int, const double*, const double*, const double*,
                           const double*, double, double*);""")
    source = r"""
    #include <math.h>
    static double dot3(const double* a, const double* b) {
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    }
    int polytope_faces(int ne, int nf, const double* u, const double* d,
                      const double* triangles, const double* normals,
                      double closest, double* out) {
        int count=0;
        for (int f=0; f<nf; ++f) {
            const double* tri=triangles+9*f;
            const double* n=normals+3*f;
            double inward[9], constants[3];
            for (int k=0; k<3; ++k) {
                double edge[3];
                for(int j=0;j<3;++j) edge[j]=tri[3*((k+1)%3)+j]-tri[3*k+j];
                double* v=inward+3*k;
                v[0]=n[1]*edge[2]-n[2]*edge[1];
                v[1]=n[2]*edge[0]-n[0]*edge[2];
                v[2]=n[0]*edge[1]-n[1]*edge[0];
                constants[k]=dot3(tri+3*k,v);
            }
            for (int e=0; e<ne; ++e) {
                const double* a=u+3*e;const double* b=d+3*e;
                double lo=0., hi=1.; int valid=1;
                for (int k=0;k<3;++k) {
                    double value=dot3(a,inward+3*k)-constants[k];
                    double slope=dot3(b,inward+3*k);
                    if (fabs(slope)>1e-15) {
                        double crossing=-value/slope;
                        if(slope>0.) lo=fmax(lo,crossing);else hi=fmin(hi,crossing);
                    } else if(value < -1e-12) valid=0;
                }
                if(!valid || lo>hi) continue;
                for(int endpoint=0;endpoint<2;++endpoint) {
                    double t=endpoint ? hi:lo, p[3], offset[3];
                    for(int j=0;j<3;++j) {p[j]=a[j]+t*b[j];offset[j]=p[j]-tri[j];}
                    double distance=dot3(offset,n);
                    if(distance<0. || distance>closest+1e-7) continue;
                    for(int j=0;j<3;++j) {out[7*count+j]=p[j];out[7*count+3+j]=n[j];}
                    out[7*count+6]=distance;++count;
                }
            }
        }
        return count;
    }
    static double derivative(const double* u, const double* d,
                             double a, double radius, double half) {
        double x=u[0]+a*d[0], y=u[1]+a*d[1], z=u[2]+a*d[2];
        double rho=sqrt(x*x+y*y), g=0.;
        if (rho > radius) g=((rho-radius)/rho)*(x*d[0]+y*d[1]);
        if (z > half) g+=(z-half)*d[2];
        else if (z < -half) g+=(z+half)*d[2];
        return g;
    }
    void cylinder_edges(int count, const double* u, const double* d,
                        double radius, double half, double* alpha) {
        for (int i=0; i<count; ++i, u+=3, d+=3) {
            if (derivative(u,d,0.,radius,half) >= 0.) { alpha[i]=0.; continue; }
            if (derivative(u,d,1.,radius,half) <= 0.) { alpha[i]=1.; continue; }
            double lo=0., hi=1.;
            // Monotone derivative of squared distance to a convex cylinder.
            // 52 bisections resolve fractions to double-precision accuracy.
            for (int k=0; k<52; ++k) {
                double mid=.5*(lo+hi);
                if (derivative(u,d,mid,radius,half) < 0.) lo=mid;
                else hi=mid;
            }
            alpha[i]=.5*(lo+hi);
        }
    }
    void polytope_cylinder(int ne, int nf, const double* vertices, const int* edges,
                          const int* faces, const double* normals,
                          double radius, double half, double* out) {
        out[0]=INFINITY;
        for(int i=0;i<ne;++i) {
            const double* a=vertices+3*edges[2*i];
            const double* b=vertices+3*edges[2*i+1];
            double d[3],alpha,p[3],c[3],offset[3];
            for(int j=0;j<3;++j) d[j]=b[j]-a[j];
            cylinder_edges(1,a,d,radius,half,&alpha);
            for(int j=0;j<3;++j) c[j]=p[j]=a[j]+alpha*d[j];
            double rho=sqrt(p[0]*p[0]+p[1]*p[1]);
            if(rho>radius) {c[0]*=radius/rho;c[1]*=radius/rho;}
            c[2]=fmax(-half,fmin(half,p[2]));
            for(int j=0;j<3;++j) offset[j]=p[j]-c[j];
            double distance=sqrt(dot3(offset,offset));
            if(distance<out[0]) {
                out[0]=distance;
                for(int j=0;j<3;++j){out[1+j]=p[j];out[4+j]=c[j];}
            }
        }
        for(int f=0;f<nf;++f) {
            const double* n=normals+3*f;
            const double* a=vertices+3*faces[3*f];
            double radial=sqrt(n[0]*n[0]+n[1]*n[1]);
            double support[3]={0.,0.,n[2]>0 ? -half:(n[2]<0 ? half:0.)};
            if(radial>0.){support[0]=-radius*n[0]/radial;support[1]=-radius*n[1]/radial;}
            double offset[3],projected[3];
            for(int j=0;j<3;++j)offset[j]=support[j]-a[j];
            double gap=dot3(offset,n);
            if(gap<0. || gap>=out[0])continue;
            for(int j=0;j<3;++j)projected[j]=support[j]-gap*n[j];
            int inside=1;
            for(int k=0;k<3;++k) {
                const double* u=vertices+3*faces[3*f+k];
                const double* v=vertices+3*faces[3*f+(k+1)%3];
                double edge[3],to[3],cross[3];
                for(int j=0;j<3;++j){edge[j]=v[j]-u[j];to[j]=projected[j]-u[j];}
                cross[0]=edge[1]*to[2]-edge[2]*to[1];
                cross[1]=edge[2]*to[0]-edge[0]*to[2];
                cross[2]=edge[0]*to[1]-edge[1]*to[0];
                if(dot3(cross,n)<0.) {inside=0;break;}
            }
            if(inside){out[0]=gap;for(int j=0;j<3;++j){out[1+j]=projected[j];out[4+j]=support[j];}}
        }
    }
    """
    cache = Path(tempfile.gettempdir())/('solverer_geometry_'+str(os.getuid()))
    cache.mkdir(exist_ok=True)
    # Multiple independent clips may initialize together. CFFI names compiled
    # modules by their source content; serialize only the first build/load.
    with (cache/'compile.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        native = ffi.verify(source, tmpdir=str(cache),
                            extra_compile_args=['-O3', '-fno-fast-math', '-ffp-contract=off'])
    return ffi, native
