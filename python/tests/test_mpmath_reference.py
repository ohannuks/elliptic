"""Reference tests of every public function against mpmath (30-digit arithmetic).

One test per function family over parameter grids that include the regimes
repaired for 4.2.0, plus one dedicated test per 4.2.0 fix:

ellipj         sn, cn, dn = mpmath.ellipfun; am = int_0^u dn (continuous amplitude,
               am(u+2K) = am(u)+pi, checked beyond one period)
elliptic12     F, E = mpmath.ellipf/ellipe on the whole real line; Z = E - (E/K) F
elliptic3      Pi(n; phi|m) = mpmath.ellippi, phases past pi/2
ellipticBD     B, D by quadrature, S = (D-B)/m including the small-m series branch
ellipticBDJ    B, D, J by quadrature, phases reduced mod pi
jacobiEDJ      E_u, D_u, J_u by quadrature in the Jacobi argument
carlsonR*      mpmath.elliprf/elliprd/elliprj/elliprc
cel, cel1-3    Bulirsch integral by quadrature; cel1 = K(1-kc^2) down to kc = 1e-9
weierstrass    P by the sn form; P' by differentiation; zeta, sigma by the theta
               forms (DLMF 23.6.8/9/13), sigma tested past 2*omega1
theta          theta_j, theta_j' and Jacobi Theta/H = mpmath.jtheta, large angles
elliptic12i    complex phase, including Re(u) = pi/2
ellipji        complex argument = mpmath.ellipfun
nomeq          mpmath.qfrom; inversenomeq = mpmath.mfrom up to q ~ 0.778
inverselliptic2 forward mpmath.ellipe then inverse
agm            mpmath.agm; arclength_ellipse by quadrature
edge inputs    NaN propagates, empty arrays stay empty
"""
import numpy as np
import mpmath as mp
import pytest

import elliptic

mp.mp.dps = 30
PI = mp.pi

M_GRID = [1e-10, 1e-4, 0.1, 0.5, 0.9, 0.999, 1 - 1e-9]
RTOL = 1e-12          # moderate parameters
RTOL_EXTREME = 2e-10  # m within 1e-9 of 1 or below 1e-8


def _f(v):
    return float(np.asarray(v).flat[0])


def _c(v):
    return complex(np.asarray(v).flat[0])


def _tol(m):
    return RTOL_EXTREME if (m > 1 - 1e-6 or m < 1e-8) else RTOL


def _close(got, ref, m=0.5, atol=1e-14):
    ref = float(ref) if not isinstance(ref, complex) else ref
    np.testing.assert_allclose(got, ref, rtol=_tol(m), atol=atol)


# --------------------------------------------------------------------------
# ellipj
# --------------------------------------------------------------------------

class TestEllipj:
    U = [-13.7, -3.0, 0.3, 2.5, 7.3, 21.9]

    @pytest.mark.parametrize("m", M_GRID)
    @pytest.mark.parametrize("u", U)
    def test_sn_cn_dn(self, u, m):
        sn, cn, dn, _ = elliptic.ellipj(u, m)
        for got, name in ((sn, "sn"), (cn, "cn"), (dn, "dn")):
            _close(_f(got), mp.ellipfun(name, u, m=m), m)

    @pytest.mark.parametrize("m", M_GRID)
    @pytest.mark.parametrize("u", U)
    def test_am_is_continuous(self, u, m):
        """4.2.0: am(u|m) is the continuous amplitude int_0^u dn, not its principal value."""
        _, _, _, am = elliptic.ellipj(u, m)
        K = mp.ellipk(m); s = 1 if u > 0 else -1
        pts = [s * p for p in [mp.mpf(0)] + [K * k for k in range(1, int(abs(u) / K) + 1)] + [mp.mpf(abs(u))]]
        ref = mp.quad(lambda t: mp.ellipfun("dn", t, m=m), pts)
        _close(_f(am), ref, m, atol=1e-13)

    @pytest.mark.parametrize("m", [0.3, 0.9, 0.999])
    def test_am_quasi_period(self, m):
        K = float(mp.ellipk(m))
        for u in (0.4, 3.3, 11.0):
            a0 = _f(elliptic.ellipj(u, m)[3]); a1 = _f(elliptic.ellipj(u + 2 * K, m)[3])
            np.testing.assert_allclose(a1 - a0, np.pi, rtol=1e-11)


# --------------------------------------------------------------------------
# elliptic12 / elliptic3
# --------------------------------------------------------------------------

class TestElliptic12:
    PHI = [-7.1, -1.2, 0.0, 0.4, float(PI / 2), 2.9, 6.5, 12.7]

    @pytest.mark.parametrize("m", M_GRID)
    @pytest.mark.parametrize("phi", PHI)
    def test_F_E_Z(self, phi, m):
        F, E, Z = elliptic.elliptic12(phi, m)
        Fr, Er = mp.ellipf(phi, m), mp.ellipe(phi, m)
        Zr = Er - mp.ellipe(m) / mp.ellipk(m) * Fr
        _close(_f(F), Fr, m); _close(_f(E), Er, m); _close(_f(Z), Zr, m, atol=1e-13)


class TestElliptic3:
    @pytest.mark.parametrize("m", [1e-4, 0.3, 0.7, 0.95])
    @pytest.mark.parametrize("n", [-1.5, -0.3, 0.0, 0.4, 0.9])
    @pytest.mark.parametrize("phi", [-4.4, 0.6, float(PI / 2), 2.2, 5.1])
    def test_Pi(self, phi, m, n):
        """4.2.0: phases past [0, pi/2] (and negative) are continued as integrals."""
        got = elliptic.elliptic3(phi, m, n)
        _close(_f(got), mp.ellippi(n, phi, m), m)


# --------------------------------------------------------------------------
# associate integrals
# --------------------------------------------------------------------------

def _B(phi, m): return mp.quad(lambda t: mp.cos(t) ** 2 / mp.sqrt(1 - m * mp.sin(t) ** 2), [0, phi])
def _D(phi, m): return mp.quad(lambda t: mp.sin(t) ** 2 / mp.sqrt(1 - m * mp.sin(t) ** 2), [0, phi])
def _J(phi, m, n): return mp.quad(lambda t: mp.sin(t) ** 2 / ((1 - n * mp.sin(t) ** 2) * mp.sqrt(1 - m * mp.sin(t) ** 2)), [0, phi])


class TestEllipticBD:
    @pytest.mark.parametrize("m", [1e-8, 1e-6, 1e-3, 0.02, 0.3, 0.8, 0.999])
    def test_B_D_S(self, m):
        B, D, S = elliptic.ellipticBD(m)
        with mp.workdps(50):
            Br, Dr = _B(PI / 2, m), _D(PI / 2, m); Sr = (Dr - Br) / m
        _close(_f(B), Br, m); _close(_f(D), Dr, m)
        np.testing.assert_allclose(_f(S), float(Sr), rtol=1e-11)   # series branch below 1e-2


class TestEllipticBDJ:
    @pytest.mark.parametrize("m", [1e-4, 0.4, 0.9])
    @pytest.mark.parametrize("n", [-0.7, 0.3, 0.8])
    @pytest.mark.parametrize("phi", [-4.8, 0.7, float(PI / 2), 2.5, 5.9, 9.1])
    def test_B_D_J(self, phi, m, n):
        """4.2.0: phases are reduced mod pi before the Carlson forms."""
        B, D, J = elliptic.ellipticBDJ(phi, m, n)
        _close(_f(B), _B(phi, m), m); _close(_f(D), _D(phi, m), m); _close(_f(J), _J(phi, m, n), m)


class TestJacobiEDJ:
    @pytest.mark.parametrize("m", [1e-4, 0.4, 0.9])
    @pytest.mark.parametrize("n", [-0.7, 0.3])
    @pytest.mark.parametrize("u", [-8.2, 0.4, 1.7, 5.3, 13.1])
    def test_Eu_Du_Ju(self, u, m, n):
        Eu, Du, Ju = elliptic.jacobiEDJ(u, m, n)
        sn = lambda t: mp.ellipfun("sn", t, m=m); dn = lambda t: mp.ellipfun("dn", t, m=m)
        pts = mp.linspace(0, u, 8)
        _close(_f(Eu), mp.quad(lambda t: dn(t) ** 2, pts), m)
        _close(_f(Du), mp.quad(lambda t: sn(t) ** 2, pts), m)
        _close(_f(Ju), mp.quad(lambda t: sn(t) ** 2 / (1 - n * sn(t) ** 2), pts), m)


# --------------------------------------------------------------------------
# Carlson and Bulirsch
# --------------------------------------------------------------------------

class TestCarlson:
    TRIPLES = [(0, 1, 2), (0.5, 1, 1), (1e-6, 2, 3), (2, 3, 4), (0, 1e-3, 1), (1, 1, 1)]

    @pytest.mark.parametrize("x,y,z", TRIPLES)
    def test_RF_RD(self, x, y, z):
        _close(_f(elliptic.carlsonRF(x, y, z)), mp.elliprf(x, y, z))
        _close(_f(elliptic.carlsonRD(x, y, z)), mp.elliprd(x, y, z))

    @pytest.mark.parametrize("x,y,z,p", [(0, 1, 2, 3), (0.5, 1, 1, 0.3), (2, 3, 4, 5), (1e-6, 2, 3, 0.7)])
    def test_RJ(self, x, y, z, p):
        _close(_f(elliptic.carlsonRJ(x, y, z, p)), mp.elliprj(x, y, z, p))

    @pytest.mark.parametrize("x,y", [(0, 9), (1, 2), (2, 1), (1e-8, 1), (3, 3)])
    def test_RC(self, x, y):
        _close(_f(elliptic.carlsonRC(x, y)), mp.elliprc(x, y))


def _cel_ref(kc, p, a, b):
    return mp.quad(lambda t: (a * mp.cos(t) ** 2 + b * mp.sin(t) ** 2)
                   / ((mp.cos(t) ** 2 + p * mp.sin(t) ** 2) * mp.sqrt(mp.cos(t) ** 2 + kc * kc * mp.sin(t) ** 2)), [0, PI / 2])


class TestBulirsch:
    @pytest.mark.parametrize("kc", [1e-9, 1e-4, 0.1, 0.6, 1.0, 1.7])
    def test_cel1_is_K(self, kc):
        """4.2.0: kc-native Bulirsch; cel1(1e-9) = ln(4/kc) + ..., not 2e6."""
        m = 1 - mp.mpf(kc) ** 2
        np.testing.assert_allclose(_f(elliptic.cel1(kc)), float(mp.ellipk(m)), rtol=1e-11)

    @pytest.mark.parametrize("kc,p,a,b", [(0.6, 0.8, 1.2, 0.7), (0.05, 2.0, 1.0, 1.0), (0.9, 0.3, 0.0, 1.0), (1.3, 1.5, 2.0, -1.0), (1e-3, 0.5, 1.0, 0.5)])
    def test_cel(self, kc, p, a, b):
        with mp.workdps(40):
            ref = _cel_ref(kc, p, a, b)
        np.testing.assert_allclose(_f(elliptic.cel(kc, p, a, b)), float(ref), rtol=1e-11)

    @pytest.mark.parametrize("kc,a,b", [(0.4, 1.0, 0.3), (0.95, 2.0, 1.0)])
    def test_cel2(self, kc, a, b):
        np.testing.assert_allclose(_f(elliptic.cel2(kc, a, b)), float(_cel_ref(kc, 1, a, b)), rtol=1e-11)

    @pytest.mark.parametrize("kc,p", [(0.4, 0.7), (0.9, 1.6), (0.2, 0.4)])
    def test_cel3_is_Pi(self, kc, p):
        n, m = 1 - p, 1 - kc * kc
        np.testing.assert_allclose(_f(elliptic.cel3(kc, p)), float(mp.ellippi(n, m)), rtol=1e-11)


# --------------------------------------------------------------------------
# Weierstrass (theta forms, DLMF 23.6)
# --------------------------------------------------------------------------

LATTICES = [(2.0, 0.5, -2.5), (1.0, 0.0, -1.0), (3.0, 2.9, -5.9), (0.7, -0.2, -0.5)]


def _lattice(e1, e2, e3):
    m = mp.mpf(e2 - e3) / (e1 - e3); w = mp.sqrt(e1 - e3)
    omega1 = mp.ellipk(m) / w; q = mp.qfrom(m=m)
    eta1 = -(PI ** 2 / (12 * omega1)) * mp.jtheta(1, 0, q, 3) / mp.jtheta(1, 0, q, 1)
    return m, w, omega1, q, eta1


def _P_ref(z, e1, e2, e3):
    m, w, *_ = _lattice(e1, e2, e3)
    return e3 + (e1 - e3) / mp.ellipfun("sn", w * z, m=m) ** 2


def _zeta_ref(z, e1, e2, e3):
    m, w, omega1, q, eta1 = _lattice(e1, e2, e3); v = PI * z / (2 * omega1)
    return eta1 * z / omega1 + PI / (2 * omega1) * mp.jtheta(1, v, q, 1) / mp.jtheta(1, v, q)


def _sigma_ref(z, e1, e2, e3):
    m, w, omega1, q, eta1 = _lattice(e1, e2, e3); v = PI * z / (2 * omega1)
    return 2 * omega1 / PI * mp.exp(eta1 * z * z / (2 * omega1)) * mp.jtheta(1, v, q) / mp.jtheta(1, 0, q, 1)


class TestWeierstrass:
    @pytest.mark.parametrize("e1,e2,e3", LATTICES)
    @pytest.mark.parametrize("zf", [0.13, 0.37, 0.8, 1.6, 2.7])
    def test_P_and_Pprime(self, zf, e1, e2, e3):
        z = zf * float(_lattice(e1, e2, e3)[2])       # fractions of omega1, off the lattice
        np.testing.assert_allclose(_f(elliptic.weierstrassP(z, e1, e2, e3)), float(_P_ref(z, e1, e2, e3)), rtol=1e-10)
        dP = mp.diff(lambda t: _P_ref(t, e1, e2, e3), z)
        np.testing.assert_allclose(_f(elliptic.weierstrassPPrime(z, e1, e2, e3)), float(dP), rtol=1e-9)

    @pytest.mark.parametrize("e1,e2,e3", LATTICES)
    @pytest.mark.parametrize("zf", [0.13, 0.37, 0.8, 1.6, 2.7, 4.3])
    def test_zeta_sigma_theta_forms(self, zf, e1, e2, e3):
        """4.2.0: sigma by the closed theta form is right past 2*omega1 (zf > 2)."""
        z = zf * float(_lattice(e1, e2, e3)[2])
        np.testing.assert_allclose(_f(elliptic.weierstrassZeta(z, e1, e2, e3)), float(_zeta_ref(z, e1, e2, e3)), rtol=1e-10)
        np.testing.assert_allclose(_f(elliptic.weierstrassSigma(z, e1, e2, e3)), float(_sigma_ref(z, e1, e2, e3)), rtol=1e-10)

    @pytest.mark.parametrize("e1,e2,e3", LATTICES)
    def test_invariants(self, e1, e2, e3):
        g2, g3, disc = elliptic.weierstrassInvariants(e1, e2, e3)
        np.testing.assert_allclose(_f(g2), -4 * (e1 * e2 + e2 * e3 + e3 * e1), rtol=1e-13)
        np.testing.assert_allclose(_f(g3), 4 * e1 * e2 * e3, rtol=1e-13)
        np.testing.assert_allclose(_f(disc), 16 * ((e1 - e2) * (e2 - e3) * (e1 - e3)) ** 2, rtol=1e-12)


# --------------------------------------------------------------------------
# theta functions
# --------------------------------------------------------------------------

class TestTheta:
    V = [-7.3, 0.2, 1.1, 2.9, 15.7]

    @pytest.mark.parametrize("m", [1e-4, 0.3, 0.9, 0.999])
    @pytest.mark.parametrize("v", V)
    @pytest.mark.parametrize("j", [1, 2, 3, 4])
    def test_theta_and_prime(self, j, v, m):
        """4.2.0: theta() evaluates directly on v; the angle-addition recurrence holds at large v."""
        q = mp.qfrom(m=m)
        _close(_f(elliptic.theta(j, v, m)), mp.jtheta(j, v, q), m, atol=1e-13)
        th, thp = elliptic.theta_prime(j, v, m)
        _close(_f(th), mp.jtheta(j, v, q), m, atol=1e-13)
        _close(_f(thp), mp.jtheta(j, v, q, 1), m, atol=1e-13)

    @pytest.mark.parametrize("m", [1e-4, 0.3, 0.9, 0.999])
    @pytest.mark.parametrize("u", [-3.1, 0.9, 2.2, 7.5])
    def test_jacobi_Theta_H(self, u, m):
        Th, H = elliptic.jacobiThetaEta(u, m)
        q = mp.qfrom(m=m); v = PI * u / (2 * mp.ellipk(m))
        _close(_f(Th), mp.jtheta(4, v, q), m, atol=1e-13); _close(_f(H), mp.jtheta(1, v, q), m, atol=1e-13)


# --------------------------------------------------------------------------
# complex phase / argument
# --------------------------------------------------------------------------

class TestComplex:
    Z = [0.8 + 0.5j, float(PI / 2) + 0.5j, float(PI / 2) - 1.3j, 1.2 + 0.05j, 2.0 + 0.7j, -0.6 + 1.1j, 0.3j]

    @pytest.mark.parametrize("m", [1e-4, 0.3, 0.6, 0.95])
    @pytest.mark.parametrize("u", Z)
    def test_elliptic12i(self, u, m):
        """4.2.0 (#35): the imaginary part at Re(u) = pi/2 and the period term below pi/2.

        On the cut Re(u) = pi/2, |Im u| > arccosh(1/sqrt(m)) the library returns the
        boundary value from the right (documented); mpmath returns the left one, so the
        reference is taken a hair to the right of the cut there.
        """
        Fi, Ei, Zi = elliptic.elliptic12i(u, m)
        on_cut = abs(u.real - float(PI / 2)) < 1e-15 and abs(u.imag) > float(mp.acosh(1 / mp.sqrt(m)))
        z = mp.mpc(u.real, u.imag) + (mp.mpf("1e-14") if on_cut else 0)   # mpmath reduces offsets below ~1e-16 onto the cut
        Fr, Er = mp.ellipf(z, m), mp.ellipe(z, m)
        Zr = Er - mp.ellipe(m) / mp.ellipk(m) * Fr
        for got, ref in ((Fi, Fr), (Ei, Er), (Zi, Zr)):
            np.testing.assert_allclose(_c(got), complex(ref), rtol=_tol(m), atol=1e-13)

    @pytest.mark.parametrize("m,y", [(0.3, -1.3), (0.6, -1.3), (0.95, 0.5)])
    def test_elliptic12i_cut_convention(self, m, y):
        """Beyond the branch point on Re(u) = pi/2 the value is the right boundary value:
        continuous from Re(u) > pi/2, equal to 2K - conj(F_left) (F(pi - w) = 2K - F(w)),
        and both sides agree with mpmath off the cut."""
        assert abs(y) > float(mp.acosh(1 / mp.sqrt(m)))
        K, E = float(mp.ellipk(m)), float(mp.ellipe(m))
        F0, E0, _ = (_c(v) for v in elliptic.elliptic12i(float(PI / 2) + 1j * y, m))
        for d in (1e-9, -1e-9):
            z = float(PI / 2) + d + 1j * y
            Fd, Ed, _ = (_c(v) for v in elliptic.elliptic12i(z, m))
            np.testing.assert_allclose(Fd, complex(mp.ellipf(mp.mpc(z.real, z.imag), m)), rtol=1e-8)
            np.testing.assert_allclose(Ed, complex(mp.ellipe(mp.mpc(z.real, z.imag), m)), rtol=1e-8)
            if d > 0:
                np.testing.assert_allclose(F0, Fd, rtol=1e-8); np.testing.assert_allclose(E0, Ed, rtol=1e-8)
            else:
                np.testing.assert_allclose(F0, 2 * K - np.conj(Fd), rtol=1e-8)
                np.testing.assert_allclose(E0, 2 * E - np.conj(Ed), rtol=1e-8)

    @pytest.mark.parametrize("m", [1e-4, 0.3, 0.6, 0.95])
    @pytest.mark.parametrize("u", [0.8 + 0.5j, 2.4 - 0.9j, -1.7 + 1.4j, 0.2j])
    def test_ellipji(self, u, m):
        sn, cn, dn = elliptic.ellipji(u, m); z = mp.mpc(u.real, u.imag)
        for got, name in ((sn, "sn"), (cn, "cn"), (dn, "dn")):
            np.testing.assert_allclose(_c(got), complex(mp.ellipfun(name, z, m=m)), rtol=_tol(m), atol=1e-13)


# --------------------------------------------------------------------------
# nome, inverses, agm, arc length
# --------------------------------------------------------------------------

class TestNome:
    @pytest.mark.parametrize("m", M_GRID)
    def test_nomeq(self, m):
        _close(_f(elliptic.nomeq(m)), mp.qfrom(m=m), m, atol=1e-15)

    @pytest.mark.parametrize("q", [1e-6, 1e-3, 0.01, 0.3, 0.6, 0.75, 0.778])
    def test_inversenomeq(self, q):
        """4.2.0: closed theta form, exact up to q_max ~ 0.7789."""
        np.testing.assert_allclose(_f(elliptic.inversenomeq(q)), float(mp.mfrom(q=q)), rtol=1e-13, atol=1e-16)

    @pytest.mark.parametrize("m", [1e-6, 0.2, 0.7, 0.99])
    def test_round_trip(self, m):
        np.testing.assert_allclose(_f(elliptic.inversenomeq(_f(elliptic.nomeq(m)))), m, rtol=1e-11)


class TestInverseE:
    @pytest.mark.parametrize("m", [1e-4, 0.4, 0.9, 0.999])
    @pytest.mark.parametrize("phi", [-2.2, 0.3, 1.4, 2.9, 7.7])
    def test_inverselliptic2(self, phi, m):
        E = float(mp.ellipe(phi, m))
        np.testing.assert_allclose(_f(elliptic.inverselliptic2(E, m)), phi, rtol=1e-10, atol=1e-11)


class TestAgmArc:
    @pytest.mark.parametrize("a,b", [(1.0, 0.3), (1.0, 1e-8), (2.5, 2.5), (0.7, 3.1)])
    def test_agm(self, a, b):
        np.testing.assert_allclose(_f(elliptic.agm(a, b)), float(mp.agm(a, b)), rtol=1e-13)

    @pytest.mark.parametrize("a,b,t0,t1", [(2, 1, 0, 1.1), (1, 2, 0.3, 2.9), (3, 0.5, -1.0, 4.0), (2, 1, 0, None)])
    def test_arclength(self, a, b, t0, t1):
        got = elliptic.arclength_ellipse(a, b, t0, t1) if t1 is not None else elliptic.arclength_ellipse(a, b)
        T1 = t1 if t1 is not None else 2 * PI
        ref = mp.quad(lambda t: mp.sqrt(a * a * mp.sin(t) ** 2 + b * b * mp.cos(t) ** 2), mp.linspace(t0, T1, 6))
        np.testing.assert_allclose(_f(got), float(ref), rtol=1e-12)


# --------------------------------------------------------------------------
# edge inputs
# --------------------------------------------------------------------------

class TestEdgeInputs:
    def test_nan_propagates(self):
        for got in elliptic.ellipj(np.nan, 0.5): assert np.isnan(_f(got))
        for got in elliptic.elliptic12(np.nan, 0.5): assert np.isnan(_f(got))
        assert np.isnan(_f(elliptic.elliptic3(0.5, np.nan, 0.2)))
        assert np.isnan(_f(elliptic.nomeq(np.nan)))

    def test_empty_arrays(self):
        e = np.array([])
        for got in elliptic.ellipj(e, e): assert got.shape == (0,)
        for got in elliptic.elliptic12(e, e): assert got.shape == (0,)
        B, D, S = elliptic.ellipticBD(e); assert B.shape == D.shape == S.shape == (0,)
