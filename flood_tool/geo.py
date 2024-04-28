from numpy import array, asarray, mod, sin, cos, tan, sqrt, arctan2, floor, rad2deg, deg2rad, stack
from scipy.linalg import inv
import numpy as np

__all__ = ['get_easting_northing_from_gps_lat_long',
           'get_gps_lat_long_from_easting_northing']

class Ellipsoid(object):
    """ Data structure for a global ellipsoid. """

    def __init__(self, a, b, F_0):
        self.a = a
        self.b = b
        self.n = (a - b) / (a + b)
        self.e2 = (a ** 2 - b ** 2) / a ** 2
        self.F_0 = F_0
        self.H = 0


class Datum(Ellipsoid):
    """ Data structure for a global datum. """

    def __init__(self, a, b, F_0, phi_0, lam_0, E_0, N_0, H):
        super().__init__(a, b, F_0)
        self.phi_0 = phi_0
        self.lam_0 = lam_0
        self.E_0 = E_0
        self.N_0 = N_0
        self.H = H


def dms2rad(deg, min=0, sec=0):
    """Convert degrees, minutes, seconds to radians.

    Parameters
    ----------
    deg: array_like
        Angle in degrees.
    min: array_like
        (optional) Angle component in minutes.
    sec: array_like
        (optional) Angle component in seconds.
    Returns
    -------
    numpy.ndarray
        Angle in radians.
    """
    deg = asarray(deg)
    # In a full circle there are 360 degrees(°).
    # Each degree is split up into 60 parts, each part being 1/60 of a degree. These parts are called minutes(').
    # Each minute is split up into 60 parts, each part being 1/60 of a minute. These parts are called seconds('').
    # 1°=60′，1′=60″ ，1°=3600″
    return deg2rad(deg + min / 60. + sec / 3600.)


def rad2dms(rad, dms=False):
    """Convert radians to degrees, minutes, seconds.
    Parameters
    ----------
    rad: array_like
        Angle in radians.
    dms: bool
        Use degrees, minutes, seconds format. If False, use decimal degrees.
    Returns
    -------
    numpy.ndarray
        Angle in degrees, minutes, seconds or decimal degrees.
    """

    rad = asarray(rad)
    deg = rad2deg(rad)
    if dms:
        min = 60.0 * mod(deg, 1.0)
        sec = 60.0 * mod(min, 1.0)
        return stack((floor(deg), floor(min), sec.round(4)))
    else:
        return deg


osgb36 = Datum(a=6377563.396,
               b=6356256.910,
               F_0=0.9996012717,
               phi_0=deg2rad(49.0),
               lam_0=deg2rad(-2.),
               E_0=400000,
               N_0=-100000,
               H=24.7)

wgs84 = Ellipsoid(a=6378137,
                  b=6356752.3142,
                  F_0=0.9996)


def lat_long_to_xyz(phi, lam, rads=False, datum=osgb36):
    """Convert input latitude/longitude in a given datum into
    Cartesian (x, y, z) coordinates.
    Parameters
    ----------
    phi: array_like
        Latitude in degrees (if radians=False) or radians (if radians=True).
    lam: array_like
        Longitude in degrees (if radians=False) or radians (if radians=True).
    rads: bool (optional)
        If True, input latitudes and longitudes are in radians.
    datum: Datum (optional)
        Datum to use for conversion.
    """
    if not rads:
        phi = deg2rad(phi)
        lam = deg2rad(lam)

    nu = datum.a * datum.F_0 / sqrt(1 - datum.e2 * sin(phi) ** 2)

    return array(((nu + datum.H) * cos(phi) * cos(lam),
                  (nu + datum.H) * cos(phi) * sin(lam),
                  ((1 - datum.e2) * nu + datum.H) * sin(phi)))

def xyz_to_lat_long(x, y, z, rads=False, datum=osgb36):
    p = sqrt(x ** 2 + y ** 2)

    lam = arctan2(y, x)
    phi = arctan2(z, p * (1 - datum.e2))

    for _ in range(10):
        nu = datum.a * datum.F_0 / sqrt(1 - datum.e2 * sin(phi) ** 2)
        dnu = -datum.a * datum.F_0 * cos(phi) * sin(phi) / (1 - datum.e2 * sin(phi) ** 2) ** 1.5

        f0 = (z + datum.e2 * nu * sin(phi)) / p - tan(phi)
        f1 = datum.e2 * (nu ** cos(phi) + dnu * sin(phi)) / p - 1.0 / cos(phi) ** 2
        phi -= f0 / f1

    if not rads:
        phi = rad2dms(phi)
        lam = rad2dms(lam)

    return phi, lam


def get_easting_northing_from_gps_lat_long(phi, lam, rads=False):
    """ Get OSGB36 easting/northing from GPS latitude and longitude pairs.

    Parameters
    ----------
    phi: float/arraylike
        GPS (i.e. WGS84 datum) latitude value(s)
    lam: float/arrayling
        GPS (i.e. WGS84 datum) longitude value(s).
    rads: bool (optional)
        If true, specifies input is is radians.
    Returns
    -------
    numpy.ndarray
        Easting values (in m)
    numpy.ndarray
        Northing values (in m)

    Examples
    --------
    >>> get_easting_northing_from_gps_lat_long([55.5], [-1.54])
    (array([429157.5449526]), array([623009.09706685]))

    References
    ----------
    Based on the formulas in "A guide to coordinate systems in Great Britain".
    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
    """
    # from WGS84 to OSGB36

    phi, lam = WGS84toOSGB36(phi, lam, rads=False)
    phi = dms2rad(phi)
    lam = dms2rad(lam)

    a = osgb36.a
    b = osgb36.b
    n = osgb36.n
    e2 = osgb36.e2
    F_0 = osgb36.F_0
    N_0 = osgb36.N_0
    E_0 = osgb36.E_0
    phi_0 = osgb36.phi_0
    lam_0 = osgb36.lam_0

    nu = a * F_0 * (1 - e2 * sin(phi) ** 2) ** -0.5
    rho = a * F_0 * (1 - e2) * (1 - e2 * sin(phi) ** 2) ** -1.5
    eta2 = nu / rho - 1

    M = b * F_0 * ((1 + n + (5 / 4) * (n ** 2 + n ** 3)) * (phi - phi_0)
                   - (3 * (n + n ** 2) + (21 / 8) * n ** 3) * sin(phi - phi_0) * cos(phi + phi_0)
                   + ((15 / 8) * (n ** 2 + n ** 3) * sin(2 * (phi - phi_0)) * cos(2 * (phi + phi_0)))
                   - (35 / 24) * n ** 3 * sin(3 * (phi - phi_0)) * cos(3 * (phi + phi_0)))

    I = M + N_0
    II = nu / 2 * sin(phi) * cos(phi)
    III = nu / 24 * sin(phi) * cos(phi) ** 3 * (5 - tan(phi) ** 2 + 9 * eta2)
    IIIA = nu / 720 * sin(phi) * cos(phi) ** 5 * (61 - 58 * tan(phi) ** 2 + tan(phi) ** 4)
    IV = nu * cos(phi)
    V = nu / 6 * cos(phi) ** 3 * (nu / rho - tan(phi) ** 2)
    VI = nu / 120 * cos(phi) ** 5 * (5 - 18 * tan(phi) ** 2 + tan(phi) ** 4 + 14 * eta2 - 58 * tan(phi) ** 2 * eta2)

    N = I + II * (lam - lam_0) ** 2 + III * (lam - lam_0) ** 4 + IIIA * (lam - lam_0) ** 6

    E = E_0 + IV * (lam - lam_0) + V * (lam - lam_0) ** 3 + VI * (lam - lam_0) ** 5

    return E,N

def get_gps_lat_long_from_easting_northing(east, north, rads=False, dms=False):
    """ Get OSGB36 easting/northing from GPS latitude and
    longitude pairs.

    Parameters
    ----------
    east: float/arraylike
        OSGB36 easting value(s) (in m).
    north: float/arrayling
        OSGB36 easting value(s) (in m).
    rads: bool (optional)
        If true, specifies ouput is is radians.
    dms: bool (optional)
        If true, output is in degrees/minutes/seconds. Incompatible
        with rads option.

    Returns
    -------
    numpy.ndarray
        GPS (i.e. WGS84 datum) latitude value(s).
    numpy.ndarray
        GPS (i.e. WGS84 datum) longitude value(s).

    Examples
    --------
    >>> get_gps_lat_long_from_easting_northing([429157], [623009])
    (array([55.49999916]), array([-1.54000861]))

    References
    ----------
    Based on the formulas in "A guide to coordinate systems in Great Britain".
    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
    """

    a = osgb36.a
    b = osgb36.b
    n = osgb36.n
    e2 = osgb36.e2
    F_0 = osgb36.F_0
    N_0 = osgb36.N_0
    E_0 = osgb36.E_0
    phi_0 = osgb36.phi_0
    lam_0 = osgb36.lam_0

    my_phi = []
    my_lam = []

    for i in range(len(east)):
        E = east[i]
        N = north[i]

        phi = (N - N_0) / (a * F_0) + phi_0

        M = b * F_0 * ((1 + n + (5 / 4) * (n ** 2 + n ** 3)) * (phi - phi_0)
                       - (3 * (n + n ** 2) + (21 / 8) * n ** 3) * sin(phi - phi_0) * cos(phi + phi_0)
                       + ((15 / 8) * (n ** 2 + n ** 3) * sin(2 * (phi - phi_0)) * cos(2 * (phi + phi_0)))
                       - (35 / 24) * n ** 3 * sin(3 * (phi - phi_0)) * cos(3 * (phi + phi_0)))

        while abs(N - N_0 - M) >= 0.00001:
            phi = (N - N_0 - M) / (a * F_0) + phi

            M = b * F_0 * ((1 + n + (5 / 4) * (n ** 2 + n ** 3)) * (phi - phi_0)
                           - (3 * (n + n ** 2) + (21 / 8) * n ** 3) * sin(phi - phi_0) * cos(phi + phi_0)
                           + ((15 / 8) * (n ** 2 + n ** 3) * sin(2 * (phi - phi_0)) * cos(2 * (phi + phi_0)))
                           - (35 / 24) * n ** 3 * sin(3 * (phi - phi_0)) * cos(3 * (phi + phi_0)))

        nu = a * F_0 * (1 - e2 * sin(phi) ** 2) ** -0.5
        rho = a * F_0 * (1 - e2) * (1 - e2 * sin(phi) ** 2) ** -1.5
        eta2 = nu / rho - 1

        VII = tan(phi) / (2 * rho * nu)
        VIII = tan(phi) / (24 * rho * nu ** 3) * (5 + 3 * tan(phi) ** 2 + eta2 - 9 * tan(phi) ** 2 * eta2)
        IX = tan(phi) / (720 * rho * nu ** 5) * (61 + 90 * tan(phi) ** 2 + 45 * tan(phi) ** 4)
        X = 1 / (nu * cos(phi))
        XI = (nu / rho + 2 * tan(phi) ** 2) / (cos(phi) * 6 * nu ** 3)
        XII = (5 + 28 * tan(phi) ** 2 + 24 * tan(phi) ** 4) / (cos(phi) * 120 * nu ** 5)
        XIIA = (61 + 662 * tan(phi) ** 2 + 1320 * tan(phi) ** 4 + 720 * tan(phi) ** 6) / (cos(phi) * 5040 * nu ** 7)

        phi = phi - VII * (E - E_0) ** 2 + VIII * (E - E_0) ** 4 - IX * (E - E_0) ** 6
        lam = lam_0 + X * (E - E_0) - XI * (E - E_0) ** 3 + XII * (E - E_0) ** 5 - XIIA * (E - E_0) ** 7
        phi, lam = OSGB36toWGS84(phi, lam, rads=True)
        my_phi.append(rad2dms(phi)[0])
        my_lam.append(rad2dms(lam)[0])

    return np.array(my_phi), np.array(my_lam)


class HelmertTransform(object):
    """Callable class to perform a Helmert transform."""

    def __init__(self, s, rx, ry, rz, T):
        self.T = T.reshape((3, 1))

        self.M = array([[1 + s, -rz, ry],
                        [rz, 1 + s, -rx],
                        [-ry, rx, 1 + s]])

    def __call__(self, X):
        X = X.reshape((3, -1))
        return self.T + self.M @ X


class HelmertInverseTransform(object):
    """Callable class to perform the inverse of a Helmert transform."""

    def __init__(self, s, rx, ry, rz, T):
        self.T = T.reshape((3, 1))

        self.M = inv(array([[1 + s, -rz, ry],
                            [rz, 1 + s, -rx],
                            [-ry, rx, 1 + s]]))

    def __call__(self, X):
        X = X.reshape((3, -1))
        return self.M @ (X - self.T)


OSGB36transform = HelmertTransform(20.4894e-6,
                                   -dms2rad(0, 0, 0.1502),
                                   -dms2rad(0, 0, 0.2470),
                                   -dms2rad(0, 0, 0.8421),
                                   array([-446.448, 125.157, -542.060]))

WGS84transform = HelmertInverseTransform(20.4894e-6,
                                         -dms2rad(0, 0, 0.1502),
                                         -dms2rad(0, 0, 0.2470),
                                         -dms2rad(0, 0, 0.8421),
                                         array([-446.448, 125.157, -542.060]))


def WGS84toOSGB36(phi, lam, rads=False):
    """Convert WGS84 latitude/longitude to OSGB36 latitude/longitude.

    Parameters
    ----------
    phi : array_like or float
        Latitude in degrees or radians on WGS84 datum.
    lam : array_like or float
        Longitude in degrees or radians on WGS84 datum.
    rads : bool, optional
        If True, phi and lam are in radians. If False, phi and lam are in degrees.
    Returns
    -------
    tuple of numpy.ndarrays
        Latitude and longitude on OSGB36 datum in degrees or radians.
    """
    xyz = OSGB36transform(lat_long_to_xyz(asarray(phi), asarray(lam),
                                          rads=rads, datum=wgs84))
    return xyz_to_lat_long(*xyz, rads=rads, datum=osgb36)


def OSGB36toWGS84(phi, lam, rads=False):
    """Convert OSGB36 latitude/longitude to WGS84 latitude/longitude.

    Parameters
    ----------
    phi : array_like or float
        Latitude in degrees or radians on OSGB36 datum.
    lam : array_like or float
        Longitude in degrees or radians on OSGB36 datum.
    rads : bool, optional
        If True, phi and lam are in radians. If False, phi and lam are in degrees.
    Returns
    -------
    tuple of numpy.ndarrays
        Latitude and longitude on WGS84 datum in degrees or radians.
    """
    xyz = WGS84transform(lat_long_to_xyz(asarray(phi), asarray(lam),
                                         rads=rads, datum=osgb36))
    return xyz_to_lat_long(*xyz, rads=rads, datum=wgs84)