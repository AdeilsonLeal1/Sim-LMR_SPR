from numpy import *

def Reflectance(nLayers, d,  index, theta_i, wavelenght, pol):
    """ The numerical model is based on the attenuated total reflection method combined with the transfer matrix
        method for a multilayer system according to:
        * PALIWAL, N.; JOHN, J. Lossy mode resonance based fiber optic sensors. In: Fiber Optic Sensors.
        [S_TM.l.]: Springer, 2017. p. 31-50. DOI : 10.1007/978-3-319-42625-9_2."""

    j = complex(0, 1)  # Simplification for the complex number "j"
    k0 = (2 * pi) / wavelenght  # Wave number

    b = []  # b_j -> Phase shift in each layer
    q = []  # q_j -> Optical Admittance 
   

    m_ = []  # M_j -> Transfer matrix between each layer - TM polarization
   
    for layer in range(nLayers):
        y = sqrt((index[layer] ** 2) - ((index[0] * sin(theta_i)) ** 2))

        b.append(k0 * d[layer] * y)
        if pol == 'TM':
            q.append(y / index[layer] ** 2)
        else:
            q.append(y)

        # Total Transfer Matrix
        if layer < (nLayers - 1):
            m_.append(array([[cos(b[layer]), (-j / q[layer]) * sin(b[layer])],
                                [-j * q[layer] * sin(b[layer]), cos(b[layer])]]))

    Mt_ = m_[0]  # Mt_ -> Total Transfer Matrix - TM polarization
    for k in range(nLayers - 2):
        Mt_ = Mt_ @ m_[k + 1]


    num = (Mt_[0][0] + Mt_[0][1] * q[ nLayers - 1]) * q[0] - (
            Mt_[1][0] + Mt_[1][1] * q[ nLayers - 1])
    den = (Mt_[0][0] + Mt_[0][1] * q[ nLayers - 1]) * q[0] + (
            Mt_[1][0] + Mt_[1][1] * q[ nLayers - 1])


    r = num / den # 'r_'-> Fresnel reflection coefficient


    return abs(r) ** 2 # Reflectance - TE or TM polarization