#!/usr/bin/python3
#-*- coding: utf-8 -*-

from numpy.random import random
from numpy import mean, shape, linspace
from numpy import sqrt as nsqrt
from numpy import append as nappend
import resiland as rl



def two_FHN_NoN_f():
    """ ODE for FitzHugh-Nagumo oscillators in a network of networks.

    Parameters
    ----------
    
    Returns
    -------
    dXdt, dYdt :                            ODE for x- and y-component in N dimensions of the FitzHugh-Nagumo oscillators
    dZdt :                                  ODE for the coupling of the mean field to the potential landscape

    References
    ----------

    """
    for q in [0,1]:
        for i in range(N):
            yield rl.x_component(X, Y, i, q, a, N, k_int1, k_int2, k_b, Sym, S, B, noise = False) ### dXdt

        for i in range(N):
            yield rl.y_component(X, Y, i, q, b, c, noise = False) ### dYdt

    yield rl.coupling_to_potential(now, amp, mu, var, Z ,fb, gamma, X, N) ### dZdz



def generate_time_series(amp, mu, var, initial_z, times, k_int2_list, k_b_list, fragment_barrier_position = [[0,0]]):
    """ Integration of the ODEs during possible changes of the potential landscape

    Parameters
    ----------
    amp, mu, var :                          Control parameters of each Gaussian functions for the amplitude, mu and variance (list)
    initial_z :                             initial position z in the potential landscape  (float)
    times :                                 intergation time (array)
    fragment_barrier_position :             2d array of the positions of the start- and endpoint of the Cantor sets

    Returns
    -------
    x_1, y_1, x_2, y_2 :                    x_1, y_1, x_2, y_2 components of the NoNs of FHN oscillators
    z :                                     Coupling function to the potential landscape 

    References
    ----------

    """

    fb, gamma = 0, 1
    k_int2, k_b = 0.116, 0.00001
       
    initial_state = random(4*N)
    new_initial = nappend(initial_state, initial_z)
    I.set_initial_value(new_initial, 0.0)
    x_1, y_1, x_2, y_2, z, events = [], [], [], [], [], []

    number_cantor,_ = shape(fragment_barrier_position)
    fb_pos = [rl.create_cantor(4, fragment_barrier_position[i][0], fragment_barrier_position[i][1]) for i in range(number_cantor)]

    for time in times:
        get_control, amp, mu, var = rl.return_set_pa(k_int2, k_b, gamma, fb, amp, mu, var)
        sigma = nsqrt(var)
        I.set_parameters(get_control)
        state = I.integrate(time)
        x_1.append(mean(state[0*N:1*N]))
        y_1.append(mean(state[1*N:2*N]))
        x_2.append(mean(state[2*N:3*N]))
        y_2.append(mean(state[3*N:4*N]))
        z.extend(state[4*N:4*N+1])

    
        diff_pos = rl.calculate_diff(mu)
        if z[-1] <= diff_pos[0]:
            k_int2 = k_int2_list[0]
            k_b =  k_b_list[0]
        if z[-1]>= diff_pos[-1]:
            k_int2 = k_int2_list[-1]
            k_b =  k_b_list[-1]
        for i in range(len(diff_pos)-1):
            if diff_pos[i] < z[-1] <= diff_pos[i+1]:
                k_int2 = k_int2_list[i+1]
                k_b =  k_b_list[i+1]

        fb = 0
        rl.test_potential(z[-1], amp,mu, sigma)

        for i in range(number_cantor):
            if fb_pos[i][0] < z[-1] < fb_pos[i][-1]:
                fb =  rl.cantor_slope(z[-1], fb_pos[i], steepness, amp, mu, sigma)

        if(len(x_1)>5):
            peak_pos = rl.find_max(x_1[-3:] + x_2[-3:], 0.863, time)
            if (peak_pos !=None):
                events.append(peak_pos)
        if(len(events)!=0 and time == events[-1][0]):
            gamma = rl.sign_change(gamma, z, 0 , 0, random = False)

    return  x_1, y_1, x_2, y_2, z 



if __name__== '__main__':
    N, p = 25, 0.0 ### number of oscillators in network

    S = rl.create_network(N, N-1, p, "ws")
    B = [
            [ 0, 1 ],
            [ 1, 0 ],
        ] ### bidirectional coupling for network of networks (NoNs)

    now = 2 ### number of wells in the potential landscape

    a, b, c = -0.0276, linspace(0.006,0.014, num = N), 0.02 ### control parameters of FitzHugh-Nagumo oscillators
    k_int1      = 0.115                                     ### global coupling strength for network 1 (all equal for each well)
    k_int2_list = [0.116, 0.125]                            ### global coupling strength for network 2 (different for each well)
    k_b_list    = [1.045e-5, 1.5e-5]                        ### global coupling strength between network 1 and 2 (different for each well)


    X, Y, Z, Sym, helpers = rl.create_dynamic_variables(N)  
    
    k_int2, k_b, gamma, fb = rl.symbol_dynamics()
    amp, mu, var = rl.symbol_potential(now)

    
    steepness  = 0.3 ### steepness parameter of the potential landscape inside the fragmented barrier

    control_para = rl.make_control_pars(k_int2, k_b, gamma, fb, now)

    I = rl.prepare_jitcode(two_FHN_NoN_f, helpers, N, control_para)

    amp = [5.5, 5.5]                            ### control parameters of the amplitude of the potential landscape
    mu  = [-2, 2]                               ### cantrol parameters of the mean of the potential landscape
    var = [1.8, 1.8]                            ### control parameters of the variance of the potential landscape
   
    x_1, y_1, x_2, y_2, z = generate_time_series(amp, mu, var, -1, range(1000, 101000), k_int2_list, k_b_list)

