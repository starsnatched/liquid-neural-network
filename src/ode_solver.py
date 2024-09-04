def fused_ode_solver(x, f, dt, params):
    """
    Fused ODE solver for LTC networks.
    
    Args:
    x (torch.Tensor): Current state
    f (callable): Function representing the ODE
    dt (float): Time step
    params (dict): Additional parameters for f
    
    Returns:
    torch.Tensor: Next state
    """
    x_next = (x + dt * f(x, params['I'], params['t'], params['theta']) * params['A']) / (1 + dt * (1/params['tau'] + f(x, params['I'], params['t'], params['theta'])))
    return x_next