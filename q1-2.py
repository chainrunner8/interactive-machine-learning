import numpy as np

K = 10
T = 10_000
L_ODD = np.array([1,-1,*[0.1 for _ in range(K-2)]])
L_EVEN = np.array([-1,1,*[0.1 for _ in range(K-2)]])

def ftl():
    cumul_losses = np.zeros(K)
    ftl_loss = 0

    for t in range(1, T):
        I_t = np.argmin(cumul_losses)
        if t%2==0:
            ftl_loss += L_EVEN[I_t]
            cumul_losses += L_EVEN
        else:
            ftl_loss += L_ODD[I_t]
            cumul_losses += L_ODD
    return ftl_loss

def exp_weights():
    cumul_losses = np.zeros(K)
    eta = np.sqrt(2*np.log(K)/T)
    experts = np.arange(K)
    p_arr = np.array([1/K]*K)
    rng = np.random.default_rng()
    exp_weight_loss = 0

    for t in range(1, T):
        I_t = rng.choice(experts, size=1, p=p_arr)
        if t%2==0:
            exp_weight_loss += L_EVEN[I_t]
            cumul_losses += L_EVEN
        else:
            exp_weight_loss += L_ODD[I_t]
            cumul_losses += L_ODD
        
    return 