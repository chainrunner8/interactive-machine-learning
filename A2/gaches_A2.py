import numpy as np
import matplotlib.pyplot as plt

K = 2
T = 10_000
M = 20

def sample_loss(rng, q):
    success = rng.binomial(n=1, p=q, size=1).item()
    return 1 if success else -1

# cant use FTL, cant use EW

# reasoning: Adahedge does work with random losses
# however we still need to be able to observe all losses every round
# what if we can't? then take a non-informative prior: q_hat=0.5
# perform bayesian updating of q_hat.
# when the loss is not observe: take 2*q_hat - 1

def b_ada_hedge(rng, q1, q2, theta, phi):
    '''returns the loss of the algo'''
    smallest = np.argmin([q1,q2])

    cumul_losses = np.zeros(K)
    experts = np.arange(K)

    alphas = [1,1]
    betas = [1,1]
    q_hats = [.5,.5]

    p_exp = np.array([1/K]*K)
    b_ada_regret = 0

    #for visualising
    eta_history = np.zeros(T)
    var_history = np.zeros((T,K))

    # here A_t is Beta_t from the assignment
    A_t = 2*np.log(K)
    for t in range(T):
        eta = np.log(K)/(A_t + np.exp(-t/theta + phi))
        eta_history[t] = eta

        I_t = rng.choice(experts, size=1, p=p_exp).item()

        sampled_losses = [sample_loss(rng, q1), sample_loss(rng, q2)]

        # we suffer the sampled loss for the chosen arm
        # and impute for the other arm
        if I_t == 0:
            losses = [sampled_losses[0], 2*q_hats[1]-1]
        else:
            losses = [2*q_hats[0]-1, sampled_losses[1]]

        # next we update the estimate of q for the I_t arm
        if sampled_losses[I_t] == 1: alphas[I_t]+=1
        else: betas[I_t]+=1
        q_hats[I_t] = alphas[I_t]/(alphas[I_t]+betas[I_t]) #bayesian updating of q hat

        #the below chunk just collects data for viz
        for j in range(K):
            alpha_beta = alphas[j]+betas[j]
            # this is the variance of the beta distro:
            var_history[t,j] = (alphas[j]*betas[j]/(alpha_beta**2*(alpha_beta+1)))

        b_ada_regret += sampled_losses[I_t] - sampled_losses[smallest]
        cumul_losses += losses

        # below i changed the way i compute p_t but the result is the same
        # i still ran into overflow errors with the log sum exp trick we used in A1
        # the probas are the same, i just rescale them with -max(log_weights)
        # that way the exponent are strongly shrunk
        log_weights = -eta*cumul_losses
        log_weights -= np.max(log_weights)
        p_exp = np.exp(log_weights)
        p_exp /= np.sum(p_exp)

        A_t += np.dot(p_exp, losses) + np.log(np.dot(p_exp, np.exp(-eta*np.array(losses))))/eta

    return b_ada_regret, q_hats, eta_history, var_history

def plot_lr_var(pairs, all_var_histories, all_eta_histories):
    '''
    outputs two figures each with 6 subplots
    for each pair:
    - first figure shows the average variance (solid line) +-1 sd (shaded regions) across the 20 trials
    - second figure shows the average lr (solid line) +-1 sd (shaded regions) across the 20 trials
    '''
    fig_eta, axes_eta = plt.subplots(2, 3, figsize=(15, 8))
    fig_var, axes_var = plt.subplots(2, 3, figsize=(15, 8))

    axes_eta = axes_eta.ravel()
    axes_var = axes_var.ravel()

    for i, pair in enumerate(pairs):

        # ---------- learning rate ----------
        eta = all_eta_histories[i]

        eta_mean = np.mean(eta, axis=0)
        eta_std = np.std(eta, axis=0)

        axes_eta[i].plot(eta_mean)
        axes_eta[i].fill_between(
            np.arange(T),
            eta_mean - eta_std,
            eta_mean + eta_std,
            alpha=0.2
        )

        axes_eta[i].set_title(f"q = {pair}")
        axes_eta[i].set_xlabel("t")
        axes_eta[i].set_ylabel(r"$\eta_t$")
        axes_eta[i].grid(True)

        # ---------- Beta variance --------
        var = all_var_histories[i]

        var_mean = np.mean(var, axis=0)
        var_std = np.std(var, axis=0)

        for j in range(2):
            axes_var[i].plot(
                var_mean[:, j],
                label=f"Expert {j+1}"
            )
            axes_var[i].fill_between(
                np.arange(T),
                var_mean[:, j] - var_std[:, j],
                var_mean[:, j] + var_std[:, j],
                alpha=0.2
            )

        axes_var[i].set_title(f"q = {pair}")
        axes_var[i].set_xlabel("t")
        axes_var[i].set_ylabel("Beta posterior variance")
        axes_var[i].grid(True)
        axes_var[i].legend()

    fig_eta.suptitle("Evolution of learning rate", fontsize=16)
    fig_var.suptitle("Evolution of Beta posterior variance", fontsize=16)
    fig_eta.tight_layout()
    fig_var.tight_layout()
    plt.show()

if __name__ == '__main__':
    import itertools
    
    seed = 2026
    rng = np.random.default_rng(seed=seed)
    pairs = [
        [.1,.25]
        , [.3,.25]
        ,[.9,.55]
        ,[.6,.95]
        , [.5,.5+1/10_000]
        , [.5,.5-1/10_000]
    ]
    thetas = map(float, np.linspace(start=10, stop=300, num=10).round(3))
    phis = map(float, np.linspace(start=.1, stop=5, num=10).round(3))
    combis = list(itertools.product(thetas, phis))
    print(f"{'hypers':<{17}} {'sum avg regret':>{14}} {'sd':>{10}}",flush=True)

    for comb in combis:
        summed_avg_regret = 0
        # all_eta_histories =[]
        # all_var_histories =[]
        var = 0
        for pair in pairs:
            regrets = np.zeros(M)
            # q_hats = np.zeros((M,2))
            # eta_histories = np.zeros((M,T))
            # var_histories = np.zeros((M,T,2))
            for i in range(M):
                regrets[i], _, _,_ = b_ada_hedge(
                    rng
                    , q1=pair[0]
                    , q2=pair[1]
                    , theta=comb[0]
                    , phi=comb[1]
                    )
            # all_eta_histories.append(eta_histories)
            # all_var_histories.append(var_histories)
            avg_regret = np.average(regrets)
            summed_avg_regret += avg_regret
            var_regret = np.var(regrets,ddof=1)
            var += var_regret
        print(f"{str(comb):<{17}} {summed_avg_regret:>{14}.3f} {np.sqrt(var):>{10}.3f}",flush=True)
    # plot_lr_var(pairs, all_var_histories, all_eta_histories)