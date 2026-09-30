import numpy as np
import matplotlib.pyplot as plt
import itertools


K = 2
T = 10_000
M = 20
PHI = 4
THETA = 200

def sample_loss(rng, q):
    success = rng.binomial(n=1, p=q, size=1).item()
    return 1 if success else -1

# cant use FTL, cant use EW

# reasoning: Adahedge does work with random losses
# however we still need to be able to observe all losses every round
# what if we can't? then take a non-informative prior: q_hat=0.5
# perform bayesian updating of p_hat.
# when the loss is not observe: take 2*p_hat - 1

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

    eta_history = np.zeros(T)
    var_history = np.zeros((T,K))

    # here A_t is Beta_t from the assignment
    A_t = 2*np.log(K)
    for t in range(T):
        eta = np.log(K)/(A_t + np.exp(-t/theta + phi))
        eta_history[t] = eta

        I_t = rng.choice(experts, size=1, p=p_exp).item()

        sampled_losses = [sample_loss(rng, q1), sample_loss(rng, q2)]

        if I_t == 0:
            losses = [sampled_losses[0], 2*q_hats[1]-1]
        else:
            losses = [2*q_hats[0]-1, sampled_losses[1]]
        if sampled_losses[I_t] == 1: alphas[I_t]+=1
        else: betas[I_t]+=1
        q_hats[I_t] = alphas[I_t]/(alphas[I_t]+betas[I_t]) #bayesian updating of p

        for j in range(K):
            alpha_beta = alphas[j]+betas[j]
            # this is the variance of the beta distro
            var_history[t,j] = (alphas[j]*betas[j]/(alpha_beta**2*(alpha_beta+1)))

        b_ada_regret += sampled_losses[I_t] - sampled_losses[smallest]
        cumul_losses += losses

        # below i changed how i compute p_t because of overflow errors
        log_weights = -eta*cumul_losses
        log_weights -= np.max(log_weights)
        p_exp = np.exp(log_weights)
        p_exp /= np.sum(p_exp)

        A_t += np.dot(p_exp, losses) + np.log(np.dot(p_exp, np.exp(-eta*np.array(losses))))/eta

    return b_ada_regret, q_hats, eta_history, var_history

def plot_lr_var(pair, all_var_histories, all_eta_histories):

    # How many individual runs to show?
    n_runs = 10
    all_var_histories = all_var_histories[:,:2000,:]
    all_eta_histories = all_eta_histories[:,:2000]

    # -------------------------------------------------
    # Learning rate: 10 individual runs
    # -------------------------------------------------

    fig_eta, ax_eta = plt.subplots(figsize=(10, 6))

    for i in range(n_runs):
        ax_eta.plot(
            all_eta_histories[i],
            label=f"Run {i+1}",
            alpha=0.8
        )

    ax_eta.set_title(f"Learning rate evolution — q = {pair}")
    ax_eta.set_xlabel("t")
    ax_eta.set_ylabel(r"$\eta_t$")
    ax_eta.grid(True)
    ax_eta.legend(loc='upper right')

    fig_eta.tight_layout()


    # -------------------------------------------------
    # Beta posterior variance: 10 individual runs
    # Separate subplot for each expert
    # -------------------------------------------------

    fig_var, axes_var = plt.subplots(1, 2, figsize=(14, 6))

    for j in range(2):
        for i in range(n_runs):
            axes_var[j].plot(
                all_var_histories[i, :, j],
                label=f"Run {i+1}",
                alpha=0.8
            )

        axes_var[j].set_title(f"Expert {j+1}")
        axes_var[j].set_xlabel("t")
        axes_var[j].set_ylabel("Beta posterior variance")
        axes_var[j].grid(True)
        axes_var[j].legend(loc='upper right')

    fig_var.suptitle(
        f"Beta posterior variance evolution — q = {pair}",
        fontsize=16
    )

    fig_var.tight_layout()

    plt.show()

if __name__ == '__main__':
    seed = 1234
    rng = np.random.default_rng(seed=seed)
    pairs = [
        [.1,.25]
        , [.3,.25]
        ,[.9,.55]
        ,[.6,.95]
        , [.5,.5+1/10_000]
        , [.5,.5-1/10_000]
    ]
    pair=pairs[5]
    print(f"{'pair':<{17}} {'avg regret':>{10}} {'sd':>{10}} {'q_hats':>{16}}",flush=True)


    thetas = map(float, np.linspace(start=40, stop=60, num=5).round(3))
    phis = map(float, np.linspace(start=.5, stop=1.5, num=7).round(3))
    combis = list(itertools.product(thetas, phis))
    for comb in combis:
        regrets = np.zeros(M)
        q_hats = np.zeros((M,2))
        # eta_histories = np.zeros((M,T))
        # var_histories = np.zeros((M,T,2))
        for i in range(M):
            regrets[i], q_hats[i], _, _ = b_ada_hedge(rng, q1=pair[0], q2=pair[1], theta=comb[0], phi=comb[1])
        avg_regret = np.average(regrets)
        sd_regret = np.std(regrets,ddof=1)
        print(f"{str(comb):<{17}} {avg_regret:>{10}.3f}  {sd_regret:>{10}.3f} {str(np.average(q_hats, axis=0).round(3)):>{16}}",flush=True)
    # print('summed average regret:', round(summed_avg_regret,3))

    # plot_lr_var(pairs, var_histories, eta_histories)