# %pip install torch
import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
from sklearn.datasets import make_moons
import wandb as wnb

class DiscreteFlow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 128, v: int = 128):
        super().__init__()
        self.v = v
        self.embed = nn.Embedding(v, h)
        self.net = nn.Sequential(
            nn.Linear(dim * h + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim * v))

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t[:, None], self.embed(x_t).flatten(1, 2)), -1)).reshape(list(x_t.shape) + [self.v])
    
def chi_square_test(dset1, dset2):
    """
    Perform chi-squared test on two datasets.
    """
    from scipy.stats import chi2_contingency
    import pandas as pd
    
    s1 = pd.Series(dset1, name='X1')
    s2 = pd.Series(dset2, name='Xt')
    contingency_table = pd.crosstab(s1, s2)
    
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-squared test: chi2={chi2}, p-value={p}")
    return chi2, p
    
if __name__ == "__main__":
    # Visualize the dataset
    batch_size = 1000
    x_1 = make_moons(batch_size, noise=0.0)[0]
    plt.hist(x_1[:, 0], bins=50, alpha=0.5, label='x1[:, 0]')
    plt.hist(x_1[:, 1], bins=50, alpha=0.5, label='x1[:, 1]')
    plt.title("Distribution of x1")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    # Training
    batch_size = 256
    vocab_size = 128
    lr = 0.001
    num_epochs = 10000

    model = DiscreteFlow(v=vocab_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    use_wnb = False
    if use_wnb:
        wnb.init(project="Contact Force Estimation", name="DiscreteFlowMatchingMoon")
        wnb.watch(model, log="all")
        
        # log hyperparameters
        wnb.config.update({
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "learning_rate": lr,
            "num_epochs": num_epochs,
        })

    for epoch in range(num_epochs):
        x_1 = Tensor(make_moons(batch_size, noise=0.05)[0])
        x_1 = torch.round(torch.clip(x_1 * 35 + 50, min=0.0, max=vocab_size - 1)).long()
        # x_1 = torch.round(torch.clip(x_1 * 350 + 500, min=0.0, max=vocab_size - 1)).long()
        
        x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, 2))

        t = torch.rand(batch_size)
        x_t = torch.where(torch.rand(batch_size, 2) <  t[:, None], x_1, x_0)

        logits = model(x_t, t)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # t_vis = t.flatten().detach().cpu().numpy()
        # plt.hist(t_vis, bins=50, density=True)
        # plt.title("Distribution of t_flat")
        # plt.xlabel("t_flat")
        # plt.ylabel("Density")
        # plt.show()
        
        if use_wnb:
            wnb.log({"loss": loss.item()})
            wnb.log({"learning_rate": optim.param_groups[0]['lr']})
                
    # Sampling
    x_t = torch.randint(low=0, high=vocab_size, size=(200, 2))
    t = 0.0
    div = 1
    h_c = 0.1 / div
    results = [(x_t, t)]
    results_x_t = []
    while t < 1.0 - 1e-3:
        p1 = torch.softmax(model(x_t, torch.ones(200) * t), dim=-1)
        h = min(h_c, 1.0 - t)
        one_hot_x_t = nn.functional.one_hot(x_t, vocab_size).float()
        u = (p1 - one_hot_x_t) / (1.0 - t)
        x_t = torch.distributions.Categorical(probs=one_hot_x_t + h * u).sample()
        t += h
        results.append((x_t, t))
        results_x_t.append(x_t)

    if div > 1:
        num_figs = len(results) // div + 1
    else:
        num_figs = len(results) // div
    fig, axes = plt.subplots(1, num_figs, figsize=(15, 2), sharex=True, sharey=True)

    count = 0
    for (x_t, t) in results:
        if count % div == 0:
            ax = axes[count // div]
            ax.scatter(x_t.detach()[:, 0], x_t.detach()[:, 1], s=10)
            ax.set_title(f't={t:.1f}')
        count += 1

    # perform chi-squared test
    results_x_t = torch.stack(results_x_t)
    results_x_t = results_x_t.flatten()
    dset_x1 = Tensor(make_moons(len(results_x_t), noise=0.05)[0])
    dset_x1 = torch.round(torch.clip(dset_x1 * 35 + 50, min=0.0, max=vocab_size - 1)).long()
    dset_x1 = dset_x1.flatten()
    chi2, p = chi_square_test(dset_x1.numpy(), results_x_t.numpy())
    
    if use_wnb:
        wnb.log({"sampled_moons": wnb.Image(fig)})
        wnb.finish()
    plt.tight_layout()
    plt.show()
