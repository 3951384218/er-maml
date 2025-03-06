#Inner loop: grad norm
policy_grad = torch.autograd.grad(policy_loss, new_policy.parameters(),
                                          retain_graph=True)
        grad_ratio = params['grad_rate']   # 0.001
        g_norm = []
        for x in policy_grad:
            if x is not None:
                norm = 1 / (6 * torch.norm(x))
            else:
                norm = 0.0
            g_norm.append(norm)

        theta = [p.data for p in new_policy.parameters()]
        theta_ = [j + grad_ratio * g_norm[idx] for idx, j in enumerate(theta)]
        for idx, t in enumerate(new_policy.parameters()):
            t.data = theta_[idx]
        theta__densities = new_policy.density(states)
        theta__log_probs = theta__densities.log_prob(actions).mean(dim=1, keepdim=True)
        loss_ = ppo.policy_loss(theta__log_probs, old_log_probs, advantages, clip=0.2)
        grad_ = torch.autograd.grad(loss_, new_policy.parameters(), allow_unused=anil)
        grads = []
        for g, t_g in zip(policy_grad, grad_):
            if g is not None:
                grad = (1 - params['norm_a']) * g + params['norm_a'] * t_g
                grads.append(grad)
            # else:
            #     grads.append(None)
        mean_grad = [g.detach() for g in grads]
        mean_grads += parameters_to_vector(mean_grad)
        del policy_grad, grad_

# evo grad
def evo_actor(policy, states, actions, old_log_probs, advantages, sigma, temp, n_model=2):
    train_actor = [i.detach() for i in get_func_params(policy)]
    actor_loss_lt = []
    actor_list = [[j + sigma * torch.sign(torch.randn_like(j)) for j in train_actor] for i in
                   range(n_model)]
    for theta in actor_list:
        for idx, t in enumerate(policy.parameters()):
            t.data = theta[idx]
        evo_densities = policy.density(states)
        evo_log_probs = evo_densities.log_prob(actions).mean(dim=1, keepdim=True)
        actor_loss = ppo.policy_loss(evo_log_probs, old_log_probs, advantages, clip=0.2)
        actor_loss_lt.append(actor_loss.item())
    weights = torch.softmax(-torch.tensor(actor_loss_lt) / temp, 0)
    actor_updated = [sum(map(mul, param, weights)) for param in zip(*actor_list)]

    for idx, t in enumerate(policy.parameters()):
        t.data = actor_updated[idx]
    return policy


def meta_evo_update(iter_replays, iter_policies, policy, baseline, params):
    evo_grads = 0.0
    for task_replays, old_policy in zip(iter_replays, iter_policies):
        train_episodes = task_replays[:-1][-1]
        valid_episods = task_replays[-1]
        new_policy = clone_module(policy)

        # train_episodes = train_replays[-1]
        train_states, train_actions, train_rewards, train_dones, train_next_states = get_episode_values(train_episodes, device=params['device'])
        train_old_densities = new_policy.density(train_states)
        train_advantages = compute_advantages(baseline, params['tau'], params['gamma'], train_rewards, train_dones, train_states, train_next_states)
        train_advantages = ch.normalize(train_advantages).detach()
        train_old_log_probs = train_old_densities.log_prob(train_actions).mean(dim=1, keepdim=True).detach()

        # Calculate KL from the validation episodes
        states, actions, rewards, dones, next_states = get_episode_values(valid_episods, device=params['device'])
        old_densities = new_policy.density(states)
        advantages = compute_advantages(baseline, params['tau'], params['gamma'], rewards, dones, states, next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()

        # todo change evo
        for i in range(3):
            evo_policy = evo_actor(new_policy, train_states, train_actions, train_old_log_probs, train_advantages,
                                   params['sigma'], params['temp'], params['n_model'])
            evo_densities = evo_policy.density(states)
            evo_kl = kl_divergence(evo_densities, old_densities).mean()
            evo_log_probs = evo_densities.log_prob(actions).mean(dim=1, keepdim=True)
            evo_valid_loss = trpo.policy_loss(evo_log_probs, old_log_probs, advantages)
            if evo_kl < params['max_kl']:
                break
        evo_valid_gradients = torch.autograd.grad(evo_valid_loss, evo_policy.parameters(), retain_graph=False,
                                                  create_graph=False)
        evo_grad = [g.detach() for g in evo_valid_gradients]
        evo_grads += parameters_to_vector(evo_grad)
        del evo_policy

    evo_grads /= len(iter_replays)
    evo_grads_ = [torch.zeros_like(p.data) for p in policy.parameters()]
    vector_to_parameters(evo_grads, evo_grads_)
    for (name, p), grads in zip(policy.named_parameters(), evo_grads_):
        if 'head' in name:
            p.data = p.data - grads * params['evo_lr']

