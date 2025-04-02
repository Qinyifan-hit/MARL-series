import numpy as np
def eval_func(env, models, e_turns, agents):
    num_agent = len(agents)
    score = np.zeros(num_agent)
    for _ in range(e_turns):
        s, _ = env.reset()
        done = False
        while not done:
            s_list = [np.array(s[agent], dtype=np.float32) for agent in agents]
            a_list = []
            for a_id in range(num_agent):
                a_list.append(models[a_id].action_select(s_list[a_id], True))
            a = {agents[a_id]: a_list[a_id] for a_id in range(num_agent)}
            s_, r, dw, tr, _ = env.step(a)
            dones = {agent: (dw[agent] or tr[agent]) for agent in agents}
            done_list = [np.array(dones[agent], dtype=np.bool_) for agent in agents]

            done = any(done_list)
            for a_id in range(num_agent):
                score[a_id] += r[agents[a_id]]
            s = s_

    return score / e_turns
