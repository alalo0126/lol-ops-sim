"""numpy만으로 만든 작은 PPO. GPU·PyTorch 없이 kk PC(기기 VM 포함)와 클라우드 어디서나 같은 코드로 돈다.

정책망: 관측 → 128 tanh → 128 tanh → [주 행동 36, 와드 3, 강타 2] 로짓 (마스크 밖은 −∞)
가치망: 관측 → 128 tanh → 128 tanh → 1 (정책망과 따로)
행동 로그확률 = 주 행동 + 와드 + (캠프·카정·바위게일 때만) 강타.
"""
import numpy as np

from .agent import N_MAIN, N_WARD, N_SMITE, A_GANK

HEADS = [(0, N_MAIN), (N_MAIN, N_MAIN + N_WARD), (N_MAIN + N_WARD, N_MAIN + N_WARD + N_SMITE)]
N_OUT = N_MAIN + N_WARD + N_SMITE
NEG = -1e9


def init_mlp(rng, sizes, last_scale):
    params = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        scale = last_scale if i == len(sizes) - 2 else np.sqrt(2.0)
        W = rng.standard_normal((a, b)) * scale / np.sqrt(a)
        params += [W.astype(np.float64), np.zeros(b)]
    return params


def mlp_forward(params, x):
    """tanh 은닉층. 역전파용 중간값도 돌려준다."""
    acts = [x]
    h = x
    n = len(params) // 2
    for i in range(n):
        h = h @ params[2 * i] + params[2 * i + 1]
        if i < n - 1:
            h = np.tanh(h)
        acts.append(h)
    return h, acts


def mlp_backward(params, acts, gout):
    grads = [None] * len(params)
    n = len(params) // 2
    g = gout
    for i in reversed(range(n)):
        if i < n - 1:
            g = g * (1 - acts[i + 1] ** 2)
        grads[2 * i] = acts[i].T @ g
        grads[2 * i + 1] = g.sum(0)
        g = g @ params[2 * i].T
    return grads


def masked_log_softmax(z, mask):
    z = np.where(mask, z, NEG)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z) * mask
    return z - np.log(e.sum(-1, keepdims=True)), e / e.sum(-1, keepdims=True)


class Policy:
    def __init__(self, obs_dim, seed=0, hidden=128):
        rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.pi = init_mlp(rng, [obs_dim, hidden, hidden, N_OUT], 0.01)
        self.vf = init_mlp(rng, [obs_dim, hidden, hidden, 1], 1.0)
        self.rng = np.random.default_rng(seed + 1)

    # -------------------------------------------------------------- 저장
    def state(self):
        return {f"pi{i}": p for i, p in enumerate(self.pi)} | {f"vf{i}": p for i, p in enumerate(self.vf)} | {"obs_dim": np.array(self.obs_dim)}

    def load(self, st):
        self.pi = [st[f"pi{i}"] for i in range(len(self.pi))]
        self.vf = [st[f"vf{i}"] for i in range(len(self.vf))]
        return self

    def save(self, path):
        np.savez(path, **self.state())

    @classmethod
    def from_file(cls, path):
        st = np.load(path)
        p = cls(int(st["obs_dim"]))
        return p.load(st)

    # -------------------------------------------------------------- 행동
    def act(self, obs, m_main, m_ward, m_smite, greedy=False):
        """한 결정. 기록용 dict(관측·마스크·행동·로그확률·가치)를 함께 돌려준다."""
        x = obs[None, :].astype(np.float64)
        z, _ = mlp_forward(self.pi, x)
        v = float(mlp_forward(self.vf, x)[0][0, 0])
        masks = [m_main, m_ward, m_smite]
        acts, logp = [], 0.0
        for k, ((a, b), m) in enumerate(zip(HEADS, masks)):
            lp, p = masked_log_softmax(z[0, a:b], m)
            i = int(np.argmax(p)) if greedy else int(self.rng.choice(len(p), p=p))
            acts.append(i)
            if k < 2 or acts[0] < A_GANK:
                logp += lp[i]
        rec = {"obs": obs, "m": np.concatenate(masks), "a": np.array(acts), "logp": logp, "v": v}
        return acts[0], acts[1], acts[2], rec

    def values(self, X):
        return mlp_forward(self.vf, X)[0][:, 0]

    # -------------------------------------------------------------- 학습
    def loss_grads(self, X, M, A, logp_old, adv, ret, clip=0.2, ent_coef=0.01, vf_coef=0.5):
        z, acts = mlp_forward(self.pi, X)
        n = len(X)
        logp = np.zeros(n)
        gz = np.zeros_like(z)
        ent = np.zeros(n)
        use_smite = (A[:, 0] < A_GANK).astype(float)
        cache = []
        for k, (a, b) in enumerate(HEADS):
            lp, p = masked_log_softmax(z[:, a:b], M[:, a:b])
            w = use_smite if k == 2 else np.ones(n)
            logp += w * lp[np.arange(n), A[:, k]]
            plogp = np.where(M[:, a:b], p * lp, 0.0)
            H = -plogp.sum(1)
            ent += w * H
            cache.append((k, a, b, p, lp, w, H))
        ratio = np.exp(logp - logp_old)
        s1, s2 = ratio * adv, np.clip(ratio, 1 - clip, 1 + clip) * adv
        pg_loss = -np.minimum(s1, s2).mean()
        dL_dlogp = np.where(s1 <= s2, ratio * adv, 0.0) / n           # d(min)/d(logp)
        for k, a, b, p, lp, w, H in cache:
            onehot = np.zeros_like(p)
            onehot[np.arange(n), A[:, k]] = 1
            g_logp = (onehot - p) * (w * dL_dlogp)[:, None]
            g_ent = -p * (np.where(p > 0, lp, 0.0) + H[:, None]) * (w / n)[:, None]
            gz[:, a:b] = -(g_logp + ent_coef * g_ent)
        gpi = mlp_backward(self.pi, acts, gz)
        v, vacts = mlp_forward(self.vf, X)
        v = v[:, 0]
        v_loss = 0.5 * ((v - ret) ** 2).mean()
        gvf = mlp_backward(self.vf, vacts, (vf_coef * (v - ret) / n)[:, None])
        info = {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent.mean(),
                "approx_kl": float(((ratio - 1) - np.log(ratio)).mean()), "clipfrac": float((np.abs(ratio - 1) > clip).mean())}
        return gpi, gvf, info


class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_grads(grads, max_norm):
    norm = np.sqrt(sum((g ** 2).sum() for g in grads))
    if norm > max_norm:
        grads = [g * (max_norm / norm) for g in grads]
    return grads, norm


def gae(rewards, values, gamma, lam):
    """한 경기(끝 = 20:00, 다음 가치 0)의 이득과 반환값."""
    n = len(rewards)
    adv = np.zeros(n)
    last = 0.0
    for t in reversed(range(n)):
        nv = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * nv - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values
