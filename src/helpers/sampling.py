import torch
from torch import enable_grad
from tqdm import trange
from typing import Optional, Literal

# from k-diffusion
# https://github.com/crowsonkb/k-diffusion/blob/cc49cf6182284e577e896943f8e29c7c9d1a7f2c/k_diffusion/sampling.py#L585
# by Katherine Crowson
# implements DPM-Solver++(2M), Lu et al. 2022
# https://arxiv.org/abs/2211.01095
@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None, warmup_lms: Optional[Literal['2s', 'jvp']] = None, order2_until = 0):
    """
    DPM-Solver++(2M).
    warmup_lms can be used to warm up the linear multistep, beginning with a second-order step instead of a Euler step.
        the hope is to get a better approximation of the ODE trajectory at the beginning, where there is lots of uncertainty
        and the composition is being decided.
        2s: warm up with a DPM++(2S) ancestral step. this incurs an extra model invocation.
        jvp: use forward-mode autodiff to determine second-order derivative. this incurs an extra two model invocations (but could be made free by training the model to output the jvp, like in Nvidia GENIE)

    order2_until can be used to specify at which sigma you wish to switch to first-order steps, to reduce rainbow-style artifacting
    (note: makes the image softer/blurrier).
      https://github.com/crowsonkb/k-diffusion/issues/43#issuecomment-1781149170
    to end with *two* first-order steps, use:
      order2_until = sigmas[-3]
    the default is to end with *one* first-order step (i.e. from the final non-zero sigma down to 0).
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None and warmup_lms:
            # warmup the linear multistep so we can begin with a second-order step instead of a Euler step
            if warmup_lms == '2s':
                # use a DPM++(2S) step
                # Kat's idea
                # https://github.com/crowsonkb/k-diffusion/issues/43#issuecomment-1305196752
                r = 1 / 2
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
                denoised_i = model(x_2, sigma_fn(s) * s_in, **extra_args)
            elif warmup_lms == 'jvp':
                with enable_grad():
                    # TODO: use forward-mode autodiff to get second derivative of ODE
                    #       like in sample_ttm_jvp:
                    # https://github.com/Birch-san/sdxl-play/blob/edf5f53999c5f49a687f151292129756de373f4f/src/sample_ttm.py#L45
                    #       (note, sample_ttm approach should work too, but may return None depending on pytorch compatibility)
                    raise RuntimeError("jvp warmup not yet implemented.")
            else:
                raise ValueError(f"Unsupported warmup type '{warmup_lms}'. Supported values are: ['2s', 'jvp']")
        elif sigmas[i + 1] == 0 or old_denoised is None or sigmas[i] <= order2_until:
            denoised_i = denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_i = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
        x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_i
        old_denoised = denoised
    return x