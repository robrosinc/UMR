# Local HiPHI conversion resources

Place the licensed files needed by `scripts/convert_hiphi_to_umr.sh` here:

```text
external_assets/hiphi/SMPLX_NEUTRAL.npz
external_assets/hiphi/beta_fit/neutral_smpl_mean_params.h5
external_assets/hiphi/beta_fit/gmm_08.pkl
```

The converter uses these paths by default. The files are kept out of Git.
Other UMR pipelines continue to use the body models in `smpl/`.
