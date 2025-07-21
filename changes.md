# Implementation Plan for Upgrading Pixel-Space Diffusion Model

## Core Architecture Upgrades

1. **Multi-Scale Edge Prompts with Guided Filter Modules (GFM):**
   *Changes:* We will incorporate a frozen edge detector (e.g. PiDiNet or Canny) to extract edge maps from the low-dose input at multiple scales, and inject these edge features into the UNet decoder via guided filtering modules. This helps preserve structural details (edges) in the generated full-dose output.
   *Code Insertion:*

   * **Edge Detection:** Use a pre-trained edge detector (frozen) to compute an edge map from the low-dose image (`cond`). This can be done **outside the model** (e.g., in the data loader for efficiency) or at the start of `forward`. For example, in `CTPairsDataset.__getitem__`, after obtaining `img_q` (low-dose) and before transforming to tensor, compute its edges: if using Canny (via OpenCV or skimage), convert `img_q` to grayscale and apply `cv2.Canny`; if using PiDiNet, load the PiDiNet model once and run it on `img_q`. Ensure the edge map is normalized (0-1) and resized to the input resolution (256×256). Return it along with `ld, nd`.
   * **Model Input Adjustment:** Update `PixelDiffusionUNetConditional.forward` (and possibly `__init__`) to handle the edge prompt. Two options:
     **(a)** **Include edges as an additional input channel at start:** Change `init_conv` to accept 3 channels (noisy image + cond + edge). For instance, `self.init_conv = nn.Conv2d(3, base_ch, 3, padding=1)`. In `forward`, do `h = self.init_conv(torch.cat([x_t, cond, edge_map], dim=1))`. This injects edge info at the lowest level.
     **(b)** **Inject edges at multiple decoder levels (preferred):** Maintain the existing two-channel `init_conv` (for `x_t` and `cond` concatenation), and instead feed edges into the *decoder skip connections* via GFM. In `__init__`, create a list of **Guided Filter Modules**: `self.gfms = nn.ModuleList([...])` with length equal to the number of skip connections (e.g., 3 for a 4-level UNet). Each GFM takes the decoder’s skip feature and the edge map (at the corresponding scale) to produce a refined skip feature. A simple implementation of GFM could be a small convolutional block: for example, a 3×3 conv that inputs `[skip_feat, edge_feat]` (concatenated along channel) and outputs the same number of channels as `skip_feat`, optionally followed by a normalization (BatchNorm/GroupNorm) and activation. This acts like a learned guided filter, using edge information to modulate features.

     * **Edge Pyramid:** Compute downscaled edge maps for each decoder level. E.g., if the UNet has skip connections at 64×64, 128×128, and 256×256 resolutions (given input 256×256), prepare `edge_64, edge_128, edge_256`. If edge detection is done outside, you can downsample the full-res edge: use `torch.nn.functional.interpolate` or average pooling to match the spatial size of each skip. If done inside `forward`, compute once and cache: `edge_full = edge_detector(cond)` (result 1×256×256), then `edge_128 = F.avg_pool2d(edge_full, 2)`, `edge_64 = F.avg_pool2d(edge_full, 4)` (assuming powers of 2 downsampling). These are \*\*kept **detached** (no grad) since the detector is frozen.
     * **Injecting at Decoder:** In `forward`, after computing encoder `skips` in the downsampling loop, integrate edges during upsampling. For each upsample block, do:

       ```python
       h = block["up"](h)                      # upsample current feature
       skip_feat = skips.pop()                 # get corresponding skip feature from encoder
       edge_feat = edge_pyramid.pop()          # get edge map at this scale (pop if prepared as stack)
       # Apply guided filter module
       skip_guided = self.gfms[i](torch.cat([skip_feat, edge_feat], dim=1))  
       h = torch.cat([h, skip_guided], dim=1)  # fuse skip (conditioned on edges) with current feature
       ```

       This way, at multiple decoder levels the features are explicitly guided by edge information. The guided filter module can learn to preserve or sharpen edges in the skip feature based on the edge map (for example, amplifying feature responses along edges and suppressing noise in homogeneous regions).
   * **Verification:** Ensure channel dimensions match: if skip feature has C channels and edge is 1-channel, `self.gfms[i]` should input `C+1` channels and output `C`. Also confirm spatial sizes align (they should, by design of skip connections). This change means the model expects an edge map in addition to `ld` and `nd`. If modifying the dataset to return edges, adjust training loop to unpack this. If computing edges inside `forward`, pass `cond` to the edge detector internally.
   * **Freezing Edge Detector:** Mark the edge detector’s parameters with `requires_grad=False` (if it’s a model). If using an algorithm like Canny, no gradients anyway. This ensures the edge extraction is a fixed operation, providing consistent structural guidance.
   * **Staged Implementation:** This feature can be introduced first. Initially, you can test with a simple edge detector like Canny (no learning) and a basic GFM (even identity or a single conv) to see that the model can incorporate the extra channel. After integration, expect the UNet to better reconstruct edges (check outputs for sharper boundaries and improved SSIM on edges). You can later swap in PiDiNet for potentially better learned edges once the pipeline works.

2. **Global Cross-Attention for Conditioning Input:**
   *Changes:* Add Transformer-style cross-attention layers to allow the UNet to directly attend to the low-dose conditioning image’s features globally. Currently, the model only concatenates `cond` at input and relies on local convolution/skip connections to propagate that information. By introducing cross-attention, every part of the image can contextually adjust based on the entire conditioning image, potentially improving coherence and utilization of input details.
   *Code Insertion:*

   * **Cond Feature Encoder:** Create a small **conditioning encoder** to produce feature maps or tokens from the low-dose image for attention. For example, in `PixelDiffusionUNetConditional.__init__`, add a simple conv block or downsample layers:

     ```python
     self.cond_encoder = nn.ModuleList([
         nn.Conv2d(1, base_ch, 3, padding=1), nn.ReLU(),
         nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU()
     ])
     ```

     This can act on the full-resolution cond or a downsampled cond. Alternatively, reuse the encoder of the UNet for cond: e.g., pass `cond` through a parallel downsampling path. For simplicity, you might **downsample cond to a low resolution** used in attention (like 32×32 or 64×64) on the fly.
   * **Cross-Attention Layer:** Define a module for cross-attention. You can use PyTorch’s `nn.MultiheadAttention` for this. For instance, add in `__init__`:

     ```python
     self.cross_attn_mid = nn.MultiheadAttention(embed_dim=curr_ch, num_heads=4, batch_first=True)  
     self.cond_proj = nn.Conv2d(1, curr_ch, 1)  # to project cond features to same dim  
     ```

     Here `curr_ch` is the channel count at the bottleneck (e.g., 512 in the original design). We will do cross-attention at the bottleneck as a starting point (most global receptive field). You could similarly add `cross_attn` modules for other resolutions if needed (e.g., at 64×64 features), but integrating one at the bottleneck is a good initial step.
   * **Integrate in Forward:** After the UNet’s bottleneck (`self.mid`), perform cross-attention between the bottleneck feature map and the cond feature map:

     ```python
     # Assume h is [B, curr_ch, 32, 32] after bottleneck  
     B, C, H, W = h.shape  
     # Prepare query (UNet features) and key/value (cond features)  
     query = h.view(B, C, H*W).permute(0, 2, 1)    # shape [B, HW, C]  
     cond_down = F.avg_pool2d(cond, kernel_size=8)  # downsample low-dose to 32x32 (if input 256x256)  
     cond_feat = self.cond_proj(cond_down)          # shape [B, C, 32, 32]  
     key = cond_feat.view(B, C, H*W).permute(0, 2, 1)   # [B, HW, C]  
     value = key  # use same for value  
     attn_out, _ = self.cross_attn_mid(query, key, value)   # attn_out: [B, HW, C]  
     attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)  # reshape to [B, C, 32, 32]  
     h = h + attn_out   # add as residual (could also LayerNorm before/after if needed)  
     ```

     This allows each bottleneck feature to attend to all positions of the low-dose image (32×32 version). For example, if a region in the output needs information from a distant part of the input (say, a matching anatomical structure or overall intensity context), the cross-attention can fetch it.

     * **Normalization:** It’s often useful to normalize either queries and keys or apply a `nn.LayerNorm(C)` to the sequences before/after attention for stability. Consider adding `self.attn_norm = nn.LayerNorm(C)` and do `attn_out = self.attn_norm(attn_out)` or similar.
     * **Multiple Layers:** If needed, you could stack another ResNet block after this to mix the attended features. But initially, a single residual attention injection is fine.
   * **At Other Scales:** For more extensive conditioning, you could add cross-attention in earlier layers too (e.g., after the attention block at 64×64 or 128×128). That would require downsampling cond to those sizes and perhaps smaller embed\_dim. You might create `self.cross_attn64 = MultiheadAttention(embed_dim=out_ch, ...)` for the 64×64 layer (where out\_ch might be 256 in original). Then in the upsampling loop, after `block["attn"](h)` at that scale, perform a similar attention with cond\_down64. However, these additions increase complexity – it’s wise to verify the mid-layer attention helps before adding more.
   * **Data Flow:** We continue to pass `cond` normally into the network (for init conv and skip connections), so this cross-attention is *in addition* to existing conditioning methods. It provides a global path for information.
   * **Validation:** With cross-attention, monitor training for any sign of instability (sometimes attention can cause spikes in loss if not handled). If issues arise, reduce learning rate for attention layers or add grad clipping. During inference, the outputs should better respect global structures of `cond`. For example, if there’s a faint structure in `ld` that spans large areas, cross-attention helps the model consistently enhance it in the output.
   * **Minimal Disruption Approach:** Implement one cross-attention (at bottleneck) and get it working. You can initially disable it to compare (e.g., run baseline vs with cross-attn). Then incrementally add attention to other scales if needed. This upgrade can be done after edge/GFM integration is in place, as it doesn’t conflict with it.

3. **Implicit Conditional Representation (ICR) Head for Output Refinement:**
   *Changes:* Introduce an **ICR head** at the end of the UNet to refine the output in pixel space using coordinate-conditioned prediction. The idea is to use an implicit neural representation conditioned on spatial coordinates to capture fine details that a standard UNet might miss. This head will take the coarse output (from the UNet decoder) along with spatial coordinates (and possibly the low-dose image) and output a corrected final image. This is inspired by techniques where a coordinate-based MLP can represent high-frequency details.
   *Code Insertion:*

   * **ICR Module Definition:** In `PixelDiffusionUNetConditional.__init__`, after building the UNet layers, add something like:

     ```python
     coord_channels = 2  # (x, y)
     icr_in_ch = 1 + coord_channels + 1  # coarse output + coords + optional cond
     self.icr_head = nn.Sequential(
         nn.Conv2d(icr_in_ch, 32, 1), nn.ReLU(),
         nn.Conv2d(32, 32, 1), nn.ReLU(),
         nn.Conv2d(32, 1, 1)
     )
     ```

     Here, `icr_in_ch` is 4 if we include: the coarse output (1 channel), the cond image (1 channel, optional), and 2 coordinate channels. The ICR head is implemented as 1×1 convolutions with non-linearities, which effectively act like an MLP applied at each pixel (no receptive field beyond that pixel). This is crucial: by using 1×1 convs, each output pixel is a function of the corresponding pixel in coarse output, the corresponding pixel in `cond`, and the absolute position – making the prediction *coordinate-conditional*. The hidden layer size (32) can be adjusted; more might capture more complex dependencies but also risk overfitting noise.

     * Alternatively, for a more expressive model, you could use a small MLP that takes in the coordinate (x,y normalized) and maybe some pooled global info. However, the 1×1 conv approach with coordinate channels is simpler and differentiable as part of the CNN graph.
   * **Coordinate Channels:** Implement a utility to create coordinate grids. For example, inside `forward` (or as a buffer):

     ```python
     if not hasattr(self, "coords"):
         # Create a normalized coordinate grid [B,2,H,W] with values in [-1,1]
         grid_y, grid_x = torch.meshgrid(torch.linspace(-1,1,H, device=h.device), 
                                         torch.linspace(-1,1,W, device=h.device))
         coords = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
         self.coords = coords.unsqueeze(0)  # [1,2,H,W] to broadcast for batch
     coord_feats = self.coords.expand(B, -1, H, W)
     ```

     Ensure `H,W` here are the spatial dimensions of the output (256×256 after upsampling). This provides two channels where each pixel contains its normalized (x,y) location.
   * **Forward Pass Integration:** At the end of `forward` after obtaining the UNet’s output:

     ```python
     coarse = self.final(h)   # coarse output from UNet, shape [B,1,H,W], this predicts ε or image depending on training mode  
     # If the UNet is predicting ε (noise), you might first convert it to image prediction x0; but easier is to train UNet final to output the image directly by parameterization – see Loss section for details. We assume coarse is the image prediction for refinement.
     ld_fullres = cond  # cond is already [B,1,H,W] from input (assuming input was 256 crop), if not, upsample it to H,W.
     inp_icr = torch.cat([coarse, ld_fullres, coord_feats[:B]], dim=1)  # [B, 1+1+2, H, W]
     refined = self.icr_head(inp_icr)  # [B,1,H,W]
     return refined  
     ```

     Now the model’s output is this refined image. The ICR head can correct any residual errors in the coarse output by leveraging absolute position (for example, to fix any spatial bias or add high-frequency details that depend on location) and the local low-dose info. It effectively learns a mapping: (coarse\_pixel, cond\_pixel, x, y) -> refined\_pixel. This is powerful for capturing effects like spatially-varying noise patterns or biases that the UNet might not uniformly learn.
   * **Training Considerations:** The ICR head will be trained jointly with the UNet. Because it is lightweight, it can quickly learn to make small adjustments. One strategy is to have it predict a **residual** to add to the coarse output rather than the absolute value, which can make learning easier (since initial coarse output from UNet should be close to target). In that case, change the last line to `refined = coarse + self.icr_head(inp_icr)` and train the ICR head to predict the difference. This often stabilizes training (the head won’t learn to output large values from scratch, just corrections).
   * **Loss Computation:** When computing loss (discussed later), use the `refined` output against the ground truth full-dose image. The coarse output is an intermediate – we don’t necessarily supervise it directly (though you could add a secondary loss on coarse as well). The ICR head being coordinate-conditional means it can, for instance, learn that certain positions (e.g., image corners or air regions) should always be zero (if that’s the case in full-dose), or it can imprint subtle high-frequency textures that were missing.
   * **Memory/Compute:** The ICR head adds minimal overhead (few 1×1 convs). Coordinate channels add 2 to input channels which is negligible. This is a safe addition in terms of model size.
   * **Staged Implementation:** You might introduce the ICR head after getting the base UNet (with edges and attention) working with basic losses. Initially, the UNet alone might output the final image; then you attach ICR and allow it to fine-tune the output. This can be done by either training from scratch with it or fine-tuning an existing model (since it’s small, many prefer to include it from scratch). If fine-tuning, one approach is to freeze the main UNet for a few epochs and train only `icr_head` so it learns the residuals, then unfreeze everything to jointly optimize. Monitor metrics like L1/SSIM – the ICR head should drive those down further as it corrects small errors.

4. **Schrödinger-Bridge-Inspired Diffusion Process:**
   *Changes:* Modify the diffusion forward and reverse process to treat the low-dose image as the start point of diffusion and the full-dose as the end point, rather than starting from pure noise. In essence, we want to **bridge** from the low-dose distribution to the full-dose distribution using a diffusion-like process. This is inspired by Schrödinger Bridges, which find stochastic processes connecting two distributions. Practically, we will adjust how noising is done during training and how sampling is initialized, so the model learns to transform low-dose into full-dose over the diffusion trajectory.
   *Implementation Plan:*

   * **Forward Process Adjustment:** In the standard diffusion, the forward process defines \$q(x\_t | x\_{t-1})\$ by gradually adding noise to a clean image (here the clean image is full-dose ND). We want to incorporate the low-dose (LD) image into this forward path. One approach: treat a **mixture of LD and ND** as the state that gets noised. For example, redefine the forward diffusion at step *t* as:
     $x_t = \sqrt{\alpha_t}\, x_0^{ND} \;+\; \sqrt{1-\alpha_t - \gamma_t}\, x_0^{LD} \;+\; \sqrt{\gamma_t}\, \epsilon,$
     where \$x\_0^{ND}\$ is the full-dose image (target), \$x\_0^{LD}\$ is the low-dose image (condition), and \$\epsilon\$ is standard Gaussian noise. Here \$\alpha\_t\$ and \$\gamma\_t\$ are schedules that partition the variance. For instance, at \$t=0\$, we want \$x\_0 \approx x\_0^{LD}\$ (so set \$\alpha\_0 \approx 0, 1-\alpha\_0-\gamma\_0 \approx 1\$, meaning the state is mostly the low-dose image plus almost no noise). At \$t=T\$ (final diffusion step), we might want the state to be mostly noise with a bias towards ND. One intuitive design is to let the process **drift from LD to ND**: start at LD (plus tiny noise), end at ND + noise.

     * A simpler linear schedule could be: at \$t=0\$, \$x\_0 = x\_0^{LD}\$; at \$t=T/2\$, \$x\_{T/2}\$ is an even mix of LD and ND (plus moderate noise); at \$t=T\$, \$x\_T\$ is essentially ND corrupted with full noise. This way, the diffusion bridges the two images.
     * To implement: we can define a custom function `forward_diff_bridge(ld, nd, t)` that mixes `ld` and `nd` as above. We might not need an explicit \$\gamma\_t\$ separate from \$1-\alpha\_t\$; instead, interpret \$\sqrt{\alpha\_t} x^{ND} + \sqrt{1-\alpha\_t} \epsilon\$ as usual diffusion, but we **initialize** the process at \$x\_{0}=x\_0^{LD}\$ instead of \$x\_0^{ND}\$. This is non-standard (since normally \$x\_0\$ is the data), but we can simulate it by “pretending” the low-dose is a slightly noised version of ND at some step.
   * **Training Algorithm:** There are a couple of ways to train the model under this new process:
     **Method 1: Two-Forward Pass Bridging:** For each training sample, perform the standard diffusion of ND (as before) *and* incorporate a step that starts from LD. However, training two processes simultaneously is complex. Instead, we can **augment the training objective** to sometimes use LD as the starting point for noising:

     * For a given batch, with some probability, sample a time step \$t\_{br}\$ near 0 (early in diffusion). For those samples, set the noisy input explicitly to the low-dose image: \$x\_{t\_{br}} = x\_0^{LD}\$ (which is off the normal path, as normally \$x\_{t\_{br}} = \sqrt{\bar\alpha\_{t\_{br}}}x\_0^{ND} + \sqrt{1-\bar\alpha\_{t\_{br}}}\epsilon\$). Compute what noise would be required to turn ND into LD at that \$t\_{br}\$: \$\tilde\epsilon = (x\_0^{LD} - \sqrt{\bar\alpha\_{t\_{br}}}x\_0^{ND})/\sqrt{1-\bar\alpha\_{t\_{br}}}\$. Then train the model to predict \$\tilde\epsilon\$ from input \$(x\_{t\_{br}}=x\_0^{LD}, t\_{br}, cond=LD)\$. Essentially, we’re asking the model: “if at time \$t\_{br}\$ the state *is exactly the low-dose image*, what noise would you add to move towards the full-dose image?” This teaches the model that a low-dose image at a given noise level should be guided towards the full-dose.
     * The rest of the time, train as usual (diffusing ND and predicting noise). Over training, the model will then see both ordinary diffusion scenarios and these special bridging scenarios. This encourages it to handle cases where the input already resembles the low-dose image.
     * You may choose \$t\_{br}\$ = 0 or a small value (like 50 out of 1000) because we want LD to be considered a near-start state. Using \$t\_{br}=0\$ directly means predicting the difference from LD to ND in one shot, which might be too hard; a small \$t\$ means there’s some noise mixed in to soften the direct mapping.
       **Method 2: Modified \$q(x\_t|x\_{t-1})\$ Schedule:** Alternatively, redefine the diffusion process entirely: e.g., set \$x\_0 = LD\` instead of ND, and treat \$x\_T\$ as random noise (like ND is never directly used in forward). But then the model would be trying to denoise from pure noise to LD (which is not what we want). Instead, we want to end at ND. A true Schrödinger Bridge solution would require iteratively matching distributions – which is advanced. We can approximate by the above approach or by altering the sampling procedure.
   * **Sampling (Reverse Process) Changes:** The most straightforward use of this concept is at inference time: **start the reverse diffusion from the low-dose image instead of pure noise.** There are known techniques in diffusion models akin to “warm start” or using a guiding distribution. Concretely:

     * In `sample_ddim_guided` (or your sampler), instead of setting `x = torch.randn_like(ld_img)` at the beginning, initialize `x` as a noised version of the low-dose: e.g.,

       ```python
       start_step = int(0.8 * scheduler.num_train_timesteps)  # e.g. start 80% through the diffusion steps  
       x = ld_img  
       t = start_step  
       alpha_bar = scheduler.alphas_cumprod[t]  
       # add noise to ld_img at level t  
       x = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * torch.randn_like(x)  
       # then proceed the reverse diffusion from t downwards  
       for t_idx in reversed(range(0, start_step+1)):  
           ...  # standard reverse diffusion steps  
       ```

       This means we skip adding noise for the first 20% of steps (effectively treating those as if the process already drifted from LD to some intermediate). We inject the appropriate amount of noise into LD to reach the diffusion state at `t = start_step`, and then run the model from there. This way, the output is achieved in fewer steps and leverages the actual low-dose as a starting point.
     * If we incorporate the training adjustments (Method 1 above), the model will have learned to handle such scenarios (where input at some t is actually the LD image). If not, you can still try this trick as a post-training heuristic; often diffusion models can denoise from an image input with some degradation even if not explicitly trained for it.
     * The result is a **diffusion bridge** from LD to ND: initially, the image is mostly the LD image (with some noise), and as t decreases, the model removes noise and adds detail, effectively transforming LD -> ND.
   * **Coding Notes:** You will need to modify the training loop (`train_with_eval`) to implement Method 1: perhaps randomly choose some batches or fraction of iterations to do the LD-bridging noise target as described. Also modify the sampler to allow an initial image. For example, write a new function `sample_from_ld(model, ld_img, scheduler, start_step)` that does as above. This might be similar to how image-to-image diffusion works (like in Stable Diffusion’s img2img, which adds noise to an input image and then denoises).
   * **Evaluation:** After training with the bridge method, test the model’s sampling both from pure noise (should still work, perhaps producing a generic output given LD condition) and from the noised-LD start. Ideally, the latter gives better results (since it preserves more of LD’s structure and requires fewer steps). If the model is properly trained, you’ll notice it performs the transformation more like a direct denoiser. Also check that the model doesn’t simply copy the LD input (the guidance and loss ensure it moves to ND domain).
   * **Summary:** This is the most advanced feature and might require experimentation to get the schedule right. Start by implementing the inference part (starting from noised LD) and see if the current model can handle it. Then incorporate the training adjustments to explicitly teach the model that behavior. The payoff is potentially a more efficient and deterministic conversion from low-dose to full-dose (less reliance on large noise sampling, and closer to a single-step mapping). Document and monitor how this affects training loss and output quality; you might see faster convergence in later timesteps or the model focusing on refining details rather than creating structure from pure noise.

5. **Enhanced Loss Function (MSE + L1 + LPIPS + adversarial):**
   *Changes:* Use a composite loss to better guide the model’s training. The current implementation uses only MSE on the predicted noise (which correlates to MSE on the output image in diffusion). We will add **L1 loss** for direct pixel fidelity, **LPIPS perceptual loss** for visual quality, and optionally **adversarial loss** for realism. Combining these makes the training objective richer: MSE ensures correctness in denoising, L1 encourages accurate CT numbers, LPIPS pushes for perceptual similarity (sharper, more like the target’s texture), and GAN loss can help reduce blurriness by encouraging outputs indistinguishable from real full-dose.
   *Code Insertion:*

   * **Calculate Predicted Image (x0):** In the training loop (`train_with_eval` function), after obtaining `pred = model(x_t, t, cond)`, we typically interpret `pred` as the model’s prediction of the noise \$\epsilon\$. The ground truth noise is stored as `noise`. To incorporate image-space losses, convert this into a prediction of the denoised image \$x\_0\$. Using the diffusion formulation:

     ```python
     # pred is epsilon_theta(x_t, t). We have x_t and we want x0 prediction.
     alpha_bar_t = sqrt_ac[t]**2  # since sqrt_ac[t] = sqrt(\bar{\alpha}_t)  
     x0_pred = (x_t - sqrt_om[t][:, None, None, None] * pred) / (sqrt_ac[t][:, None, None, None] + 1e-8)
     ```

     Here, `sqrt_ac` and `sqrt_om` are the precomputed arrays of \$\sqrt{\bar\alpha\_t}\$ and \$\sqrt{1-\bar\alpha\_t}\$. We index them by the current `t` for each sample. After this, `x0_pred` is the model’s estimated full-dose image (in normalized \[-1,1] space) for each sample. Meanwhile, we have `nd` (the ground truth full-dose) already in the batch (also normalized to \[-1,1]).
   * **L1 Loss:** Compute `loss_l1 = F.l1_loss(x0_pred, nd)`. L1 is good for preserving overall brightness/contrast and is more robust to outliers than MSE (which we already have in noise space). This directly penalizes any residual error in the reconstructed image.
   * **LPIPS Loss:** Use a perceptual similarity metric. For example, utilize the LPIPS network (learned perceptual metric). Setup:

     * Install/Import LPIPS (from `lpips import LPIPS` or use TorchMetrics’ multi-scale SSIM/LPIPS). Initialize once outside the training loop: `lpips_fn = LPIPS(net='vgg').to(device)`【3†】. This model when called returns a distance measure between two images.
     * Before feeding images to LPIPS, ensure they are in \[0,1] range and 3-channel if required. Our images are single-channel CT slices. We can either repeat the channel to 3 (pretend it’s a gray image repeated, since LPIPS expects RGB) or use a grayscale variant if available. Simpler:

       ```python
       pred_img = (x0_pred * 0.5 + 0.5)  # to [0,1]  
       gt_img = (nd * 0.5 + 0.5)        # to [0,1]  
       pred_img_3 = pred_img.expand(-1,3,-1,-1)  # [B,3,H,W]  
       gt_img_3 = gt_img.expand(-1,3,-1,-1)  
       lpips_val = lpips_fn(pred_img_3, gt_img_3)  # returns [B,1,1,1] or [B]  
       loss_lpips = lpips_val.mean()  
       ```

       This gives a scalar perceptual loss. LPIPS will encourage the high-level features of the prediction to match the target (e.g., textures, edges as perceived by a VGG network). It’s especially useful to penalize the slight blurring that pure L2/MSE optimization can introduce.
   * **Adversarial Loss (optional, for future refinement):** Set up a discriminator network **D** to differentiate fake full-dose outputs from real ones. A suitable choice is a **PatchGAN** discriminator (commonly used in image translation tasks): a few conv layers that output a feature map of “real/fake” probabilities for patches of the image. For example:

     ```python
     class Discriminator(nn.Module):  
         def __init__(self, in_ch=1):  
             super().__init__()  
             self.model = nn.Sequential(  
                 nn.Conv2d(in_ch, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),  
                 nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  
                 nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),  
                 nn.Conv2d(256, 512, 4, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),  
                 nn.Conv2d(512, 1, 4, stride=1, padding=1)  # output patch map  
             )  
         def forward(self, img):  
             return self.model(img)  
     }  
     ```

     Initialize `D = Discriminator().to(device)` and an optimizer for it (e.g., Adam lr 1e-4). During training, for each batch:

     * Get the *predicted full-dose image* (`x0_pred`) as above. Also have the *real full-dose target* (`nd`).
     * Compute discriminator outputs: `D_fake = D(x0_pred.detach())` and `D_real = D(nd)`. (Detach `x0_pred` so the generator (diffusion model) gradients don’t flow when updating D). Each of these is a feature map (e.g., shape \[B,1,30,30] if input 256 and 4 conv downsamples).
     * Form D’s loss: a typical **LSGAN** (Least Squares GAN) loss is:

       ```python
       loss_D_real = 0.5 * F.mse_loss(D_real, torch.ones_like(D_real))  
       loss_D_fake = 0.5 * F.mse_loss(D_fake, torch.zeros_like(D_fake))  
       loss_D = loss_D_real + loss_D_fake  
       ```

       This encourages D\_real to be 1 and D\_fake to be 0. Backprop `loss_D` and update `D` optimizer.
     * **Generator adversarial loss:** Now for the diffusion model (generator), compute `D_fake_again = D(x0_pred)` (this time without detaching, so gradients flow to model). The generator wants to fool the discriminator, so we use:

       ```python
       loss_gan = F.mse_loss(D_fake_again, torch.ones_like(D_fake_again))  
       ```

       (Or if using standard GAN with BCE: use `F.binary_cross_entropy_with_logits(D_fake_again, torch.ones_like(...))` etc.) This `loss_gan` is added to the model’s loss terms.
     * We will weight this relatively small (because initially, the model might produce somewhat blurry outputs that D can easily tell apart; if the weight is too high early, the model might introduce noise or artifacts to fool D at the expense of MSE/L1). A good strategy is to start the adversarial loss after a certain number of epochs (once L1/MSE have gotten the model close to the target distribution), then fine-tune with GAN. Or start with a very low weight and increase it.
   * **Combine Losses:** Now assemble the total loss for the diffusion model (generator):

     ```python
     loss_mse = F.mse_loss(pred, noise)  # existing diffusion noise loss  
     loss = loss_mse + λ1 * loss_l1 + λ2 * loss_lpips + λ3 * loss_gan  
     ```

     Choose weights λ appropriately. For example, you might start with λ1 = 1 (equal weight to MSE), λ2 = 0.1 (perceptual loss typically has smaller scale, and we don’t want to overwhelm L1), λ3 = 1e-3 (GAN loss is usually set small to begin). These can be tuned. If not using GAN initially, set λ3 = 0.
   * **Where to Insert:** In `train_with_eval`, after computing `pred = model(...)`, currently it does `loss = F.mse_loss(pred, noise)`. Replace this with the expanded calculation: compute `x0_pred`, then the additional losses, then form `loss`. Also handle the D update each iteration (maybe even every other iteration – some GAN training schemes update D more frequently or equally). Don’t forget to zero gradients for both optimizers appropriately (`optimizer.zero_grad()` for model, and e.g. `optimizer_D.zero_grad()` for discriminator).
   * **Training Regimen:**

     * Monitor the individual loss components. The noise MSE will naturally decrease as before. L1 should also decrease and typically is on a similar scale as MSE (since both are pixel differences). LPIPS might start somewhat high and should decrease as outputs become perceptually closer. If GAN is used, keep an eye on `loss_D` and `loss_gan`. Ideally, `loss_D_real` and `loss_D_fake` stabilize around some value and `loss_gan` does not dominate. If D loss goes to zero (too easy to distinguish), the generator might be lagging; if generator’s GAN loss goes to zero, it might be overpowering D. Adjust λ3 or learning rates accordingly.
     * The combined loss should yield better results than MSE alone: L1 will reduce bias, LPIPS will encourage finer textures (less over-smoothing), and GAN (if used) will add realistic grain or edge sharpness that pure regression sometimes misses. We expect to see outputs with sharper detail and higher fidelity in perceptual metrics (e.g., check if LPIPS or SSIM on validation improves).
   * **Note:** Including LPIPS and GAN might slow training (due to heavy VGG network and extra D forward-backward). Ensure your training loop and GPU memory can handle this. You can optimize by computing LPIPS on a downscaled image (like 128×128) to save memory, or use a lightweight perceptual metric. But given medical images, preserving detail is critical, so the cost may be worth it.
   * **Optional:** You could also include **SSIM loss** or **VGG feature L2 loss** as alternatives to LPIPS. However, LPIPS (which uses multiple layers’ activations) is usually sufficient.
   * **Conclusion:** Modify the training code to implement this multi-term loss. Test on a few batches to ensure gradients flow (e.g., if LPIPS requires `requires_grad=False` for its internal weights, set that by calling `lpips_fn` in eval mode). Once integrated, you should observe faster convergence in terms of image quality: lower L1/SSIM errors and better visual similarity, at the cost of a more complex training process.

## Training Data Augmentations

In addition to model changes, we will apply several data augmentation techniques during training to make the model more robust and to simulate a variety of realistic input scenarios. These augmentations target the physics and intensity characteristics of CT images:

1. **Physics-Aware Noise Injection:**
   Simulate the effect of different dose levels by injecting synthetic noise into the images, following CT physics principles (photon noise and Beer–Lambert law). This prevents overfitting to the exact noise distribution of the training set and teaches the model to handle variations.

   * **Photon Noise Modeling:** In CT, image noise arises from the statistical nature of X-ray photon counts. A lower dose (fewer photons) yields higher noise, roughly Poisson-distributed relative to the signal. We approximate this by adding noise whose variance depends on the pixel intensity. One way: treat the normalized image as proportional to transmitted intensity and sample using Poisson. For example, if `nd` (full-dose) is our clean target in \[0,1], simulate a lower dose projection:

     1. Convert the image to a pseudo count domain: e.g., set a virtual “photon count” \$N\_0\$ for full dose. For each pixel intensity `I` (which is between 0 and 1 after min-max normalization per slice), interpret it as \$I = \exp(-\mu)\$ (assuming log scale attenuation) or directly proportional to attenuation. We can then generate noisy measurement: `I_noisy = Poisson(N0 * I) / N0`. This introduces Poisson noise (which for large N0 approximates Gaussian with variance \~I/N0).
     2. Convert back to image space if needed (in this simplistic approach, we treated intensities linearly, which is not exact HU physics, but sufficient for augmentation). If using the exponential model: simulate line integrals `L = -ln(I)` (this would be like thickness or attenuation), add Gaussian noise to `L` (since Poisson in intensity is roughly Gaussian in log domain for low variance), then exponentiate back: `I_noisy = exp(-(L + noise))`.
   * **Implementation in Dataset:** In `CTPairsDataset.__getitem__`, after loading the `ld` and `nd` images:

     ```python
     if random.random() < 0.5:  # 50% of the time, replace or augment low-dose with synthetic noise  
         # For example, further degrade the existing low-dose image or start from nd 
         arr = np.array(img_f, dtype=np.float32) / 255.0  # get ND in [0,1]  
         # Simulate quarter-dose from full-dose (if we wanted to augment beyond given LD)  
         N0 = 10000  # base photon count, hypothetical  
         I = np.clip(arr, 1e-6, 1)  # intensity  
         # Poisson noise on counts  
         counts = np.random.poisson(I * N0)  
         I_noisy = counts.astype(np.float32) / N0  
         # back to [0,1] float image  
         img_q_noisy = Image.fromarray((I_noisy*255).astype(np.uint8))  
         ld = self.transform(img_q_noisy)  # apply same transforms (crop, tensor, normalize)  
     else:  
         ld = self.transform(img_q)  # original low-dose  
     nd = self.transform(img_f)  
     return ld, nd  
     ```

     In this snippet, we took the full-dose image, attenuated it, and added Poisson noise to simulate a quarter-dose scenario. We then proceed with that as the input. You could also start from the actual `img_q` (quarter-dose) and *add even more noise* to simulate an “eight-dose” (even lower) image. For instance, add Gaussian noise with standard deviation proportional to `sqrt(img_q)` or a fixed fraction of intensity. The above uses full-dose as baseline which might produce a slightly different noise texture than the real quarter-dose (which is fine for augmentation).
   * **Inverse Beer–Lambert:** The above approach implicitly used the exponential relation in the Poisson sampling. If access to original HU values is possible (with a consistent scale), one could do: `I0 = exp(-HU * k)` for some constant k, add noise, then invert. But since we normalized per slice, our approach treats the normalized intensity as linear scale, which is an approximation.
   * **Use in Training:** This augmentation will expose the model to inputs that are noisier or differently noisy than the training low-dose images. It forces the model to rely on robust features (like edges, textures) rather than overfit to specific noise patterns. It also effectively increases the range of noise levels the model can handle (which is helpful if at test time some scans are slightly lower dose or have different noise due to scanner differences).
   * **Caveats:** Because we min-max normalize each slice, the absolute scale of noise might differ slice to slice (the normalization will somewhat squash the noise). To mitigate, you could normalize using a consistent value range across the dataset (if known) or incorporate the augmentation *before normalization*. In our code, we converted to 0-255 and PIL, so essentially we are injecting noise in the 8-bit domain. This is okay but note that a Poisson on 8-bit is not physically accurate. If possible, one could retrieve the original HU from DICOM (using `ds.RescaleIntercept` and `ds.RescaleSlope`) and do noise injection in HU, then re-normalize. If that’s too involved, the above method still adds realistic grain.
   * **Testing:** After training with this augmentation, evaluate the model on various noise levels: it should maintain performance (or degrade gracefully) on even lower-dose inputs than seen originally. One can synthesize a half-dose or eighth-dose from a validation pair and see if the model can denoise it to an acceptable level.

2. **Spectral Intensity Jitter (HU-value Augmentation):**
   This augmentation simulates variations in intensity scaling and tissue density distributions. CT images can have different intensity characteristics (due to patient size, calibration, or use of contrast agents). We introduce random intensity transformations so the model doesn’t assume a fixed relationship between input and output histograms.

   * **HU Gamma Shifts:** Apply a random gamma correction to the low-dose image (and possibly to the full-dose image for consistency). For example, choose \$\gamma \sim \mathcal{U}(0.8, 1.2)\$ and do `img = img ** γ` (if `img` is normalized to \[0,1]). A γ<1 will make darker regions relatively brighter (increasing contrast in low-intensity areas), γ>1 does the opposite. This mimics different hardness of X-rays or reconstruction kernels that change contrast.
   * **Intensity Scaling and Bias:** Randomly adjust brightness/contrast by linear scaling: `img = img * α + β`, with small α in \[0.9,1.1] and β in \[-0.05, 0.05]. This can simulate different calibration offsets or slight differences in how the scanner maps HU to pixel values. For instance, one scanner might output images a bit brighter on average for the same dose.
   * **Histogram Matching:** For a stronger augmentation, pick a reference distribution (perhaps the full-dose image’s histogram or a standard CT histogram) and adjust the low-dose image’s histogram to match it. This can be done with `skimage.exposure.match_histograms`. Alternatively, precompute a “target” CDF (maybe an average histogram of full-dose images) and randomly match some low-dose slices to it. This ensures the network sees inputs with varying tissue intensity profiles (for example, some images might appear as if the patient had different composition).
   * **Implementation in Dataset:** After loading and normalizing the images (the code currently normalizes each slice to 0-255 and then uses `transforms.Normalize(0.5,0.5)` to go to \[-1,1]). We should apply these intensity augmentations **before** that final normalization (to keep it physically meaningful). For instance:

     ```python
     # After converting to PIL or as numpy [0,255] array:
     arr_q = np.array(img_q, dtype=np.float32) / 255.0  # back to [0,1]
     arr_f = np.array(img_f, dtype=np.float32) / 255.0  
     # Gamma jitter
     γ = np.random.uniform(0.8, 1.2)
     arr_q = np.clip(arr_q ** γ, 0, 1)
     # Brightness/contrast jitter
     α = np.random.uniform(0.9, 1.1)
     β = np.random.uniform(-0.05, 0.05)
     arr_q = np.clip(arr_q * α + β, 0, 1)
     arr_f = np.clip(arr_f * α + β, 0, 1)  # (apply same α,β to full-dose for consistency)
     # (Optional) Histogram matching occasionally
     if random.random() < 0.1:  # 10% of the time
         arr_q = match_histograms(arr_q, arr_f, multichannel=False)
         # (matching LD to ND's histogram; or could use a random reference histogram)
     img_q = Image.fromarray((arr_q*255).astype(np.uint8))
     img_f = Image.fromarray((arr_f*255).astype(np.uint8))
     ld = self.transform(img_q)  # apply CenterCrop, ToTensor, Normalize
     nd = self.transform(img_f)
     ```

     In this snippet, we first normalize to \[0,1] floats, apply gamma and linear jitter to `arr_q`. We also apply the same linear scaling (α, β) to `arr_f` so that the relative difference between LD and ND remains (this keeps the task coherent: if we only altered LD, we’d essentially be changing the target mapping). We don’t apply gamma to `arr_f` in the same way because the full-dose is ground truth – one could argue to leave it unchanged. However, applying the same α,β ensures we’re not introducing a large distribution shift between input and target (except for the histogram matching case, which is a deliberate shift in LD’s distribution that the model must overcome). We use small β so that we don’t push intensities out of range significantly. The final `match_histograms` is done sparsely; it will give the model some instances where the input histogram is unusual relative to output, forcing it to rely on spatial cues (edges, etc.) rather than direct intensity correspondence.
   * **Note on HU-space:** Since each slice is min-max normalized, these augmentations operate in that normalized space, not true HU. This is acceptable for augmentation, though not exact. If we had consistent windowing (say all images clipped to \[-1000, 1500 HU] then normalized), we could do gamma in real HU units. Given per-slice normalization, our augmentation mainly simulates generic contrast changes.
   * **Impact:** This augmentation should make the model less sensitive to slight differences in input brightness/contrast. For instance, if a test scan’s low-dose images are reconstructed with a different kernel (so noise texture or contrast differs), the model trained with jitter can still handle it. It essentially teaches the model to perform **histogram normalization** as part of its job if needed.
   * **Monitoring:** Track if training loss fluctuates when applying these jitters. Typically, pixel-wise loss might increase a bit due to the added randomness, but validation on original distribution should still improve in robustness. You might also augment validation set with some jitted versions to see if the model’s performance holds up.

3. **PatchMix / MixUp Augmentation:**
   Introduce mixing of training examples to further regularize the model. This forces the model to reconstruct outputs from inputs that might be composites of different patients or scans, preventing over-reliance on exact correspondences and encouraging learning more general features. Two techniques: **MixUp** (continuous mix) and **PatchMix/CutMix** (spatial mix).

   * **MixUp (on intensities):** Take two random pairs `(ld1, nd1)` and `(ld2, nd2)` from the dataset. Create a new training pair by convex combination:
     $ld_{mix} = \lambda \cdot ld1 + (1-\lambda) \cdot ld2,$
     $nd_{mix} = \lambda \cdot nd1 + (1-\lambda) \cdot nd2,$
     where \$\lambda \sim \text{Uniform}(0.3, 0.7)\$ (to ensure a substantial contribution from both). This produces a blended low-dose image and a blended full-dose image. The rationale is that the model should still be able to output the blended full-dose given the blended low-dose (since the operation is linear, an ideal model could just do the same blend in feature space). This adds a lot of variety (any linear combination of two patients’ images is possible). It may also help the model learn a roughly linear response with respect to input intensities.

     * **Implementation:** It’s tricky to implement purely within `torchvision.transforms`, so do it in `__getitem__`:

       ```python
       if random.random() < 0.15:  # 15% of the time, perform MixUp
           # pick a different random index
           j = random.randrange(len(self.pairs))
           ld_path2, nd_path2 = self.pairs[j]
           ds_q2 = pydicom.dcmread(ld_path2); ds_f2 = pydicom.dcmread(nd_path2)
           img_q2 = process_to_uint8(ds_q2.pixel_array)  # using same steps as for img_q
           img_f2 = process_to_uint8(ds_f2.pixel_array)
           λ = np.random.uniform(0.3, 0.7)
           mix_q = cv2.addWeighted(np.array(img_q), λ, np.array(img_q2), 1-λ, 0)  # weighted blend 8-bit images
           mix_f = cv2.addWeighted(np.array(img_f), λ, np.array(img_f2), 1-λ, 0)
           img_q_mix = Image.fromarray(mix_q.astype(np.uint8))
           img_f_mix = Image.fromarray(mix_f.astype(np.uint8))
           ld = self.transform(img_q_mix)
           nd = self.transform(img_f_mix)
           return ld, nd
       ```

       Here `process_to_uint8` stands for the steps that convert DICOM pixel\_array to an 8-bit PIL image (min-max normalize and scale) similarly to what is done for the original images. We then blend using OpenCV `addWeighted` (or numpy interpolation). We replace the normal return with this mixed pair. We choose a modest probability (e.g., 10-20%) so most batches still see real pairs.
   * **PatchMix / CutMix:** Instead of a linear blend over the whole image, we splice images regionally. This addresses that different regions could come from different patients and the model should handle discontinuities. For example, randomly take a square patch from `ld2` and insert it into `ld1`, and do the same for `nd2` into `nd1` at the same location. The model then gets an input where, say, the top-left quadrant is from one scan and the rest from another, but it must produce an output that has the corresponding full-dose quadrant seamlessly. This teaches the model locality – it should rely on the local low-dose to local full-dose mapping even if surrounding context differs.

     * **Implementation:** Also in `__getitem__`, perhaps with a separate probability (e.g., 15% MixUp, 15% CutMix):

       ```python
       if random.random() < 0.15:  # perform PatchMix
           j = random.randrange(len(self.pairs))
           # load second sample (similar to above)
           # ... (get img_q2, img_f2 as uint8)
           H, W = img_q.size
           ph, pw = H//2, W//2  # for example, half-size patch
           y = random.randint(0, H-ph); x = random.randint(0, W-pw)
           patch_q2 = np.array(img_q2)[y:y+ph, x:x+pw]
           patch_f2 = np.array(img_f2)[y:y+ph, x:x+pw]
           arr_q = np.array(img_q); arr_f = np.array(img_f)
           arr_q[y:y+ph, x:x+pw] = patch_q2
           arr_f[y:y+ph, x:x+pw] = patch_f2
           img_q_pm = Image.fromarray(arr_q.astype(np.uint8))
           img_f_pm = Image.fromarray(arr_f.astype(np.uint8))
           ld = self.transform(img_q_pm); nd = self.transform(img_f_pm)
           return ld, nd
       ```

       We randomly choose a patch location and size (here for simplicity half the image, but you could randomize size too). We replace that region in the first image with the corresponding region from the second. We must ensure to do the same replacement in the full-dose images so the pair stays aligned (ld->nd mapping holds piecewise). The boundaries of the patch will be abrupt (one person’s anatomy to another), which is intentionally challenging. The model, however, only needs to produce the exactly corresponding ND patch in that area (and it might be allowed to produce a seam, since that’s what a composite of two ND images would look like). Over many random PatchMix operations, the model learns not to be thrown off by unusual artefacts or foreign structures in the input – it will treat each region based on its local conditioning.
   * **Mixing in HU-space:** Both MixUp and PatchMix as described operate on the 8-bit normalized images (which is effectively HU-space normalized per slice). That’s acceptable. If we had actual HU arrays, we could mix those directly (taking care to mix the raw values then do the transform). The main point is the model sees linear or piecewise-linear combinations that it wouldn’t see normally.
   * **Training and Effect:** Use these augmentations intermittently. You might want to ensure that not *every* batch is augmented, to still provide the model with real, coherent data frequently. A combined strategy: 70% normal pairs, 15% MixUp, 15% PatchMix (these can be adjusted). Monitor training – excessive mixing might make training slower to converge (because the target mapping is now more abstract – a mix of two CT images, which the model might average in output). However, after convergence, such a model is usually more robust. It may also help against any slight misalignment issues or outliers in real data.
   * **Validation:** Usually, you do not apply these augmentations on validation data (validate on true pairs). But you can test the model by feeding it a mixed input (just to see if it does something reasonable like output the corresponding mixed output). If everything is consistent, the output of a mixed input should be the same blend of outputs – this can be a sanity check for linearity.
   * **Alternate Approaches:** Instead of mixing at pixel level, one could do feature-space MixUp (but that’s complex to implement without modifying model training loop). The pixel-space mix is straightforward and effective for data augmentation.

By integrating these augmentations, we broaden the training distribution. The model will learn the core task (denoising low-dose to full-dose) in a way that’s invariant to certain changes in intensity scaling and robust to unfamiliar noise patterns or partial inputs. It’s essentially a form of regularization. Expect possibly a slightly higher training loss early on (due to the harder task), but a better generalization and possibly improved performance on real test scans that might not exactly match training stats.

## Future Extension: Self-Supervised Masked HU-Prediction Pretraining

For a further boost, especially if labeled data (paired low/full dose) is limited, we can pretrain the UNet encoder on a **self-supervised task** using all available CT images. The proposed task is masked intensity (HU) prediction, analogous to masked image modeling (like MAE or denoising autoencoders). The goal is to learn general features of CT scans (edges, textures, organ structures) without needing the full-dose target, and use those features to initialize the diffusion model.

* **Pretraining Task:** Randomly mask out portions of a CT image and train a network to reconstruct those missing parts. This teaches the network to understand context – e.g., if part of an image is missing, the network must use surrounding structures to predict it. For CT, predicting masked regions requires understanding typical anatomy and noise patterns, which is useful for denoising later.
* **Data:** Use the low-dose images from the training set (or even additional unlabeled low-dose images if available) since they are abundant. You could also include full-dose images in this pretraining (since the task doesn’t require pairing), but typically low-dose alone is fine. We assume we have `CTPairsDataset`, so we have `quarter_...` images accessible. We can create a dataset that returns just one image at a time (the input for self-supervision). For example:

  ```python
  class CTImagesDataset(Dataset):
      def __init__(self, list_of_dicom_paths, transform):
          self.paths = list_of_dicom_paths  # could reuse CTPairsDataset.pairs and take first of each, or gather all LD paths
          self.transform = transform
      def __len__(self): return len(self.paths)
      def __getitem__(self, idx):
          ds = pydicom.dcmread(self.paths[idx])
          arr = ds.pixel_array.astype(np.float32)
          arr = (arr - arr.min())/(arr.max()-arr.min()+1e-5)
          img = Image.fromarray((arr*255).astype('uint8'))
          img_t = self.transform(img)  # [1,H,W] normalized [-1,1]
          return img_t
  }
  ```

  This dataset yields normalized single-channel images. We will apply masking on these tensors during training.
* **Model:** We can repurpose the diffusion UNet as an inpainting autoencoder for this task. There are two possible setups:
  **Setup A: Use the same architecture, predict missing pixels directly.**

  * Remove time conditioning (since this isn’t diffusion). E.g., you can fix `t_emb = 0` or remove time embedding layers. Simplest is to modify `PixelDiffusionUNetConditional` to accept just `x` (a single image) and use `cond` input to carry something like a mask if needed. However, it might be easier to create a new model instance for pretraining to avoid confusion.
  * You can create a variant: `PixelUNetMasked` which is essentially the same UNet but instead of concatenating `x_t` and `cond`, it takes a single input (the masked image, plus an extra channel for mask). For instance, `init_conv = nn.Conv2d(2, base_ch,3,1,1)` where the 2 channels are \[masked\_image, mask]. The mask channel is 1 where pixel is present, 0 where it’s masked out. This tells the network which areas to fill in. Alternatively, fill masked areas with a sentinel value (like 0 or noise) and let the network figure it out; in that case, you can still provide the mask as a channel to explicitly inform it.
  * The downsampling, upsampling, etc., remain the same (maybe remove the attention blocks or keep them; they can still be useful for global context in filling). At the end, use `self.final` to output 1 channel which should reconstruct the missing pixels (and reconstruct even the non-missing ones, effectively trying to output the full image). Use an L1 or L2 loss on the reconstructed image versus the original. Only compute the loss on the masked regions (since predicting unmasked regions is trivial copying). This can be done by multiplying the loss by the mask complement as weights.
  * For example, if `mask` is 1 for observed, 0 for missing, define `loss_recon = L1(output * (1-mask), original * (1-mask))`. Or equivalently, set pixels where mask=1 in original to 0 (so the loss there is zero).
    **Setup B: Denoising Diffusion as pretraining (masked noise):**
  * Another approach is to use a Denoising Autoencoder idea: give the model an image with artificial noise (e.g., masked or corrupted), and have it predict the original image, using a **simplified diffusion objective** (like just one timestep of diffusion). This is similar to how diffusion models are trained for denoising. For example, randomly mask 25% of pixels (set them to 0 or random noise) and let the diffusion model at timestep t=1 predict the clean image. In fact, our diffusion model itself could be trained for one step in this manner: feed `x_t = masked_image`, `t=1`, `cond=0` (no cond) and train it to output noise that would transform masked\_image to original. However, this might be overly complicated to set up vs a direct reconstruction loss.
  * Simpler: Just do Setup A with straightforward L1 loss.
* **Training Procedure:**

  * Determine mask strategy: e.g., random block-out of 16×16 patches (to simulate large missing areas) or random pixel mask at some percentage. A common heuristic is to mask \~25-50% of the image area. For CT, perhaps mask in square chunks (since contiguous regions missing is realistic, e.g., simulate metal artifact removal or something). You can vary the mask per image each iteration.
  * During training, for each image from `CTImagesDataset`: create a mask tensor of shape \[1,H,W] with 0/1. Apply it to the image: `masked_img = img * mask` (and if using zeros for masked, that’s fine because images are normalized around \[-1,1], so zero is a meaningful value but not too far from typical range; alternatively, set masked regions to random noise in \[-1,1] to mimic diffuse noise). Provide `mask` as input channel if your model expects it.
  * The model then outputs a reconstructed image. Use loss on the masked regions as mentioned. This is essentially training a context encoder.
  * Train for several epochs over all images. Monitor the loss on a held-out set of images (should be decreasing). You can also visually inspect: take a masked image and see if the network fills it in plausibly. It might not recover exact details (since ambiguous), but it should learn to produce realistic structures and intensities consistent with surrounding context.
  * Because our ultimate goal is to initialize the diffusion model’s encoder, focus on encoder quality: the encoder (downsampling path) should learn filters that detect edges, textures, etc., to be able to fill in gaps. The decoder/up-path learns to generate pixels given those features – some of that will also be useful for denoising.
* **Transfer to Diffusion Model:**

  * After self-supervised pretraining, we will copy the weights from the pretrained network to our diffusion UNet. Specifically, copy the weights of **encoder layers (down blocks)** and perhaps the **bottleneck layers**. The encoder of the diffusion model (Resnet blocks and attention in `downs` and `mid`) can be replaced with the pretrained weights. If the architectures match exactly, this is straightforward: e.g., both have `ResnetBlockTime` layers. In pretraining, we might not have used time embedding, but those ResNet blocks still had normalization and convolution weights that we trained. We can map those over to the diffusion model’s Resnet blocks (they have an extra time embedding bias term perhaps; if so, we can still initialize the rest).
  * **Skip connections:** The pretrained model was basically an autoencoder; its skip connections helped in reconstruction. In the diffusion model, those skip connections exist too. We will also copy the decoder (up-sampling path) weights if possible. However, note: if the diffusion model’s `init_conv` expects 2 channels (x\_t + cond), whereas the pretraining model’s first conv expected 2 channels (masked image + mask). They are conceptually similar: both have 2 input channels. We can copy those weights as well – the “mask” channel in pretraining corresponds to “cond” in the diffusion model’s input, in terms of array shape. This is not a perfect correspondence (mask vs low-dose image), but if we consider that both are providing auxiliary info to the network, it’s not unreasonable to use the same weights as a starting point. Alternatively, you could just initialize the diffusion `init_conv` from scratch or copy only the part corresponding to the image channel and random init for the cond channel. It might not make a big difference.
  * If you included attention layers in pretraining (not necessary, but if so), copy those weights into the diffusion model’s attention blocks as well. The positional embeddings or time embeddings were not used in pretraining – you can keep the sinusoidal time embedding initialization as is in diffusion model. That’s fine because time embedding is a different concept the model will learn during diffusion training.
  * **Freezing or not:** Generally, you **do not freeze** the weights during diffusion training – you let them fine-tune. But the advantage is you can use a smaller learning rate for those layers initially to prevent catastrophic forgetting of good features. For example, set a lower lr for encoder blocks and a higher lr for newly added components (like time embedding, ICR head, etc.). This can be done with parameter groups in the optimizer. Monitor if the pretrained features indeed give a jumpstart: you might observe lower initial reconstruction loss or faster SSIM improvement in early epochs compared to training from scratch.
* **Optional Variant:** Instead of masked regions, one can pretrain with a **denoising task**: add Gaussian noise to a low-dose image and train a network to denoise it (without using the paired full-dose, just making the input look like a worse version and output the clean). This is like an “auto-denoiser”. But since we have the diffusion model for denoising paired data, the masked filling is more about learning context and feature extraction. You could even do both in sequence.
* **Outcome:** After this pretraining and weight transfer, the diffusion model’s encoder should recognize edges, textures, etc., from the low-dose input more readily. This often leads to needing fewer epochs of diffusion training to converge, and can improve performance when labeled data is limited. It’s especially useful if the UNet is very deep; giving it a good initialization avoids the need to learn basic visual features from random initialization.
* **Integration Note:** This pretraining can be done as a separate training script before the main training. It doesn’t require the diffusion scheduler or any complex guidance – just standard supervised training (on the self-supervised labels which are the images themselves). The result is a checkpoint of the pretrained model. Then, when constructing `PixelDiffusionUNetConditional`, load those weights where applicable. If using PyTorch, one can load the state dict and manually assign parts: e.g.,

  ```python
  diffusion_model = PixelDiffusionUNetConditional(...)  
  pretrained = torch.load('pretrained_masked_autoencoder.pth')  
  # assume pretrained model had similar structure names for conv layers  
  diffusion_model.init_conv.weight.data = pretrained.init_conv.weight.data  # etc.  
  ```

  Do this for all corresponding layers (down blocks conv and norm weights, mid blocks, up blocks if desired). Exclude any layers that don’t have equivalents (e.g., time embedding, or if channel counts differ due to concat differences – but in our plan they’re the same aside from input).
* **Final remark:** This step is optional and can be done if time permits. Even without it, the above core and augmentation steps will significantly enhance the model. If done, it should be the first stage: complete pretraining, then integrate with all the above modifications, then train on paired data. Ensure that during the paired training, all new modules (edge encoders, cross-attn, ICR) are properly initialized (either random or some logical init) since pretraining covered primarily the encoder/decoder of the base UNet.

## Practical Sequencing of Implementation

To minimize disruption and allow staged improvements, it’s best to introduce these changes one or few at a time, verifying each before adding more:

**Stage 1: Multi-Scale Edge Prompts & GFM** – Start by integrating the edge conditioning pipeline. Modify the dataset to output edge maps or compute in `forward`. Add GFM modules to the UNet and ensure the model can still run (check tensor shapes at skip connections). Train the model with the same hyperparameters as before (MSE loss only, no cross-attn yet) for a few epochs. You should see that it trains similarly or faster, and qualitatively, outputs should have better edge preservation (compare with a baseline if available). This confirms edges are being used. Edge detector and GFM can be refined (try simple Canny vs PiDiNet to see differences). Once confident, keep this feature on.

**Stage 2: Loss Function Enhancements** – Before adding new architectural complexity, improve the loss, as it’s relatively isolated. Incorporate L1 and LPIPS into the training loop. You might hold off on GAN until later (because GAN can introduce instability – you want a reasonably good model first). Train the model (with edges) using MSE+L1+LPIPS from scratch. Monitor convergence: typically L1 helps convergence speed. Check that LPIPS doesn’t cause any weird artifacts (if its weight is small it should be fine). If you have a previous model for baseline, check that metrics like SSIM or RMSE improve with the new loss. This stage ensures the model is optimizing for the right objectives. If you face difficulty balancing losses, adjust weights or even remove LPIPS early on, then add it in a fine-tuning phase. The point is to have the loss framework ready for when the architecture becomes more complex (so we’re always training with the end-goal losses).

**Stage 3: Global Cross-Attention** – Implement and plug in the cross-attention (at least at bottleneck). This touches the forward pass and model initialization. After adding, load the last checkpoint from Stage 2 as a starting point (if available) – the new attention weights will be random but others are trained. Train further and see if validation improves. Cross-attention should help especially if the input has some global features (like overall contrast or some structure that spans wide area). It might be a subtle improvement; ensure it doesn’t degrade performance. If training becomes unstable (loss spikes), you might need to reduce learning rate for the attention layer or use gradient clipping. If all is well, you can experiment with adding cross-attn to one of the decoder levels too (e.g., 64×64) and see if any gain – but do this one at a time, as each adds overhead.

**Stage 4: ICR Head** – Add the ICR refinement head to the model. You can either train from scratch with it or attach it to an existing trained model for fine-tuning. A safe approach: take the Stage 3 model (edges + new loss + cross-attn) as baseline. Add ICR head (initialized small, e.g., final conv weights to zero so it starts as identity). Then resume training. Initially, keep the main UNet weights fixed for a few epochs and train only the ICR head (set requires\_grad=False for all except ICR head) – this makes the head learn the residual mapping without affecting the rest. After it stabilizes (monitor validation L1/SSIM – they should improve as head fixes small errors), unfreeze and train all together to fine-tune. The improvement from ICR might be seen in metrics like a slight drop in MAE or a boost in high-frequency detail (visually). It essentially gives the model an easier way to correct structural biases. If you find the head is not learning (maybe because the UNet is already doing well), you can increase its capacity or learning rate. Ensure the coordinate channels are correctly implemented – e.g., verify that the head’s output actually changes if you shift the image in the frame (it should, because coordinates changed).

**Stage 5: Schrödinger Bridge Diffusion** – This is a more experimental feature. Once the supervised model is strong (after Stage 4), you can attempt to train with the diffusion bridging. It might help to first implement the inference change (starting from noised LD) and see how the current model behaves. Likely it will produce reasonable results already. To integrate into training, modify `forward_diffusion_sample` or directly the training loop as discussed (Method 1: occasionally use LD as x\_t). You might slowly increase the frequency of this bridging training. For example, in epoch 1, 0% of batches use the LD-as-x\_t trick; epoch 2, 10% of batches; epoch 3, 20%, up to perhaps 50%. This way the model gradually adapts to the new forward process. Keep an eye on loss when you introduce this – if it jumps or diverges, the model might be confused by the new task. In that case, lower the learning rate or introduce the change even more gradually. Over time, the model will learn to handle input states that are more like the actual LD image earlier in the denoising process. You can then sample with fewer steps or with the hybrid schedule. Evaluate with metrics and visually – ideally, with bridging, you might be able to use say 50 diffusion steps starting from LD and get comparable quality to 100 steps starting from noise (faster inference). Also, check that the model doesn’t overfit to just copying LD (the inclusion of noise in those bridging steps and the fact that final target is ND should prevent that).

**Stage 6: Adversarial Fine-tuning (if used)** – Once all other parts are working and the model outputs are close to ground truth, you can introduce the GAN loss to further refine realism. It’s often best to do this as a short fine-tuning at the end. Set up the discriminator and train for maybe a few epochs (or until you see GAN loss equilibrium) with a low weight on adv. The effect might be subtle: look for slightly sharper textures or noise that mimics the real full-dose grain (which could improve perceptual scores but maybe not SSIM). Be careful to save intermediate models – GAN can sometimes destabilize and produce artifacts if over-trained. If it starts to do so, you can roll back or lower λ3. If the dataset is small, GAN might overfit the discriminator; using a validation set to monitor perceptual quality or using patience to stop is important.

**Stage 7: Data Augmentations** – Data aug can actually be introduced at any earlier stage, but to isolate issues, you might first get the model working with core changes on unaugmented data. Once stable, start adding augmentations:

* Add **spectral jitter and noise injection** first (these are unlikely to cause training divergence). See if the training loss curve shifts (it might increase slightly due to harder input, but validation on original distribution should remain good). Ideally, check that after augmentation, the model doesn’t overly smooth outputs: since it has to handle varied inputs, it might become more conservative. If that happens, you might increase model capacity or training time. Generally, though, it will just learn a more diverse mapping.
* Then add **MixUp/PatchMix**. These can cause the model’s training loss to not be strictly comparable to before (since now some targets are weird mixes). It’s fine – focus on validation performance. Ensure that MixUp is not used so often that the model is always trying to reproduce blended images (which could hurt quality on real images). A small percentage is enough. You could alternate epochs with and without mixing to balance. Monitor validation SSIM/LPIPS: they should not degrade; if they do, maybe the network is struggling with the augmented task – reduce the frequency. In our experience, a moderate amount of mixing helps generalization without harming in-domain performance. It may especially help if your training dataset is small (the model sees effectively more variety).
* One augmentation to be cautious with is histogram matching: if overdone, the model might learn to ignore absolute intensity and only focus on edges, which might be okay or might reduce quantitative accuracy. Use it sparingly (or only on input not on output). If unsure, you can omit explicit histogram matching and rely on random gamma/contrast which are simpler.
* As you incorporate these, you can extend training for more epochs since the model is seeing “new” data each epoch virtually. The loss might plateau slower. Use typical early stopping or set a higher epoch count. Possibly lower learning rate later since augmentations effectively add noise to training.

**Stage 8: Self-Supervised Pretraining** – If you choose to do this, it would actually be Stage 0 in timeline (before everything). The sequence would be: pretrain the encoder on masked prediction, load weights, then proceed with Stage 1 and onwards but starting from the pretrained weights instead of random. The benefit of doing it at the very beginning is maximum influence on training. However, you can also do a variant: after Stage 2 or 3, realize the model could still be better – then do a masked pretraining on the side and use it to re-initialize and train again. That is more time-consuming because you’d redo supervised training. If possible, plan to do the pretraining upfront. The nice thing is it can be done on the same data pipeline (just ignoring the pairing). Even if you have already started training the main model, you could still incorporate a round of self-supervised learning to further fine-tune the encoder: e.g., freeze decoder, run masked training on the encoder in the context of the full model (like training it to reconstruct input with itself). But that is complex; better to pretrain standalone.

* When transferring, verify that the layers indeed got the weights (maybe print a checksum or small snippet of weight before/after). After integrating, the first epoch of diffusion training should show much lower initial loss than a scratch model would – a sign that pretraining helped (e.g., if SSIM starts higher).
* Continue with the rest as usual but possibly you can train fewer epochs to reach target performance since you started closer.

Finally, throughout this process, maintain **evaluation checks** after each major change: e.g., after adding edges, after changing loss, etc., evaluate on a fixed validation set (maybe using SSIM, PSNR, LPIPS, and visual inspection). This will ensure none of the changes unexpectedly worsened the model. If something regresses, you can revert or fix that component before adding more. Using a version control for model code and training scripts is advisable so you can track these incremental changes.

By following these steps, you will incrementally build up the model to have all the requested features. Each stage’s improvements (edge guidance, better conditioning, refined losses, and robust training) should stack to produce a final model that works in pixel space, leverages multi-scale structure from edges, global context via attention, and produces high-quality, artifact-free outputs thanks to advanced conditioning and training strategies. This staged approach reduces risk: if a bug or issue arises, it’s easier to pinpoint at which stage. Good luck with implementation and training!
