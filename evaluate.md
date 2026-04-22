 # 评估 16帧 Residual 3D 模型
  bash evaluate_3d.sh \
      --num_frames 16 \
      --checkpoint ./checkpoints/3d_residual_16f_4gpu/best/trainable_weights.pt \
      --num_processes 8

  # 对比 Baseline 16帧
  num_frames=16 bash evaluate_all_in_one.sh \
      --model llava_one_vision_qwen2_0p5b_ov_16f \
      --num_processes 8

  # 对比 Baseline 32帧
  bash evaluate_all_in_one.sh \
      --model llava_one_vision_qwen2_0p5b_ov_32f \
      --num_processes 8